"""Measured-reading interpretation of a saved cross-batch comparison plan.

``POST /api/v1/certificates/cross-batch-readings`` takes the saved
``plan_id`` (produced by ``POST /api/v1/certificates/cross-batch-plans``),
an idempotency ``request_key``, the baseline ``batch_id``, the **complete**
scan of the target plate (empty wells omitted), the per-well measured
readings and the maximum allowed absolute offset.

State is re-confirmed inside one SQLite transaction **before** the
idempotency lookup: every source certificate must still exist and be ACTIVE
at the revision the plan pinned, that revision's immutable snapshot digest
must be unchanged, and the pinned (batch, sample) members must still be
attributable to that snapshot (the sample must still occupy its frozen
planned well there and carry the source reading). Any missing certificate
is 404; any revocation, revision advance, digest change or attribution
mismatch is 409 — consequently a recorded verdict can never be replayed
once a source certificate was revoked or its revision changed.

Content checks (422, nothing saved, recorded under the request key like
other content responses): the baseline batch must participate in the plan;
the full scan must agree well by well with the plan layout (the same
reconciliation rules as release: missing/unplanned/duplicated samples,
misplacement, blocked-well occupancy, malformed wells); every planned well
must carry exactly one finite reading and no reading may name another well.

On a valid request every comparison is judged pairwise against the
baseline batch member: the source-certificate reading difference
(``r_other - r_baseline`` from the frozen snapshots), the target-plate
reading difference (same sign) and the absolute offset
``|target_delta - source_delta|`` are computed. A pair passes iff the
offset is at most ``allowed_offset``; the overall verdict is PASS only when
every pair passes, otherwise FAIL.

The complete scan, readings, confirmed source versions and per-pair
results are persisted together with the verdict and the idempotency record
in one SQLite transaction. Same key + same content replays the recorded
response verbatim; same key + different content is 409.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .certificates import (
    STATUS_ACTIVE,
    STATUS_REVOKED,
    _content_hash,
    _get_certificate,
    _idem_record,
    _idem_replay,
    _sha256,
    _utcnow,
    get_store,
)
from .plates import parse_well, well_name
from .planner import InfeasibleArrangement, _resolve_dimensions
from .reconciler import ReconcileInvalid, _decompose, _preflight
from .schemas import (
    CERTIFICATE_DIGEST_MISMATCH,
    CERTIFICATE_NOT_FOUND,
    CERTIFICATE_REVISION_MISMATCH,
    CERTIFICATE_REVOKED,
    CROSS_BATCH_PLAN_NOT_FOUND,
    CROSS_BATCH_SAMPLE_ATTRIBUTION_MISMATCH,
    VIOLATION_CLOSED_LOOP_MISPLACEMENT,
    VIOLATION_SCAN_LAYOUT_MISMATCH,
    VIOLATION_WELL_REFERENCE_INVALID,
    VIOLATION_XREAD_BASELINE_UNKNOWN,
    VIOLATION_XREAD_READING_DUPLICATED,
    VIOLATION_XREAD_READING_EXTRA,
    VIOLATION_XREAD_READING_MISSING,
    CrossBatchReadingRequest,
    PlateSpec,
    ReconcileRequest,
)

_ENDPOINT_XREAD = "POST /api/v1/certificates/cross-batch-readings"

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"


class CrossBatchReadingInvalid(Exception):
    """The scan/readings do not satisfy the saved plan (HTTP 422).

    No verdict row or idempotency record is written by the pure validator;
    the endpoint transaction records the 422 under the request key.
    """

    def __init__(self, categories: list[str], messages: list[str],
                 misplaced_wells: list[dict] | None = None,
                 cycles: list[list[str]] | None = None):
        self.categories = categories
        self.messages = messages
        self.misplaced_wells = misplaced_wells or []
        self.cycles = cycles or []
        super().__init__("; ".join(messages) if messages else "invalid cross-batch reading request")


def _invalid_body(exc: CrossBatchReadingInvalid) -> dict:
    """Canonical 422 envelope for a scan/reading that cannot be interpreted."""
    return {
        "detail": "The target-plate scan/readings cannot be interpreted against "
        "the saved cross-batch comparison plan; no verdict was saved.",
        "infeasibility_categories": exc.categories,
        "messages": exc.messages,
        "misplaced_wells": exc.misplaced_wells,
        "cycles": exc.cycles,
    }


@dataclass(frozen=True)
class _Member:
    """One plan member resolved against a current source snapshot."""

    batch_id: str
    sample_id: str
    group: str
    planned_well: str       # the member's well on the target comparison plate
    source_well: str        # the sample's planned well on its source plate
    source_reading: float
    revision: int
    digest: str
    certificate_id: str


# --------------------------------------------------------------- state check


def _check_sources(conn, plan_body: dict) -> tuple[dict[str, dict], dict[str, dict], tuple[int, dict] | None]:
    """Load every source certificate + pinned revision snapshot.

    Returns ``(certs, revisions, state_error)``: on a state-dependent refusal
    ``state_error`` is the ``(status, body)`` pair (404/409, not recorded under
    the request key) and the other mappings are empty.
    """
    certs: dict[str, dict] = {}
    revisions: dict[str, dict] = {}
    for src in plan_body["batches"]:
        bid = src["batch_id"]
        cert = _get_certificate(conn, bid)
        if cert is None:
            return {}, {}, (404, {
                "detail": "a source certificate of the plan no longer exists",
                "error": CERTIFICATE_NOT_FOUND,
                "plan_id": plan_body["plan_id"],
                "batch_id": bid,
            })
        if cert["status"] == STATUS_REVOKED:
            return {}, {}, (409, {
                "detail": "a source certificate of the plan is REVOKED; a revoked "
                "certificate can never be restored and the saved plan can no "
                "longer be interpreted, nor can an old verdict be replayed",
                "error": CERTIFICATE_REVOKED,
                "plan_id": plan_body["plan_id"],
                "certificate_id": cert["certificate_id"],
                "batch_id": bid,
                "status": STATUS_REVOKED,
                "revoked_reason": cert["revoked_reason"],
            })
        pinned_revision = src["revision"]
        if cert["revision"] != pinned_revision:
            return {}, {}, (409, {
                "detail": "a source certificate of the plan advanced to a newer "
                "revision; interpretations are confirmed only against the plan's "
                "pinned current revision and old verdicts can no longer be replayed",
                "error": CERTIFICATE_REVISION_MISMATCH,
                "plan_id": plan_body["plan_id"],
                "certificate_id": cert["certificate_id"],
                "batch_id": bid,
                "current_revision": cert["revision"],
                "expected_revision": pinned_revision,
            })
        row = conn.execute(
            "SELECT digest, snapshot FROM certificate_revisions "
            "WHERE batch_id = ? AND revision = ?",
            (bid, pinned_revision),
        ).fetchone()
        if row is None:  # pragma: no cover - invariant guard
            raise RuntimeError(
                f"revision {pinned_revision} snapshot missing for batch {bid!r}"
            )
        if row["digest"] != src["digest"]:
            return {}, {}, (409, {
                "detail": "the digest of a source revision snapshot no longer "
                "matches the digest pinned by the plan; the evidence the plan "
                "was confirmed against has changed and no verdict is produced",
                "error": CERTIFICATE_DIGEST_MISMATCH,
                "plan_id": plan_body["plan_id"],
                "certificate_id": cert["certificate_id"],
                "batch_id": bid,
                "revision": pinned_revision,
                "pinned_digest": src["digest"],
                "current_digest": row["digest"],
            })
        certs[bid] = cert
        revisions[bid] = {
            "revision": pinned_revision,
            "digest": row["digest"],
            "snapshot": json.loads(row["snapshot"]),
        }
    return certs, revisions, None


def _resolve_members(plan_body: dict, revisions: dict[str, dict]) -> tuple[dict[tuple[str, str], _Member], tuple[int, dict] | None]:
    """Resolve every plan member from its source snapshot, re-checking
    attribution: the sample must still occupy its frozen planned source well
    in that revision's scan and carry the frozen reading.
    """
    members: dict[tuple[str, str], _Member] = {}
    for comp in plan_body["comparisons"]:
        for entry in comp["members"]:
            bid, sid = entry["batch_id"], entry["sample_id"]
            key = (bid, sid)
            if key in members:
                continue
            snap = revisions[bid]["snapshot"]
            rows = snap["plate"]["rows"]
            cols = snap["plate"]["columns"]
            target = snap.get("target", {})
            scan = snap.get("scan", {})
            readings = snap.get("readings", {})
            raw_source_well = target.get(sid)
            source_pos = None
            source_well = None
            if raw_source_well is not None:
                try:
                    source_pos = parse_well(raw_source_well, rows, cols)
                    source_well = well_name(*source_pos)
                except ValueError:
                    source_well = None
            # The frozen scan must still place the sample at exactly its
            # planned source well (no empty/replaced well, no second
            # occurrence), and the frozen reading must stay finite.
            samples_at_source_well = {
                s for w, s in scan.items() if _eq_well(w, source_well)
            }
            reading = readings.get(source_well) if source_well is not None else None
            attributable = (
                source_pos is not None
                and scan.get(source_well) == sid
                and samples_at_source_well == {sid}
                and isinstance(reading, (int, float))
                and not isinstance(reading, bool)
            )
            if not attributable:
                return None, (409, {
                    "detail": "a member sample can no longer be attributed to its "
                    "source revision snapshot: it must still occupy its planned "
                    "well in the frozen scan with the frozen finite reading",
                    "error": CROSS_BATCH_SAMPLE_ATTRIBUTION_MISMATCH,
                    "plan_id": plan_body["plan_id"],
                    "batch_id": bid,
                    "sample_id": sid,
                    "revision": revisions[bid]["revision"],
                    "digest": revisions[bid]["digest"],
                })
            members[key] = _Member(
                batch_id=bid,
                sample_id=sid,
                group=entry["group"],
                planned_well=entry["well"],
                source_well=source_well,
                source_reading=float(reading),
                revision=revisions[bid]["revision"],
                digest=revisions[bid]["digest"],
                certificate_id=next(
                    b["certificate_id"]
                    for b in plan_body["batches"] if b["batch_id"] == bid
                ),
            )
    return members, None


def _eq_well(raw: str, canonical: str | None) -> bool:
    """Case-insensitive well equality used while reading frozen snapshots."""
    if canonical is None:
        return False
    return raw.strip().upper() == canonical.strip().upper()


# -------------------------------------------------------------- content check


def _best_effort_misplaced(target: dict[str, str], scan: dict[str, str],
                           rows: int, cols: int) -> list[dict]:
    """Misplaced-well list reconstructed even when preflight rejected inputs."""
    need: dict[tuple[int, int], str] = {}
    for sid, raw in target.items():
        try:
            need[parse_well(raw, rows, cols)] = sid
        except ValueError:
            continue
    scanned: dict[tuple[int, int], str] = {}
    for raw, sid in scan.items():
        try:
            scanned[parse_well(raw, rows, cols)] = sid
        except ValueError:
            continue
    return [
        {"well": well_name(*pos), "expected": need.get(pos), "actual": scanned.get(pos)}
        for pos in sorted(set(need) | set(scanned))
        if need.get(pos) != scanned.get(pos)
    ]


def _check_scan(req: CrossBatchReadingRequest, plan_body: dict,
                rows: int, cols: int) -> None:
    """The complete scan must agree well by well with the saved plan layout."""
    target = dict(plan_body["layout"])
    synth = ReconcileRequest(
        plate=PlateSpec(
            type=plan_body["plate"]["type"],
            rows=plan_body["plate"]["rows"],
            columns=plan_body["plate"]["columns"],
        ),
        target=target,
        scan=dict(req.scan),
        blocked_wells=list(plan_body["plate"].get("blocked_wells", [])),
    )
    try:
        _blocked, target_pos, scan_pos = _preflight(synth, rows, cols)
    except ReconcileInvalid as exc:
        raise CrossBatchReadingInvalid(
            exc.categories, exc.messages,
            misplaced_wells=_best_effort_misplaced(target, dict(req.scan), rows, cols),
        ) from exc

    need, _chains, cycles = _decompose(target_pos, scan_pos)
    misplaced = [
        {"well": well_name(*pos), "expected": need.get(pos), "actual": scan_pos.get(pos)}
        for pos in sorted(set(need) | set(scan_pos))
        if need.get(pos) != scan_pos.get(pos)
    ]
    if not misplaced:
        return

    categories: list[str] = []
    messages: list[str] = []
    if cycles:
        categories.append(VIOLATION_CLOSED_LOOP_MISPLACEMENT)
        messages.append(
            f"scan contains {len(cycles)} closed-loop misplacement cycle(s): "
            + "; ".join(" -> ".join(well_name(*w) for w in cyc) for cyc in cycles)
        )
    categories.append(VIOLATION_SCAN_LAYOUT_MISMATCH)
    messages.append(
        f"scan disagrees with the saved plan layout at {len(misplaced)} well(s)"
    )
    raise CrossBatchReadingInvalid(
        categories, messages,
        misplaced_wells=misplaced,
        cycles=[[well_name(*w) for w in cyc] for cyc in cycles],
    )


def _check_readings(req: CrossBatchReadingRequest, plan_body: dict,
                    rows: int, cols: int) -> dict[str, float]:
    """Exactly one finite reading per planned well; no other well may appear."""
    categories: list[str] = []
    messages: list[str] = []

    def add(category: str, msg: str) -> None:
        if category not in categories:
            categories.append(category)
        messages.append(msg)

    planned: dict[tuple[int, int], str] = {}
    for qsid, raw_well in plan_body["layout"].items():
        pos = parse_well(raw_well, rows, cols)  # plan_body wells are frozen-valid
        planned[pos] = qsid

    parsed: dict[tuple[int, int], float | None] = {}
    for raw_well, value in req.readings.items():
        try:
            pos = parse_well(raw_well, rows, cols)
        except ValueError as exc:
            add(VIOLATION_WELL_REFERENCE_INVALID, str(exc))
            continue
        if pos in parsed:
            add(
                VIOLATION_XREAD_READING_DUPLICATED,
                f"well {well_name(*pos)} appears more than once in readings",
            )
            continue
        parsed[pos] = value
        if pos not in planned:
            add(
                VIOLATION_XREAD_READING_EXTRA,
                f"reading at well {well_name(*pos)} does not belong to any planned "
                "comparison well; readings must cover planned wells only",
            )

    missing = [
        well_name(*pos) for pos in sorted(planned)
        if pos not in parsed or parsed.get(pos) is None
    ]
    if missing:
        add(
            VIOLATION_XREAD_READING_MISSING,
            "every planned comparison well must carry a finite reading; "
            f"missing/non-finite reading at well(s): {', '.join(missing)}",
        )

    if categories:
        raise CrossBatchReadingInvalid(categories, messages)

    return {well_name(*pos): float(parsed[pos]) for pos in planned}


def validate_request(req: CrossBatchReadingRequest, plan_body: dict) -> dict[str, float]:
    """Resolve geometry and run every scan/reading content check."""
    plate = plan_body["plate"]
    try:
        rows, cols = _resolve_dimensions(PlateSpec(
            type=plate["type"], rows=plate["rows"], columns=plate["columns"]
        ))
    except InfeasibleArrangement as exc:
        raise CrossBatchReadingInvalid(exc.categories, exc.messages) from exc

    if req.baseline_batch_id not in plan_body["batches_by_id"]:
        raise CrossBatchReadingInvalid(
            [VIOLATION_XREAD_BASELINE_UNKNOWN],
            [
                f"baseline_batch_id '{req.baseline_batch_id}' does not participate "
                f"in plan {plan_body['plan_id']}; participating batch(es): "
                + ", ".join(sorted(plan_body["batches_by_id"]))
            ],
        )

    _check_scan(req, plan_body, rows, cols)
    return _check_readings(req, plan_body, rows, cols)


# --------------------------------------------------------------- evaluation


def _member_descriptor(m: _Member) -> dict:
    return {
        "batch_id": m.batch_id,
        "sample_id": m.sample_id,
        "qualified_sample_id": f"{m.batch_id}:{m.sample_id}",
        "group": m.group,
        "target_well": m.planned_well,
        "source_well": m.source_well,
        "source_reading": m.source_reading,
        "revision": m.revision,
        "digest": m.digest,
        "certificate_id": m.certificate_id,
    }


def evaluate(req: CrossBatchReadingRequest, plan_body: dict,
             members: dict[tuple[str, str], _Member],
             well_readings: dict[str, float]) -> dict:
    """Pair every non-baseline member with the baseline member per comparison."""
    baseline = req.baseline_batch_id
    comparisons_out: list[dict] = []
    pair_results: list[dict] = []
    passed_pairs = 0
    max_offset = 0.0

    for comp in sorted(plan_body["comparisons"], key=lambda c: c["comparison_id"]):
        entries = {m["batch_id"]: m for m in comp["members"]}
        base_entry = entries[baseline]
        base = members[(baseline, base_entry["sample_id"])]
        pairs: list[dict] = []
        for bid in sorted(entries):
            if bid == baseline:
                continue
            other_entry = entries[bid]
            other = members[(bid, other_entry["sample_id"])]
            source_delta = other.source_reading - base.source_reading
            target_delta = (
                well_readings[other.planned_well] - well_readings[base.planned_well]
            )
            offset = abs(target_delta - source_delta)
            pair_passed = offset <= req.allowed_offset
            passed_pairs += int(pair_passed)
            max_offset = max(max_offset, offset)
            pair = {
                "other_batch_id": bid,
                "baseline": _member_descriptor(base),
                "other": _member_descriptor(other),
                "baseline_source_reading": base.source_reading,
                "other_source_reading": other.source_reading,
                "source_reading_delta": source_delta,
                "baseline_target_reading": well_readings[base.planned_well],
                "other_target_reading": well_readings[other.planned_well],
                "target_reading_delta": target_delta,
                "offset_abs": offset,
                "allowed_offset": req.allowed_offset,
                "passed": pair_passed,
            }
            pairs.append(pair)
            pair_results.append(pair)
        comparisons_out.append({
            "comparison_id": comp["comparison_id"],
            "baseline_batch_id": baseline,
            "pairs": pairs,
            "passed": all(p["passed"] for p in pairs),
        })

    total_pairs = len(pair_results)
    verdict = VERDICT_PASS if total_pairs and passed_pairs == total_pairs else VERDICT_FAIL
    return {
        "comparisons": comparisons_out,
        "summary": {
            "total_pairs": total_pairs,
            "passed_pairs": passed_pairs,
            "failed_pairs": total_pairs - passed_pairs,
            "max_offset_abs": max_offset,
            "allowed_offset": req.allowed_offset,
            "verdict": verdict,
        },
    }


# ---------------------------------------------------------------- endpoint


def _verdict_id(request_key: str) -> str:
    return "xverdict-" + _sha256(
        f"cross-batch-verdict\0{request_key}".encode("utf-8")
    )[:24]


def create_cross_batch_verdict(req: CrossBatchReadingRequest) -> tuple[int, dict]:
    """Re-confirm sources, validate scan/readings, judge and persist the verdict."""
    store = get_store()
    content_hash = _content_hash(_ENDPOINT_XREAD, req)
    with store.transaction() as conn:
        # State first (same rationale as the plan endpoints): once any source
        # certificate is revoked or advances/changes, even an old recorded
        # verdict must not be replayed.
        plan_row = conn.execute(
            "SELECT plan_body FROM cross_batch_plans WHERE plan_id = ?",
            (req.plan_id,),
        ).fetchone()
        if plan_row is None:
            return 404, {
                "detail": "no saved cross-batch comparison plan exists for plan_id",
                "error": CROSS_BATCH_PLAN_NOT_FOUND,
                "plan_id": req.plan_id,
            }
        plan_body = json.loads(plan_row["plan_body"])
        plan_body["batches_by_id"] = {b["batch_id"]: b for b in plan_body["batches"]}

        certs, revisions, state_error = _check_sources(conn, plan_body)
        if state_error is not None:
            return state_error

        replay = _idem_replay(conn, req.request_key, content_hash)
        if replay is not None:
            return replay

        members, attr_error = _resolve_members(plan_body, revisions)
        if attr_error is not None:
            return attr_error

        try:
            well_readings = validate_request(req, plan_body)
        except CrossBatchReadingInvalid as exc:
            body = _invalid_body(exc)
            _idem_record(conn, req.request_key, _ENDPOINT_XREAD,
                         content_hash, 422, body)
            return 422, body

        result = evaluate(req, plan_body, members, well_readings)
        verdict = result["summary"]["verdict"]
        verdict_id = _verdict_id(req.request_key)

        # Canonical scan/readings (canonical well names, row-major) so the
        # stored evidence does not depend on the request's key casing/order.
        rows = plan_body["plate"]["rows"]
        cols = plan_body["plate"]["columns"]

        def _canon_scan() -> dict:
            out = {}
            for raw_well, qsid in req.scan.items():
                pos = parse_well(raw_well, rows, cols)
                out[well_name(*pos)] = qsid
            return {w: out[w] for w in sorted(out, key=lambda w: parse_well(w, rows, cols))}

        source_versions = [
            {
                "batch_id": bid,
                "certificate_id": certs[bid]["certificate_id"],
                "revision": revisions[bid]["revision"],
                "digest": revisions[bid]["digest"],
                "status": STATUS_ACTIVE,
            }
            for bid in sorted(certs)
        ]

        body = {
            "verdict_id": verdict_id,
            "plan_id": req.plan_id,
            "status": STATUS_ACTIVE,
            "request_key": req.request_key,
            "verdict": verdict,
            "baseline_batch_id": req.baseline_batch_id,
            "allowed_offset": req.allowed_offset,
            "plate": {
                "type": plan_body["plate"]["type"],
                "rows": plan_body["plate"]["rows"],
                "columns": plan_body["plate"]["columns"],
                "blocked_wells": plan_body["plate"].get("blocked_wells", []),
            },
            "scan": _canon_scan(),
            "readings": {
                w: well_readings[w]
                for w in sorted(well_readings, key=lambda w: parse_well(w, rows, cols))
            },
            "source_versions": source_versions,
            "comparisons": result["comparisons"],
            "summary": result["summary"],
        }
        now = _utcnow()
        conn.execute(
            "INSERT INTO cross_batch_verdicts (verdict_id, plan_id, request_key, "
            "baseline_batch_id, allowed_offset, verdict, source_versions, "
            "verdict_body, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                verdict_id,
                req.plan_id,
                req.request_key,
                req.baseline_batch_id,
                req.allowed_offset,
                verdict,
                json.dumps(source_versions, ensure_ascii=False),
                json.dumps(body, ensure_ascii=False),
                now,
            ),
        )
        _idem_record(conn, req.request_key, _ENDPOINT_XREAD, content_hash,
                     201, body)
        return 201, body
