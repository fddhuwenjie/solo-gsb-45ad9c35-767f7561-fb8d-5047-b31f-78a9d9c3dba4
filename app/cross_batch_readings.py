"""Measured judgements of saved cross-batch comparison plate plans.

``POST /api/v1/certificates/cross-batch-readings`` takes ``plan_id``, an
idempotency ``request_key``, the ``baseline_batch_id``, the full target-plate
``scan`` (well → ``"<batch_id>:<sample_id>"``, empty wells omitted), per-well
``readings`` and one ``allowed_deviation``.

Before judging anything the service re-confirms, inside one SQLite
transaction, the provenance the plan was saved with:

* every source certificate is still ACTIVE at the pinned revision and still
  carries the pinned revision digest — any revocation, revision advance or
  digest change is a 409, so a recorded judgement can never be replayed
  against revoked or version-changed evidence;
* every planned sample is still attributable to its batch from that
  revision's current frozen snapshot (it exists, is still on its planned well
  in the frozen scan at an unblocked well with a valid group label and a
  finite reading);

then checks the submitted evidence against the plan:

* the complete scan must reproduce the saved plan layout exactly (missing
  planned samples, unplanned qualified samples, duplicate wells/samples,
  blocked-well occupancy, malformed well names are 422);
* every planned well must carry one finite reading and no reading may refer
  to any other well (missing/extra readings are 422).

Only then are the comparisons judged: for every ``comparison_id`` the
baseline batch is paired with each other participating batch, and the pair
passes iff
``|(target-reading difference) - (source-certificate reading difference)|
<= allowed_deviation``. The verdict is PASS only when every pair passes.

The complete scan, readings, source revision/digest provenance and per-pair
results are persisted together with the verdict and the idempotency record in
one SQLite transaction. Same key + same content replays the recorded verdict;
same key + different content is 409.
"""
from __future__ import annotations

import json
import math
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
from .schemas import (
    CERTIFICATE_DIGEST_MISMATCH,
    CERTIFICATE_NOT_FOUND,
    CERTIFICATE_REVISION_MISMATCH,
    CERTIFICATE_REVOKED,
    CROSS_BATCH_PLAN_NOT_FOUND,
    CROSS_BATCH_SAMPLE_ATTRIBUTION_MISMATCH,
    VIOLATION_BLOCKED_WELL_OCCUPIED,
    VIOLATION_SAMPLES_MISSING,
    VIOLATION_SCAN_LAYOUT_MISMATCH,
    VIOLATION_SCAN_SAMPLES_DUPLICATED,
    VIOLATION_SCAN_WELLS_DUPLICATED,
    VIOLATION_UNPLANNED_SAMPLES,
    VIOLATION_WELL_REFERENCE_INVALID,
    VIOLATION_XREAD_BASELINE_UNKNOWN,
    VIOLATION_XREAD_READING_DUPLICATED,
    VIOLATION_XREAD_READING_EXTRA,
    VIOLATION_XREAD_READING_MISSING,
    CrossBatchReadingRequest,
)

_ENDPOINT_XREAD = "POST /api/v1/certificates/cross-batch-readings"


class CrossBatchReadingInvalid(Exception):
    """The submitted scan/readings cannot judge the plan (HTTP 422).

    No judgement row or idempotency record is written by the validator
    itself — the endpoint transaction records the 422 under the request key.
    """

    def __init__(self, categories: list[str], messages: list[str]):
        self.categories = categories
        self.messages = messages
        super().__init__("; ".join(messages) if messages else "invalid cross-batch reading request")


def _reading_invalid_body(exc: CrossBatchReadingInvalid) -> dict:
    """Canonical 422 envelope for an unjudgeable reading request."""
    return {
        "detail": "The cross-batch comparison plate cannot be judged from the "
        "submitted scan/readings; no judgement was saved.",
        "infeasibility_categories": exc.categories,
        "messages": exc.messages,
    }


@dataclass(frozen=True)
class _Member:
    """One planned comparison member with its planned target-plate well."""

    comparison_id: str
    batch_id: str
    sample_id: str
    group: str
    target_well: str
    revision: int
    digest: str


@dataclass(frozen=True)
class _Source:
    """One source batch's certificate and current-revision snapshot."""

    batch_id: str
    certificate_id: str
    revision: int
    digest: str
    snapshot: dict


# ------------------------------------------------------------ plan loading


def _load_plan(conn, plan_id: str) -> dict | None:
    row = conn.execute(
        "SELECT plan_body FROM cross_batch_plans WHERE plan_id = ?",
        (plan_id,),
    ).fetchone()
    return json.loads(row["plan_body"]) if row is not None else None


def _reading_id(plan_id: str, request_key: str) -> str:
    return "xread-" + _sha256(
        f"cross-batch-reading\0{plan_id}\0{request_key}".encode("utf-8")
    )[:24]


# ------------------------------------------------------------- state checks


def _resolve_sources(conn, plan: dict) -> tuple[dict[str, _Source] | None, tuple[int, dict] | None]:
    """Load every source certificate, verifying status/revision/digest.

    Returns ``(sources, None)`` when every source is still exactly the
    evidence the plan pinned, or ``(None, (status, body))`` with the first
    state-dependent rejection (404/409 — never recorded under the key).
    """
    sources: dict[str, _Source] = {}
    for pinned in plan["batches"]:
        bid = pinned["batch_id"]
        cert = _get_certificate(conn, bid)
        if cert is None:
            return None, (404, {
                "detail": "no certificate exists for a source batch of the plan",
                "error": CERTIFICATE_NOT_FOUND,
                "batch_id": bid,
                "plan_id": plan["plan_id"],
            })
        if cert["status"] == STATUS_REVOKED:
            return None, (409, {
                "detail": "a source certificate of the plan is REVOKED; the saved "
                "plan can no longer be judged and a revoked certificate can "
                "never be restored",
                "error": CERTIFICATE_REVOKED,
                "certificate_id": cert["certificate_id"],
                "batch_id": bid,
                "plan_id": plan["plan_id"],
                "status": STATUS_REVOKED,
                "revoked_reason": cert["revoked_reason"],
            })
        if cert["revision"] != pinned["revision"]:
            return None, (409, {
                "detail": "a source certificate of the plan has advanced to a "
                "different revision; the plan's readings cannot be judged "
                "against a revision other than the one the plan pinned",
                "error": CERTIFICATE_REVISION_MISMATCH,
                "certificate_id": cert["certificate_id"],
                "batch_id": bid,
                "plan_id": plan["plan_id"],
                "current_revision": cert["revision"],
                "pinned_revision": pinned["revision"],
            })
        if cert["digest"] != pinned["digest"]:
            return None, (409, {
                "detail": "a source certificate's current-revision digest no "
                "longer matches the digest the plan pinned; the frozen "
                "evidence changed and the plan can no longer be judged",
                "error": CERTIFICATE_DIGEST_MISMATCH,
                "certificate_id": cert["certificate_id"],
                "batch_id": bid,
                "plan_id": plan["plan_id"],
                "revision": cert["revision"],
                "current_digest": cert["digest"],
                "pinned_digest": pinned["digest"],
            })
        row = conn.execute(
            "SELECT digest, snapshot FROM certificate_revisions "
            "WHERE batch_id = ? AND revision = ?",
            (bid, cert["revision"]),
        ).fetchone()
        if row is None:  # pragma: no cover - invariant guard
            raise RuntimeError(
                f"revision {cert['revision']} snapshot missing for batch {bid!r}"
            )
        sources[bid] = _Source(
            batch_id=bid,
            certificate_id=cert["certificate_id"],
            revision=cert["revision"],
            digest=row["digest"],
            snapshot=json.loads(row["snapshot"]),
        )
    return sources, None


def _plan_members(plan: dict) -> list[_Member]:
    members: list[_Member] = []
    for comp in plan["comparisons"]:
        for m in comp["members"]:
            members.append(
                _Member(
                    comparison_id=comp["comparison_id"],
                    batch_id=m["batch_id"],
                    sample_id=m["sample_id"],
                    group=m["group"],
                    target_well=m["well"],
                    revision=m["revision"],
                    digest=m["digest"],
                )
            )
    return members


def _verify_attribution(members: list[_Member],
                        sources: dict[str, _Source]) -> tuple[int, dict] | None:
    """Re-confirm every planned sample from its batch's current snapshot.

    The sample must exist with the planned group, its planned well must still
    carry it in the frozen scan, the well must not be blocked and it must
    carry a valid finite reading. Any drift (defensive — the digest match
    already rules it out) is a 409 attribution mismatch.
    """
    for member in sorted(members, key=lambda m: (m.batch_id, m.sample_id, m.comparison_id)):
        src = sources[member.batch_id]
        snap = src.snapshot
        rows = snap["plate"]["rows"]
        cols = snap["plate"]["columns"]
        problems: list[str] = []

        # Per-member revision/digest pin: the plan stored the revision and the
        # revision digest with every member; they must still match the
        # certificate's current revision exactly.
        if member.revision != src.revision or member.digest != src.digest:
            return 409, {
                "detail": "a planned member's pinned revision/digest no longer "
                "matches its source certificate's current revision; the plan "
                "can no longer be judged",
                "error": CERTIFICATE_DIGEST_MISMATCH,
                "batch_id": member.batch_id,
                "sample_id": member.sample_id,
                "comparison_id": member.comparison_id,
                "current_revision": src.revision,
                "pinned_revision": member.revision,
                "current_digest": src.digest,
                "pinned_digest": member.digest,
            }

        group_of = {s["id"]: s["group"] for s in snap["samples"]}
        if member.sample_id not in snap.get("target", {}):
            problems.append("the sample no longer exists in the frozen layout")
        else:
            if group_of.get(member.sample_id) != member.group:
                problems.append("the sample's experiment group changed")
            pos = parse_well(snap["target"][member.sample_id], rows, cols)
            well = well_name(*pos)
            blocked = {parse_well(w, rows, cols) for w in snap.get("blocked_wells", [])}
            if pos in blocked:
                problems.append(f"the planned source well {well} is blocked")
            if snap.get("scan", {}).get(well) != member.sample_id:
                problems.append(
                    f"the frozen scan no longer places the sample at {well}"
                )
            value = snap.get("readings", {}).get(well)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"the source well {well} no longer has a finite reading")

        if problems:
            return 409, {
                "detail": "a planned sample's attribution no longer matches its "
                "batch's current-revision snapshot; the plan can no longer be judged",
                "error": CROSS_BATCH_SAMPLE_ATTRIBUTION_MISMATCH,
                "batch_id": member.batch_id,
                "sample_id": member.sample_id,
                "comparison_id": member.comparison_id,
                "revision": src.revision,
                "digest": src.digest,
                "problems": problems,
            }
    return None


# ------------------------------------------------------- scan / readings


def _canonical_well_map(raw: dict[str, str], rows: int, cols: int,
                        *, well_category: str, value_label: str,
                        categories: list[str], messages: list[str]):
    """Normalise a well-keyed string map, reporting invalid/duplicate keys.

    Returns the canonical ``well → value`` map (well names upper-cased) or an
    incomplete map when problems were found; callers keep aggregating.
    """
    out: dict[str, str] = {}
    for raw_well, value in raw.items():
        try:
            pos = parse_well(raw_well, rows, cols)
        except ValueError as exc:
            if VIOLATION_WELL_REFERENCE_INVALID not in categories:
                categories.append(VIOLATION_WELL_REFERENCE_INVALID)
            messages.append(str(exc))
            continue
        well = well_name(*pos)
        if well in out:
            if well_category not in categories:
                categories.append(well_category)
            messages.append(
                f"well {well} appears more than once (case-insensitive) in the "
                f"{value_label}"
            )
            continue
        out[well] = value
    return out


def _validate_request(
    req: CrossBatchReadingRequest, plan: dict, sources: dict[str, _Source]
) -> tuple[dict[str, _Member], dict[str, float], list[_Member]] | CrossBatchReadingInvalid:
    """Check the scan and readings against the saved plan; build judgement data.

    Returns ``(well_to_member, canonical_readings, members)`` on success or a
    ``CrossBatchReadingInvalid`` aggregating every content problem (422).
    """
    categories: list[str] = []
    messages: list[str] = []

    def add(category: str, msg: str) -> None:
        if category not in categories:
            categories.append(category)
        messages.append(msg)

    plate = plan["plate"]
    rows, cols = plate["rows"], plate["columns"]
    blocked = {
        well_name(*parse_well(w, rows, cols)) for w in plate.get("blocked_wells", [])
    }
    members = _plan_members(plan)
    batch_ids = [b["batch_id"] for b in plan["batches"]]

    if req.baseline_batch_id not in batch_ids:
        add(
            VIOLATION_XREAD_BASELINE_UNKNOWN,
            f"baseline_batch_id '{req.baseline_batch_id}' is not one of the "
            f"plan's batches: {', '.join(sorted(batch_ids))}",
        )

    # plan layout: target well -> qualified sample id / member
    well_to_member: dict[str, _Member] = {}
    expected: dict[str, str] = {}
    for m in members:
        qid = f"{m.batch_id}:{m.sample_id}"
        well_to_member[m.target_well] = m
        expected[m.target_well] = qid

    # ---- scan -------------------------------------------------------------
    scan = _canonical_well_map(
        req.scan, rows, cols,
        well_category=VIOLATION_SCAN_WELLS_DUPLICATED, value_label="scan",
        categories=categories, messages=messages,
    )
    # Duplicate qualified sample ids across scanned wells.
    seen_qids: dict[str, str] = {}
    for well, qid in scan.items():
        if qid in seen_qids:
            add(
                VIOLATION_SCAN_SAMPLES_DUPLICATED,
                f"qualified sample '{qid}' is scanned at both {seen_qids[qid]} "
                f"and {well}",
            )
        else:
            seen_qids[qid] = well
    for well, qid in scan.items():
        if well in blocked:
            add(
                VIOLATION_BLOCKED_WELL_OCCUPIED,
                f"blocked well {well} is occupied in the scan by '{qid}'",
            )
        elif well not in expected:
            add(
                VIOLATION_UNPLANNED_SAMPLES,
                f"well {well} is scanned as '{qid}' but the plan places no "
                "sample there",
            )
        elif expected[well] != qid:
            add(
                VIOLATION_SCAN_LAYOUT_MISMATCH,
                f"well {well} is scanned as '{qid}' but the plan places "
                f"'{expected[well]}' there",
            )
    for well, qid in sorted(expected.items()):
        if well not in scan:
            add(
                VIOLATION_SAMPLES_MISSING,
                f"planned sample '{qid}' at well {well} is absent from the scan",
            )

    # ---- readings ---------------------------------------------------------
    raw_readings: dict[str, float] = {}
    seen_reading_wells: set[str] = set()
    for raw_well, value in req.readings.items():
        try:
            pos = parse_well(raw_well, rows, cols)
        except ValueError as exc:
            if VIOLATION_WELL_REFERENCE_INVALID not in categories:
                categories.append(VIOLATION_WELL_REFERENCE_INVALID)
            messages.append(str(exc))
            continue
        well = well_name(*pos)
        if well in seen_reading_wells:
            if VIOLATION_XREAD_READING_DUPLICATED not in categories:
                categories.append(VIOLATION_XREAD_READING_DUPLICATED)
            messages.append(
                f"well {well} appears more than once (case-insensitive) in the "
                "readings"
            )
            continue
        seen_reading_wells.add(well)
        raw_readings[well] = value
    for well in sorted(raw_readings):
        if well not in well_to_member:
            add(
                VIOLATION_XREAD_READING_EXTRA,
                f"a reading is given for well {well}, which is not a planned "
                "comparison well",
            )
    for well, m in sorted(well_to_member.items()):
        if well not in raw_readings:
            add(
                VIOLATION_XREAD_READING_MISSING,
                f"planned well {well} ({m.batch_id}:{m.sample_id}, comparison "
                f"{m.comparison_id}) has no finite reading",
            )

    if categories:
        raise CrossBatchReadingInvalid(categories, messages)

    # The baseline must exist in every comparison for pairwise judging; the
    # plan builder guarantees one member per batch, but assert defensively.
    members_by_comparison: dict[str, dict[str, _Member]] = {}
    for m in members:
        members_by_comparison.setdefault(m.comparison_id, {})[m.batch_id] = m
    for cid, per_batch in members_by_comparison.items():
        if req.baseline_batch_id not in per_batch:
            add(
                VIOLATION_XREAD_BASELINE_UNKNOWN,
                f"comparison '{cid}' has no member from baseline batch "
                f"'{req.baseline_batch_id}'",
            )
    if categories:
        raise CrossBatchReadingInvalid(categories, messages)

    return well_to_member, raw_readings, members


# -------------------------------------------------------------- judgement


def _source_reading(source: _Source, sample_id: str) -> tuple[str, float]:
    """The frozen source-certificate well and reading of one member sample."""
    snap = source.snapshot
    rows = snap["plate"]["rows"]
    cols = snap["plate"]["columns"]
    pos = parse_well(snap["target"][sample_id], rows, cols)
    well = well_name(*pos)
    return well, snap["readings"][well]


def judge_comparisons(
    req: CrossBatchReadingRequest,
    plan: dict,
    sources: dict[str, _Source],
    well_to_member: dict[str, _Member],
    readings: dict[str, float],
) -> list[dict]:
    """Pair the baseline batch with every other batch within each comparison."""
    members_by_comparison: dict[str, list[_Member]] = {}
    for m in well_to_member.values():
        members_by_comparison.setdefault(m.comparison_id, []).append(m)

    baseline = req.baseline_batch_id
    comparisons_out: list[dict] = []
    for cid in sorted(members_by_comparison):
        per_batch = {m.batch_id: m for m in members_by_comparison[cid]}
        base_member = per_batch[baseline]
        base_source = sources[baseline]
        base_src_well, base_src_reading = _source_reading(base_source, base_member.sample_id)
        base_target_reading = readings[base_member.target_well]

        pairs_out: list[dict] = []
        for other_batch in sorted(b for b in per_batch if b != baseline):
            other_source = sources[other_batch]
            other_member = per_batch[other_batch]
            other_src_well, other_src_reading = _source_reading(
                other_source, other_member.sample_id
            )
            other_target_reading = readings[other_member.target_well]

            source_delta = other_src_reading - base_src_reading
            target_delta = other_target_reading - base_target_reading
            offset = abs(target_delta - source_delta)
            passed = offset <= req.allowed_deviation

            def member_view(member: _Member, src_well: str, src_reading: float,
                            target_reading: float) -> dict:
                return {
                    "batch_id": member.batch_id,
                    "sample_id": member.sample_id,
                    "qualified_sample_id": f"{member.batch_id}:{member.sample_id}",
                    "source_certificate_well": src_well,
                    "source_certificate_reading": src_reading,
                    "target_well": member.target_well,
                    "target_plate_reading": target_reading,
                    "revision": member.revision,
                    "digest": member.digest,
                }

            pairs_out.append(
                {
                    "baseline_batch_id": baseline,
                    "other_batch_id": other_batch,
                    "baseline": member_view(
                        base_member, base_src_well, base_src_reading, base_target_reading
                    ),
                    "other": member_view(
                        other_member, other_src_well, other_src_reading,
                        other_target_reading
                    ),
                    "source_certificate_reading_delta": source_delta,
                    "target_plate_reading_delta": target_delta,
                    "deviation_offset": offset,
                    "allowed_deviation": req.allowed_deviation,
                    "passed": passed,
                }
            )
        comparisons_out.append({"comparison_id": cid, "pairs": pairs_out})
    return comparisons_out


# ------------------------------------------------------------------ result


def _result_body(*, req: CrossBatchReadingRequest, plan: dict,
                 sources: dict[str, _Source], readings: dict[str, float],
                 well_to_member: dict[str, _Member],
                 comparisons_out: list[dict], reading_id: str) -> dict:
    rows, cols = plan["plate"]["rows"], plan["plate"]["columns"]
    blocked = sorted(
        {well_name(*parse_well(w, rows, cols)) for w in plan["plate"].get("blocked_wells", [])}
    )

    # Canonical row-major scan and per-well readings (empty wells never
    # appear — only planned wells carry readings).
    scan_canon = {
        m.target_well: f"{m.batch_id}:{m.sample_id}"
        for m in sorted(well_to_member.values(),
                        key=lambda m: parse_well(m.target_well, rows, cols))
    }
    readings_out = []
    for m in sorted(well_to_member.values(),
                    key=lambda m: parse_well(m.target_well, rows, cols)):
        readings_out.append(
            {
                "well": m.target_well,
                "batch_id": m.batch_id,
                "sample_id": m.sample_id,
                "qualified_sample_id": f"{m.batch_id}:{m.sample_id}",
                "comparison_id": m.comparison_id,
                "reading": readings[m.target_well],
            }
        )

    sources_out = [
        {
            "batch_id": src.batch_id,
            "certificate_id": src.certificate_id,
            "status": STATUS_ACTIVE,
            "revision": src.revision,
            "digest": src.digest,
        }
        for src in (sources[b["batch_id"]] for b in plan["batches"])
    ]

    pairs = [p for c in comparisons_out for p in c["pairs"]]
    passed_pairs = [p for p in pairs if p["passed"]]
    verdict = "PASS" if len(passed_pairs) == len(pairs) and pairs else "FAIL"
    return {
        "reading_id": reading_id,
        "plan_id": req.plan_id,
        "request_key": req.request_key,
        "status": STATUS_ACTIVE,
        "verdict": verdict,
        "baseline_batch_id": req.baseline_batch_id,
        "allowed_deviation": req.allowed_deviation,
        "plate": {
            "type": plan["plate"]["type"],
            "rows": rows,
            "columns": cols,
            "blocked_wells": blocked,
        },
        "sources": sources_out,
        "scan": scan_canon,
        "well_readings": readings_out,
        "comparisons": comparisons_out,
        "pair_count": len(pairs),
        "passed_pair_count": len(passed_pairs),
        "failed_pair_count": len(pairs) - len(passed_pairs),
        "all_pairs_passed": verdict == "PASS",
    }


# ---------------------------------------------------------------- endpoint


def create_cross_batch_reading(req: CrossBatchReadingRequest) -> tuple[int, dict]:
    """Verify provenance, validate the scan/readings, judge and persist."""
    store = get_store()
    content_hash = _content_hash(_ENDPOINT_XREAD, req)
    with store.transaction() as conn:
        # The plan must exist; this is state, not content — not keyed.
        plan = _load_plan(conn, req.plan_id)
        if plan is None:
            return 404, {
                "detail": "no saved cross-batch comparison plan exists for the "
                "given plan_id",
                "error": CROSS_BATCH_PLAN_NOT_FOUND,
                "plan_id": req.plan_id,
            }

        # State is checked BEFORE the idempotency lookup on purpose: once any
        # source certificate is revoked or advances/changes, even a replay of
        # a recorded judgement must be refused rather than replaying a verdict
        # confirmed against stale evidence.
        sources, state_error = _resolve_sources(conn, plan)
        if state_error is not None:
            status, body = state_error
            body["plan_id"] = req.plan_id
            return status, body

        replay = _idem_replay(conn, req.request_key, content_hash)
        if replay is not None:
            return replay

        members = _plan_members(plan)
        attribution_error = _verify_attribution(members, sources)
        if attribution_error is not None:
            status, body = attribution_error
            body["plan_id"] = req.plan_id
            return status, body

        try:
            well_to_member, readings, _ = _validate_request(req, plan, sources)
        except CrossBatchReadingInvalid as exc:
            body = _reading_invalid_body(exc)
            _idem_record(conn, req.request_key, _ENDPOINT_XREAD,
                         content_hash, 422, body)
            return 422, body

        comparisons_out = judge_comparisons(
            req, plan, sources, well_to_member, readings
        )
        reading_id = _reading_id(req.plan_id, req.request_key)
        body = _result_body(
            req=req, plan=plan, sources=sources, readings=readings,
            well_to_member=well_to_member, comparisons_out=comparisons_out,
            reading_id=reading_id,
        )
        now = _utcnow()
        conn.execute(
            "INSERT INTO cross_batch_readings (reading_id, plan_id, request_key, "
            "baseline_batch_id, allowed_deviation, verdict, sources, scan, "
            "readings, result_body, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                reading_id,
                req.plan_id,
                req.request_key,
                req.baseline_batch_id,
                req.allowed_deviation,
                body["verdict"],
                json.dumps(body["sources"], ensure_ascii=False),
                json.dumps(body["scan"], ensure_ascii=False),
                json.dumps(body["well_readings"], ensure_ascii=False),
                json.dumps(body, ensure_ascii=False),
                now,
            ),
        )
        _idem_record(conn, req.request_key, _ENDPOINT_XREAD, content_hash,
                     201, body)
        return 201, body
