"""Cross-batch comparison measured-judgement tests.

Endpoint ``POST /api/v1/certificates/cross-batch-readings``:

* 201 PASS/FAIL with complete scan, per-well readings, source revision/digest
  provenance and per-pair deltas/offset, persisted in one SQLite transaction
  with the idempotency record;
* same key + same content replays the recorded verdict; same key + different
  content is 409;
* 404 for an unknown plan id (or a missing source certificate);
* 409 when a source certificate is revoked, advances revision, changes digest
  or loses sample attribution — including a replay of an earlier recorded
  judgement after such a change;
* 422 (nothing saved) when the scan disagrees with the plan (missing/unplanned
  sample, wrong sample, duplicate well/sample, blocked-well occupancy,
  malformed well), a planned well lacks a finite reading, an extra reading is
  present or the baseline batch is unknown;
* 422 request-shape errors (missing fields, NaN/Infinity readings).
"""
import json
import math
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import certificates
from app.cross_batch_readings import _reading_id
from app.main import app

client = TestClient(app)

LAYOUT_4X4 = {
    "gA1": "A1", "gA2": "A3", "gB1": "A2", "gB2": "A4",
    "gB3": "B1", "gB4": "B3", "gA3": "B2", "gA4": "B4",
    "gA5": "C1", "gA6": "C3", "gB5": "C2", "gB6": "C4",
    "gB7": "D1", "gB8": "D3", "gA7": "D2", "gA8": "D4",
}
SAMPLES_4X4 = [
    {"id": s, "group": "GA" if s.startswith("gA") else "GB"}
    for s in LAYOUT_4X4
]
PLATE_4X4 = {"type": "custom", "rows": 4, "columns": 4}


@pytest.fixture(autouse=True)
def fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CERT_DB_PATH", str(tmp_path / "certificates.db"))
    certificates.reset_stores()
    yield tmp_path / "certificates.db"
    certificates.reset_stores()


def _hash(text: str) -> int:
    import hashlib
    return int(hashlib.md5(text.encode()).hexdigest()[:6], 16)


def _readings():
    return {
        w: (100.0 if s.startswith("gA") else 0.0) + (_hash(s) % 7) - 3
        for s, w in LAYOUT_4X4.items()
    }


def _release_payload(batch_seed=7):
    return {
        "plate": PLATE_4X4,
        "target": dict(LAYOUT_4X4),
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "blocked_wells": [],
        "samples": [dict(s) for s in SAMPLES_4X4],
        "readings": _readings(),
        "seed": batch_seed,
        "max_spatial_effect": 10.0,
        "min_p_value": 0.001,
        "permutations": 100,
        "tolerance": 1e-6,
        "max_iterations": 100,
    }


def _create_cert(batch_id="BATCH-1", key=None, **over):
    payload = {
        "batch_id": batch_id,
        "request_key": key or f"create-{batch_id}",
        "release": _release_payload(),
    }
    payload.update(over)
    r = client.post("/api/v1/certificates", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _member(batch_id, sample_id):
    return {"batch_id": batch_id, "sample_id": sample_id}


def _xcomp_payload(**over):
    payload = {
        "request_key": "xcomp-1",
        "batches": [
            {"batch_id": "BATCH-1", "expected_revision": 1},
            {"batch_id": "BATCH-2", "expected_revision": 1},
        ],
        "comparisons": [
            {"comparison_id": "C1", "members": [
                _member("BATCH-1", "gA1"), _member("BATCH-2", "gA2")]},
            {"comparison_id": "C2", "members": [
                _member("BATCH-1", "gB1"), _member("BATCH-2", "gB2")]},
            {"comparison_id": "C3", "members": [
                _member("BATCH-1", "gA5"), _member("BATCH-2", "gB5")]},
        ],
        "plate": {"type": "24"},
        "blocked_wells": [],
        "seed": 42,
    }
    payload.update(over)
    return payload


def _create_plan():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    r = client.post(
        "/api/v1/certificates/cross-batch-plans", json=_xcomp_payload()
    )
    assert r.status_code == 201, r.text
    return r.json()


# The two certified batches share the same release payload, so every member
# pair has identical source-certificate readings when the sample ids match.
def _source_reading(sample_id: str) -> float:
    return _readings()[LAYOUT_4X4[sample_id]]


def _scan_from_plan(plan):
    return {s["well"]: s["qualified_sample_id"] for s in plan["samples"]}


def _readings_payload(plan, *, delta=0.0, allowed=1.0):
    """Target readings reproducing (or perturbing) the source deltas.

    Every target-plate well receives its source-certificate reading plus a
    batch offset: baseline BATCH-1 members get 0, BATCH-2 members get
    ``delta``. With ``delta == 0`` every target difference equals the source
    difference (offset 0 → PASS for any positive allowed deviation).
    """
    readings = {}
    for s in plan["samples"]:
        value = _source_reading(s["sample_id"])
        if s["batch_id"] == "BATCH-2":
            value += delta
        readings[s["well"]] = value
    return readings


def _reading_request(plan, *, key="xread-1", baseline="BATCH-1",
                     allowed=1.0, delta=0.0, scan=None, readings=None):
    return {
        "plan_id": plan["plan_id"],
        "request_key": key,
        "baseline_batch_id": baseline,
        "scan": scan if scan is not None else _scan_from_plan(plan),
        "readings": readings if readings is not None
        else _readings_payload(plan, delta=delta, allowed=allowed),
        "allowed_deviation": allowed,
    }


def _post_reading(**over):
    plan = _create_plan()
    payload = _reading_request(plan, **over)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    return plan, payload, r


# ================================================================ happy path


def test_reading_pass_full_body_and_persistence(fresh_store):
    plan, payload, r = _post_reading(allowed=0.5)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["reading_id"] == _reading_id(plan["plan_id"], "xread-1")
    assert body["reading_id"].startswith("xread-")
    assert body["plan_id"] == plan["plan_id"]
    assert body["request_key"] == "xread-1"
    assert body["status"] == "ACTIVE"
    assert body["verdict"] == "PASS"
    assert body["baseline_batch_id"] == "BATCH-1"
    assert body["allowed_deviation"] == 0.5
    assert body["all_pairs_passed"] is True
    assert body["pair_count"] == 3
    assert body["passed_pair_count"] == 3 and body["failed_pair_count"] == 0

    # Source provenance echoes the plan's pinned revisions/digests.
    sources = {s["batch_id"]: s for s in body["sources"]}
    pinned = {b["batch_id"]: b for b in plan["batches"]}
    assert set(sources) == {"BATCH-1", "BATCH-2"}
    for bid, src in sources.items():
        assert src["status"] == "ACTIVE"
        assert src["revision"] == pinned[bid]["revision"]
        assert src["digest"] == pinned[bid]["digest"]

    # Scan and readings are fully traceable back to the plan.
    assert body["scan"] == _scan_from_plan(plan)
    assert len(body["well_readings"]) == len(plan["samples"])
    for entry in body["well_readings"]:
        assert entry["reading"] == payload["readings"][entry["well"]]
        assert entry["qualified_sample_id"] == f"{entry['batch_id']}:{entry['sample_id']}"

    # Per-pair arithmetic.
    comps = {c["comparison_id"]: c for c in body["comparisons"]}
    assert set(comps) == {"C1", "C2", "C3"}
    for cid, comp in comps.items():
        assert len(comp["pairs"]) == 1
        p = comp["pairs"][0]
        assert p["baseline_batch_id"] == "BATCH-1"
        assert p["other_batch_id"] == "BATCH-2"
        assert p["passed"] is True
        assert p["allowed_deviation"] == 0.5
        assert p["deviation_offset"] == pytest.approx(0.0)
        b1, b2 = p["baseline"], p["other"]
        assert b1["batch_id"] == "BATCH-1" and b2["batch_id"] == "BATCH-2"
        assert p["source_certificate_reading_delta"] == pytest.approx(
            b2["source_certificate_reading"] - b1["source_certificate_reading"])
        assert p["target_plate_reading_delta"] == pytest.approx(
            b2["target_plate_reading"] - b1["target_plate_reading"])
        assert p["deviation_offset"] == pytest.approx(abs(
            p["target_plate_reading_delta"] - p["source_certificate_reading_delta"]))
        # Source readings come from the frozen certificate snapshots.
        members = {m["batch_id"]: m for m in
                   next(c for c in plan["comparisons"]
                        if c["comparison_id"] == cid)["members"]}
        assert b1["sample_id"] == members["BATCH-1"]["sample_id"]
        assert b2["sample_id"] == members["BATCH-2"]["sample_id"]
        assert b1["source_certificate_reading"] == pytest.approx(
            _source_reading(members["BATCH-1"]["sample_id"]))
        assert b2["source_certificate_reading"] == pytest.approx(
            _source_reading(members["BATCH-2"]["sample_id"]))
        assert b1["revision"] == members["BATCH-1"]["revision"]
        assert b1["digest"] == members["BATCH-1"]["digest"]

    # Persisted in the same transaction as the idempotency record.
    conn = sqlite3.connect(str(fresh_store))
    row = conn.execute(
        "SELECT result_body, verdict FROM cross_batch_readings"
    ).fetchone()
    conn.close()
    assert row is not None and row[1] == "PASS"
    assert json.loads(row[0]) == body


def test_reading_fail_when_offset_exceeds_allowed(fresh_store):
    # A constant +0.8 offset on every BATCH-2 target well: each pair's target
    # delta is 0.8 away from its source delta. allowed 0.5 -> FAIL all pairs.
    plan, payload, r = _post_reading(allowed=0.5, delta=0.8)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["verdict"] == "FAIL"
    assert body["all_pairs_passed"] is False
    assert body["passed_pair_count"] == 0 and body["failed_pair_count"] == 3
    for comp in body["comparisons"]:
        p = comp["pairs"][0]
        assert p["passed"] is False
        assert p["deviation_offset"] == pytest.approx(0.8)
        assert p["target_plate_reading_delta"] == pytest.approx(
            p["source_certificate_reading_delta"] + 0.8)
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute(
        "SELECT verdict FROM cross_batch_readings").fetchone()[0] == "FAIL"
    conn.close()


def test_boundary_offset_equal_to_allowed_passes():
    plan, payload, r = _post_reading(allowed=0.8, delta=0.8)
    assert r.status_code == 201
    assert r.json()["verdict"] == "PASS"
    for comp in r.json()["comparisons"]:
        assert comp["pairs"][0]["passed"] is True


def test_one_failed_pair_makes_overall_fail():
    plan = _create_plan()
    readings = _readings_payload(plan)
    # Shift just one BATCH-2 well (C1's member) beyond the allowed deviation.
    c1_other = next(
        s for s in plan["samples"]
        if s["comparison_id"] == "C1" and s["batch_id"] == "BATCH-2")
    readings[c1_other["well"]] += 3.0
    payload = _reading_request(plan, readings=readings, allowed=0.5)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 201
    body = r.json()
    assert body["verdict"] == "FAIL"
    assert body["passed_pair_count"] == 2 and body["failed_pair_count"] == 1
    failed = [p for c in body["comparisons"] for p in c["pairs"] if not p["passed"]]
    assert len(failed) == 1
    cids_failed = {
        c["comparison_id"] for c in body["comparisons"]
        if not all(p["passed"] for p in c["pairs"])}
    assert cids_failed == {"C1"}


def test_baseline_b2_pairs_against_b1():
    plan, payload, r = _post_reading(baseline="BATCH-2", allowed=0.5)
    assert r.status_code == 201
    body = r.json()
    for comp in body["comparisons"]:
        p = comp["pairs"][0]
        assert p["baseline_batch_id"] == "BATCH-2"
        assert p["other_batch_id"] == "BATCH-1"
        assert p["baseline"]["batch_id"] == "BATCH-2"
        assert p["passed"] is True


# ============================================================== idempotency


def test_replay_same_key_same_content_returns_recorded_response():
    plan, payload, r1 = _post_reading()
    assert r1.status_code == 201
    r2 = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r2.status_code == 201
    assert r2.json() == r1.json()


def test_same_key_different_content_is_409_and_saves_nothing(fresh_store):
    plan = _create_plan()
    payload = _reading_request(plan, allowed=1.0)
    first = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert first.status_code == 201
    changed = json.loads(json.dumps(payload))
    changed["allowed_deviation"] = 0.25
    r = client.post("/api/v1/certificates/cross-batch-readings", json=changed)
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute(
        "SELECT COUNT(*) FROM cross_batch_readings").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'xread-1'"
    ).fetchone()[0] == 1
    conn.close()


def test_422_is_recorded_and_replays_identically(fresh_store):
    plan = _create_plan()
    payload = _reading_request(plan)
    del payload["readings"][plan["samples"][0]["well"]]
    b1 = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    b2 = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert b1.status_code == 422 and b2.status_code == 422
    assert b1.json() == b2.json()
    assert "XREAD_READING_MISSING" in b1.json()["infeasibility_categories"]
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute(
        "SELECT COUNT(*) FROM cross_batch_readings").fetchone()[0] == 0
    conn.close()
    changed = json.loads(json.dumps(payload))
    changed["allowed_deviation"] = 0.1
    assert client.post(
        "/api/v1/certificates/cross-batch-readings", json=changed
    ).status_code == 409


def test_request_key_shared_with_plan_endpoint_conflicts():
    plan = _create_plan()  # already recorded under key 'xcomp-1'
    payload = _reading_request(plan, key="xcomp-1")
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"


# ============================================================ state errors


def test_unknown_plan_id_is_404(fresh_store):
    _create_cert("BATCH-1")
    payload = {
        "plan_id": "xplan-doesnotexist",
        "request_key": "xread-x",
        "baseline_batch_id": "BATCH-1",
        "scan": {"A1": "BATCH-1:g1"},
        "readings": {"A1": 1.0},
        "allowed_deviation": 1.0,
    }
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 404
    assert r.json()["error"] == "CROSS_BATCH_PLAN_NOT_FOUND"
    assert r.json()["plan_id"] == "xplan-doesnotexist"
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'xread-x'"
    ).fetchone()[0] == 0
    conn.close()


def test_source_certificate_missing_is_404(fresh_store):
    plan = _create_plan()
    # Delete BATCH-2's certificate rows to simulate a vanished source.
    conn = sqlite3.connect(str(fresh_store))
    conn.execute("DELETE FROM certificates WHERE batch_id = 'BATCH-2'")
    conn.commit()
    conn.close()
    certificates.reset_stores()
    payload = _reading_request(plan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 404
    assert r.json()["error"] == "CERTIFICATE_NOT_FOUND"
    assert r.json()["batch_id"] == "BATCH-2"


def test_revoked_source_certificate_is_409():
    plan = _create_plan()
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-1", "request_key": "rv-1", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items() if w != "A1"},
        "readings": _readings(),
    })
    payload = _reading_request(plan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVOKED"
    assert body["batch_id"] == "BATCH-1"
    assert body["revoked_reason"] == "SAMPLES_MISSING"


def test_source_revision_advance_is_409_and_blocks_old_replay():
    plan = _create_plan()
    payload = _reading_request(plan)
    first = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert first.status_code == 201
    new_readings = dict(_readings())
    new_readings["A1"] = 1.0
    rv = client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-2", "request_key": "rv-2", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": new_readings,
    })
    assert rv.status_code == 200 and rv.json()["revision"] == 2
    replay = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert replay.status_code == 409
    assert replay.json()["error"] == "CERTIFICATE_REVISION_MISMATCH"
    assert replay.json()["batch_id"] == "BATCH-2"
    assert replay.json()["current_revision"] == 2
    assert replay.json()["pinned_revision"] == 1


def test_source_digest_change_at_same_revision_is_409(fresh_store):
    plan = _create_plan()
    # Tamper BATCH-1 revision-1 snapshot content (and recompute its digest),
    # keeping revision == 1: the digest the plan pinned no longer matches.
    conn = sqlite3.connect(str(fresh_store))
    row = conn.execute(
        "SELECT snapshot FROM certificates WHERE batch_id = 'BATCH-1'").fetchone()
    snap = json.loads(row[0])
    snap["readings"]["A1"] = 77.77
    snap_json = certificates._canonical_json(snap)
    digest = certificates._sha256(snap_json)
    conn.execute(
        "UPDATE certificates SET snapshot = ?, digest = ? WHERE batch_id = 'BATCH-1'",
        (snap_json.decode(), digest))
    conn.execute(
        "UPDATE certificate_revisions SET snapshot = ?, digest = ? "
        "WHERE batch_id = 'BATCH-1' AND revision = 1",
        (snap_json.decode(), digest))
    conn.commit()
    conn.close()
    certificates.reset_stores()

    payload = _reading_request(plan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_DIGEST_MISMATCH"
    assert body["batch_id"] == "BATCH-1"
    assert body["current_digest"] == digest
    assert body["revision"] == 1


def test_state_409_does_not_burn_request_key(fresh_store):
    plan = _create_plan()
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-2", "request_key": "rv-2", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": {**_readings(), "A1": 1.0},
    })
    payload = _reading_request(plan)
    stale = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert stale.status_code == 409
    # No new plan can be re-pinned to revision 2 via the old plan; but the key
    # stays unused: a structurally different valid request under the same key
    # against a matching plan must not get REQUEST_KEY_CONFLICT. Here we simply
    # assert no idempotency row exists for the key.
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'xread-1'"
    ).fetchone()[0] == 0
    conn.close()


# ============================================================ content 422s


def test_missing_planned_sample_is_422_and_saves_nothing(fresh_store):
    plan = _create_plan()
    scan = _scan_from_plan(plan)
    del scan[plan["samples"][0]["well"]]
    payload = _reading_request(plan, scan=scan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "SAMPLES_MISSING" in r.json()["infeasibility_categories"]
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute(
        "SELECT COUNT(*) FROM cross_batch_readings").fetchone()[0] == 0
    conn.close()


def test_unplanned_scanned_sample_is_422():
    plan = _create_plan()
    scan = _scan_from_plan(plan)
    scan["A6"] = "BATCH-1:gA7"
    payload = _reading_request(plan, scan=scan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "UNPLANNED_SAMPLES" in r.json()["infeasibility_categories"]


def test_wrong_sample_at_planned_well_is_422():
    plan = _create_plan()
    scan = _scan_from_plan(plan)
    well = plan["samples"][0]["well"]
    scan[well] = "BATCH-2:gB7"
    payload = _reading_request(plan, scan=scan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "SCAN_LAYOUT_MISMATCH" in r.json()["infeasibility_categories"]


def test_duplicate_scan_well_case_insensitive_is_422():
    plan = _create_plan()
    scan = _scan_from_plan(plan)
    first_well = plan["samples"][0]["well"]
    # Same well submitted under two casings with two JSON keys.
    scan[first_well.lower()] = scan[first_well]
    payload = _reading_request(plan, scan=scan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "SCAN_WELLS_DUPLICATED" in r.json()["infeasibility_categories"]


def test_duplicate_scanned_sample_is_422():
    plan = _create_plan()
    scan = _scan_from_plan(plan)
    wells = [s["well"] for s in plan["samples"]]
    scan[wells[1]] = scan[wells[0]]
    payload = _reading_request(plan, scan=scan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "SCAN_SAMPLES_DUPLICATED" in r.json()["infeasibility_categories"]


def test_blocked_well_occupied_is_422():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload(
        request_key="xcomp-blocked", blocked_wells=["D6"])
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 201, r.text
    plan = r.json()
    # A blocked well cannot be a planned well; occupy it additionally.
    scan = _scan_from_plan(plan)
    scan["D6"] = "BATCH-1:gA8"
    req = _reading_request(plan, scan=scan, key="xread-blocked")
    rr = client.post("/api/v1/certificates/cross-batch-readings", json=req)
    assert rr.status_code == 422
    cats = rr.json()["infeasibility_categories"]
    assert cats == ["BLOCKED_WELL_OCCUPIED"]


def test_invalid_scan_well_is_422():
    plan = _create_plan()
    scan = _scan_from_plan(plan)
    scan["Z99"] = "BATCH-1:gA8"
    payload = _reading_request(plan, scan=scan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]


def test_missing_reading_is_422():
    plan = _create_plan()
    readings = _readings_payload(plan)
    missing_well = plan["samples"][0]["well"]
    del readings[missing_well]
    payload = _reading_request(plan, readings=readings)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "XREAD_READING_MISSING" in r.json()["infeasibility_categories"]
    assert any(missing_well in m for m in r.json()["messages"])


def test_extra_reading_is_422():
    plan = _create_plan()
    readings = _readings_payload(plan)
    readings["A6"] = 9.9
    payload = _reading_request(plan, readings=readings)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "XREAD_READING_EXTRA" in r.json()["infeasibility_categories"]


def test_duplicate_reading_well_case_insensitive_is_422():
    plan = _create_plan()
    readings = dict(_readings_payload(plan))
    target = plan["samples"][0]["well"]
    readings[target.lower()] = readings[target]
    payload = _reading_request(plan, readings=readings)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "XREAD_READING_DUPLICATED" in r.json()["infeasibility_categories"]


def test_unknown_baseline_batch_is_422():
    plan, payload, r = _post_reading(baseline="BATCH-9")
    assert r.status_code == 422
    assert "XREAD_BASELINE_UNKNOWN" in r.json()["infeasibility_categories"]


def test_multiple_scan_problems_are_aggregated():
    plan = _create_plan()
    scan = _scan_from_plan(plan)
    wells = [s["well"] for s in plan["samples"]]
    del scan[wells[0]]           # missing planned sample
    scan[wells[1]] = scan[wells[2]]  # duplicate sample (and mismatch)
    payload = _reading_request(plan, scan=scan)
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "SAMPLES_MISSING" in cats
    assert "SCAN_SAMPLES_DUPLICATED" in cats
    assert "SCAN_LAYOUT_MISMATCH" in cats


# =========================================================== request shape


def test_missing_field_is_request_validation_error():
    plan = _create_plan()
    payload = _reading_request(plan)
    del payload["allowed_deviation"]
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_unknown_field_is_request_validation_error():
    plan = _create_plan()
    payload = _reading_request(plan)
    payload["extra"] = 1
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


@pytest.mark.parametrize("token", ["0.0", "-1.0", "NaN", "Infinity"])
def test_invalid_allowed_deviation_is_422(token):
    plan = _create_plan()
    payload = _reading_request(plan, allowed=1.0)
    raw = json.dumps(payload).replace(
        '"allowed_deviation": 1.0', f'"allowed_deviation": {token}', 1)
    r = client.post(
        "/api/v1/certificates/cross-batch-readings",
        content=raw, headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_nan_reading_is_request_validation_error():
    plan = _create_plan()
    payload = _reading_request(plan)
    target_well = plan["samples"][0]["well"]
    raw = json.dumps(payload)
    # Replace the finite value at the target well with the NaN token.
    needle = f'"{target_well}": {payload["readings"][target_well]}'
    raw = raw.replace(needle, f'"{target_well}": NaN', 1)
    r = client.post(
        "/api/v1/certificates/cross-batch-readings",
        content=raw, headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_reading_id_deterministic():
    plan, payload, r = _post_reading()
    assert r.json()["reading_id"] == _reading_id(plan["plan_id"], payload["request_key"])


def test_existing_endpoints_still_available():
    assert client.get("/health").json() == {"status": "ok"}
    _create_cert("BATCH-1")
    r = client.post(
        "/api/v1/arrangements/batch-release", json=_release_payload())
    assert r.status_code == 200 and r.json()["verdict"] == "RELEASED"
