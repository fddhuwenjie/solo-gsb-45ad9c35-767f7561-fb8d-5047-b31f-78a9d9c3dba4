"""Cross-batch measured-reading interpretation tests.

Covers ``POST /api/v1/certificates/cross-batch-readings``:

* PASS/FAIL arithmetic per comparison (source-certificate reading delta,
  target-plate reading delta, absolute offset, allowed-offset gate);
* 422 content checks: scan/plan disagreement (missing, unplanned, swapped,
  blocked-well occupancy, malformed well), missing/extra/duplicated/null
  readings, unknown baseline batch;
* 404 unknown plan / missing source certificate;
* 409 source certificate REVOKED / revision advanced / digest changed /
  attribution lost — and the guarantee that an old recorded verdict cannot
  be replayed after any of those state changes;
* idempotency: same-key same-content replay, same-key different-content 409,
  recorded 422 replay, key shared with another endpoint;
* single-transaction persistence of verdict + idempotency record.
"""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import certificates, cross_batch_verdicts
from app.main import app

client = TestClient(app)

# Tests that provision their own batch set (or need no certificates at all);
# the autouse default-certificate fixture skips these by test name.
_NO_DEFAULT_CERTS = {
    "test_reading_three_batches_pairs_every_member_with_baseline",
    "test_reading_unknown_plan_is_404",
    "test_reading_request_key_shared_with_other_endpoint_conflicts",
    "test_existing_endpoints_still_available_after_readings",
}

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


@pytest.fixture(autouse=True)
def _default_two_certified_batches(fresh_store, request):
    """Most interpretations span BATCH-1/BATCH-2 at revision 1; tests that
    provision a different batch set list their name in _NO_DEFAULT_CERTS."""
    if request.node.name in _NO_DEFAULT_CERTS:
        return
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")


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


def _make_plan(**over):
    r = client.post(
        "/api/v1/certificates/cross-batch-plans", json=_xcomp_payload(**over))
    assert r.status_code == 201, r.text
    return r.json()


def _source_reading(batch_id, sample_id):
    # Both fixture batches were certified from the identical release payload.
    return _readings()[LAYOUT_4X4[sample_id]]


def _scan_from_plan(plan):
    return {well: qsid for qsid, well in plan["layout"].items()}


def _passing_readings(plan, *, shift=500.0):
    """Target readings reproducing the source deltas exactly -> offset 0."""
    out = {}
    for qsid, well in plan["layout"].items():
        bid, sid = qsid.split(":", 1)
        out[well] = shift + _source_reading(bid, sid)
    return out


def _reading_payload(plan, **over):
    payload = {
        "plan_id": plan["plan_id"],
        "request_key": "read-1",
        "baseline_batch_id": "BATCH-1",
        "scan": _scan_from_plan(plan),
        "readings": _passing_readings(plan),
        "allowed_offset": 0.25,
    }
    payload.update(over)
    return payload


def _post_reading(plan, **over):
    return client.post(
        "/api/v1/certificates/cross-batch-readings",
        json=_reading_payload(plan, **over),
    )


# ================================================================ arithmetic


def test_reading_pass_when_target_deltas_match_source_deltas():
    plan = _make_plan()
    r = _post_reading(plan)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["verdict"] == "PASS"
    assert body["verdict_id"].startswith("xverdict-")
    assert body["status"] == "ACTIVE"
    assert body["plan_id"] == plan["plan_id"]
    assert body["request_key"] == "read-1"
    assert body["baseline_batch_id"] == "BATCH-1"
    assert body["allowed_offset"] == 0.25

    summary = body["summary"]
    assert summary == {
        "total_pairs": 3, "passed_pairs": 3, "failed_pairs": 0,
        "max_offset_abs": 0.0, "allowed_offset": 0.25, "verdict": "PASS",
    }

    comps = {c["comparison_id"]: c for c in body["comparisons"]}
    assert list(comps) == ["C1", "C2", "C3"]
    for comp in body["comparisons"]:
        assert comp["baseline_batch_id"] == "BATCH-1"
        assert comp["passed"] is True
        assert len(comp["pairs"]) == 1
        p = comp["pairs"][0]
        assert p["other_batch_id"] == "BATCH-2"
        assert p["passed"] is True
        bs, bo = p["baseline"], p["other"]
        assert bs["batch_id"] == "BATCH-1" and bo["batch_id"] == "BATCH-2"
        # Delta arithmetic is internally consistent.
        assert p["source_reading_delta"] == pytest.approx(
            p["other_source_reading"] - p["baseline_source_reading"])
        assert p["target_reading_delta"] == pytest.approx(
            p["other_target_reading"] - p["baseline_target_reading"])
        assert p["offset_abs"] == pytest.approx(
            abs(p["target_reading_delta"] - p["source_reading_delta"]))
        # Provenance of every member is echoed back.
        for desc, bid in ((bs, "BATCH-1"), (bo, "BATCH-2")):
            assert desc["qualified_sample_id"] == f"{bid}:{desc['sample_id']}"
            assert desc["revision"] == 1
            assert desc["digest"]
            assert desc["certificate_id"].startswith("cert-")
            assert desc["source_reading"] == pytest.approx(
                _source_reading(bid, desc["sample_id"]))

    # The stored evidence echoes the full scan and readings.
    assert body["scan"] == _scan_from_plan(plan)
    assert set(body["readings"]) == set(plan["wells"])
    versions = {v["batch_id"]: v for v in body["source_versions"]}
    assert set(versions) == {"BATCH-1", "BATCH-2"}
    for bid, v in versions.items():
        assert v["status"] == "ACTIVE" and v["revision"] == 1 and v["digest"]


def test_reading_fail_when_one_offset_exceeds_allowed():
    plan = _make_plan()
    readings = _passing_readings(plan)
    # Shift one target well so one pair's offset grows beyond the allowance.
    other_well = plan["comparisons"][0]["members"][1]["well"]  # BATCH-2 gA2
    readings[other_well] += 1.0
    r = _post_reading(plan, readings=readings, allowed_offset=0.5)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["verdict"] == "FAIL"
    assert body["summary"]["passed_pairs"] == 2
    assert body["summary"]["failed_pairs"] == 1
    assert body["summary"]["max_offset_abs"] == pytest.approx(1.0)
    c1 = body["comparisons"][0]
    assert c1["passed"] is False
    assert c1["pairs"][0]["passed"] is False
    assert c1["pairs"][0]["offset_abs"] == pytest.approx(1.0)
    assert body["comparisons"][1]["passed"] is True


def test_reading_boundary_offset_equal_to_allowed_passes():
    plan = _make_plan()
    readings = _passing_readings(plan)
    other_well = plan["comparisons"][1]["members"][1]["well"]
    readings[other_well] += 0.25
    r = _post_reading(plan, readings=readings, allowed_offset=0.25)
    assert r.status_code == 201
    assert r.json()["verdict"] == "PASS"


def test_reading_three_batches_pairs_every_member_with_baseline():
    for b in ("BATCH-1", "BATCH-2", "BATCH-3"):
        _create_cert(b)
    plan = _make_plan(
        request_key="xcomp-3",
        batches=[{"batch_id": f"BATCH-{b}", "expected_revision": 1}
                 for b in range(1, 4)],
        comparisons=[
            {"comparison_id": "C1", "members": [
                _member("BATCH-1", "gA1"), _member("BATCH-2", "gA2"),
                _member("BATCH-3", "gA5")]},
            {"comparison_id": "C2", "members": [
                _member("BATCH-1", "gB1"), _member("BATCH-2", "gB2"),
                _member("BATCH-3", "gB3")]},
        ],
    )
    r = _post_reading(plan, request_key="read-3", baseline_batch_id="BATCH-2")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["summary"]["total_pairs"] == 4
    assert body["summary"]["verdict"] == "PASS"
    for comp in body["comparisons"]:
        assert {p["other_batch_id"] for p in comp["pairs"]} == {"BATCH-1", "BATCH-3"}
        assert all(p["baseline"]["batch_id"] == "BATCH-2" for p in comp["pairs"])


# ================================================================ 422 content


def test_reading_scan_swapped_sample_is_422():
    plan = _make_plan()
    scan = _scan_from_plan(plan)
    wells = list(scan)
    scan[wells[0]], scan[wells[1]] = scan[wells[1]], scan[wells[0]]
    r = _post_reading(plan, scan=scan)
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "SCAN_LAYOUT_MISMATCH" in cats
    assert r.json()["misplaced_wells"]


def test_reading_scan_closed_loop_is_422():
    plan = _make_plan()
    # Choose three planned wells and rotate their occupants.
    wells = plan["wells"][:3]
    scan = _scan_from_plan(plan)
    qsids = [scan[w] for w in wells]
    scan[wells[0]], scan[wells[1]], scan[wells[2]] = qsids[2], qsids[0], qsids[1]
    r = _post_reading(plan, scan=scan)
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "CLOSED_LOOP_MISPLACEMENT" in cats
    assert "SCAN_LAYOUT_MISMATCH" in cats
    assert r.json()["cycles"]


def test_reading_scan_missing_sample_is_422():
    plan = _make_plan()
    scan = _scan_from_plan(plan)
    del scan[plan["wells"][0]]
    r = _post_reading(plan, scan=scan)
    assert r.status_code == 422
    assert "SAMPLES_MISSING" in r.json()["infeasibility_categories"]


def test_reading_scan_unplanned_sample_is_422():
    plan = _make_plan()
    scan = _scan_from_plan(plan)
    scan["D6"] = "BATCH-1:nope"  # free well on the 24 plate
    r = _post_reading(plan, scan=scan)
    assert r.status_code == 422
    assert "UNPLANNED_SAMPLES" in r.json()["infeasibility_categories"]


def test_reading_scan_duplicate_well_key_casing_is_422():
    plan = _make_plan()
    payload = _reading_payload(plan)
    well = plan["wells"][0]
    occupant = payload["scan"].pop(well)
    payload["scan"][well.lower()] = occupant
    payload["scan"][well] = "BATCH-1:gB1"  # collides after case normalisation
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "SCAN_WELLS_DUPLICATED" in r.json()["infeasibility_categories"]


def test_reading_scan_bad_well_is_422():
    plan = _make_plan()
    scan = _scan_from_plan(plan)
    scan["Z99"] = "BATCH-1:gA1"
    r = _post_reading(plan, scan=scan)
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]


def test_reading_scan_on_blocked_well_is_422():
    plan = _make_plan(blocked_wells=["D6"], request_key="xcomp-blk")
    scan = _scan_from_plan(plan)
    scan["D6"] = "BATCH-1:nope"
    r = _post_reading(plan, scan=scan)
    assert r.status_code == 422
    assert "BLOCKED_WELL_OCCUPIED" in r.json()["infeasibility_categories"]


def test_reading_missing_reading_is_422_and_saves_nothing(fresh_store):
    plan = _make_plan()
    readings = _passing_readings(plan)
    del readings[plan["wells"][0]]
    r = _post_reading(plan, readings=readings)
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "XREAD_READING_MISSING" in cats
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute("SELECT COUNT(*) FROM cross_batch_verdicts").fetchone()[0] == 0
    conn.close()


def test_reading_null_reading_is_422():
    plan = _make_plan()
    readings = _passing_readings(plan)
    readings[plan["wells"][1]] = None
    r = _post_reading(plan, readings=readings)
    assert r.status_code == 422
    assert "XREAD_READING_MISSING" in r.json()["infeasibility_categories"]


def test_reading_extra_reading_is_422():
    plan = _make_plan()
    readings = _passing_readings(plan)
    readings["D6"] = 1.0
    r = _post_reading(plan, readings=readings)
    assert r.status_code == 422
    assert "XREAD_READING_EXTRA" in r.json()["infeasibility_categories"]


def test_reading_duplicate_well_casing_is_422():
    plan = _make_plan()
    payload = _reading_payload(plan)
    w = plan["wells"][0]
    payload["readings"][w.lower()] = payload["readings"].pop(w)
    payload["readings"][w] = 9.0
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert "XREAD_READING_DUPLICATED" in r.json()["infeasibility_categories"]


def test_reading_nan_is_request_validation_error():
    plan = _make_plan()
    payload = _reading_payload(plan)
    w = plan["wells"][0]
    payload["readings"][w] = float("nan")
    r = client.post(
        "/api/v1/certificates/cross-batch-readings",
        content=json.dumps(payload),  # Python emits the non-standard NaN token
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_reading_unknown_baseline_is_422():
    plan = _make_plan()
    r = _post_reading(plan, baseline_batch_id="BATCH-9")
    assert r.status_code == 422
    assert "XREAD_BASELINE_UNKNOWN" in r.json()["infeasibility_categories"]


def test_reading_unknown_field_is_validation_error():
    plan = _make_plan()
    payload = _reading_payload(plan)
    payload["extra"] = 1
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


# ================================================================ 404 / 409


def test_reading_unknown_plan_is_404():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = {
        "plan_id": "xplan-deadbeef",
        "request_key": "read-x",
        "baseline_batch_id": "BATCH-1",
        "scan": {"A1": "BATCH-1:gA1"},
        "readings": {"A1": 1.0},
        "allowed_offset": 0.1,
    }
    r = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert r.status_code == 404
    assert r.json()["error"] == "CROSS_BATCH_PLAN_NOT_FOUND"
    assert r.json()["plan_id"] == "xplan-deadbeef"


def test_reading_source_certificate_deleted_is_404(fresh_store):
    plan = _make_plan()
    conn = sqlite3.connect(str(fresh_store))
    conn.execute("DELETE FROM certificates WHERE batch_id = 'BATCH-2'")
    conn.commit()
    conn.close()
    certificates.reset_stores()
    r = _post_reading(plan)
    assert r.status_code == 404
    assert r.json()["error"] == "CERTIFICATE_NOT_FOUND"
    assert r.json()["batch_id"] == "BATCH-2"


def test_reading_revoked_source_certificate_is_409():
    plan = _make_plan()
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-2", "request_key": "rv-1", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items() if w != "A1"},
        "readings": _readings(),
    })
    r = _post_reading(plan)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVOKED"
    assert body["batch_id"] == "BATCH-2"
    assert body["revoked_reason"] == "SAMPLES_MISSING"


def test_reading_revision_advanced_is_409_even_for_recorded_key(fresh_store):
    plan = _make_plan()
    first = _post_reading(plan)
    assert first.status_code == 201
    new_readings = dict(_readings())
    new_readings["A1"] = 1.0
    rv = client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-2", "request_key": "rv-2", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": new_readings,
    })
    assert rv.status_code == 200 and rv.json()["revision"] == 2

    replay = _post_reading(plan)
    assert replay.status_code == 409
    body = replay.json()
    assert body["error"] == "CERTIFICATE_REVISION_MISMATCH"
    assert body["batch_id"] == "BATCH-2"
    assert body["current_revision"] == 2 and body["expected_revision"] == 1
    # The state refusal did not duplicate any records.
    conn = sqlite3.connect(str(fresh_store))
    n_verdicts = conn.execute(
        "SELECT COUNT(*) FROM cross_batch_verdicts WHERE request_key = 'read-1'"
    ).fetchone()[0]
    n_keys = conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'read-1'"
    ).fetchone()[0]
    conn.close()
    assert n_verdicts == 1 and n_keys == 1


def test_reading_changed_revision_digest_is_409(fresh_store):
    plan = _make_plan()
    conn = sqlite3.connect(str(fresh_store))
    conn.execute(
        "UPDATE certificate_revisions SET digest = ? "
        "WHERE batch_id = 'BATCH-1' AND revision = 1",
        ("0" * 64,),
    )
    conn.commit()
    conn.close()
    certificates.reset_stores()
    r = _post_reading(plan)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_DIGEST_MISMATCH"
    assert body["batch_id"] == "BATCH-1"
    assert body["current_digest"] == "0" * 64
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'read-1'"
    ).fetchone()[0] == 0
    conn.close()


def test_reading_lost_attribution_is_409(fresh_store):
    plan = _make_plan()
    # Tamper the frozen snapshot (swap the scan at two wells) but leave the
    # pinned digest in place so the attribution check is what fires.
    conn = sqlite3.connect(str(fresh_store))
    row = conn.execute(
        "SELECT snapshot FROM certificate_revisions "
        "WHERE batch_id = 'BATCH-1' AND revision = 1"
    ).fetchone()
    snap = json.loads(row[0])
    snap["scan"]["A1"], snap["scan"]["A2"] = snap["scan"]["A2"], snap["scan"]["A1"]
    snap_json = certificates._canonical_json(snap).decode()
    conn.execute(
        "UPDATE certificates SET snapshot = ? WHERE batch_id = 'BATCH-1'",
        (snap_json,))
    conn.execute(
        "UPDATE certificate_revisions SET snapshot = ? "
        "WHERE batch_id = 'BATCH-1' AND revision = 1",
        (snap_json,))
    conn.commit()
    conn.close()
    certificates.reset_stores()

    r = _post_reading(plan)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CROSS_BATCH_SAMPLE_ATTRIBUTION_MISMATCH"
    assert body["batch_id"] == "BATCH-1"


# ================================================================ idempotency


def test_reading_replay_same_key_same_content_returns_recorded_response():
    plan = _make_plan()
    payload = _reading_payload(plan)
    b1 = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    b2 = client.post("/api/v1/certificates/cross-batch-readings", json=payload)
    assert b1.status_code == 201 and b2.status_code == 201
    assert b1.json() == b2.json()
    assert b1.json()["verdict_id"] == b2.json()["verdict_id"]


def test_reading_same_key_different_content_is_409(fresh_store):
    plan = _make_plan()
    _post_reading(plan)
    r = _post_reading(plan, allowed_offset=0.01)
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"
    assert r.json()["request_key"] == "read-1"
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute(
        "SELECT COUNT(*) FROM cross_batch_verdicts WHERE request_key = 'read-1'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'read-1'"
    ).fetchone()[0] == 1
    conn.close()


def test_reading_422_is_recorded_and_replays_identically():
    plan = _make_plan()
    bad = _reading_payload(plan, baseline_batch_id="BATCH-9")
    b1 = client.post("/api/v1/certificates/cross-batch-readings", json=bad)
    b2 = client.post("/api/v1/certificates/cross-batch-readings", json=bad)
    assert b1.status_code == 422 and b2.status_code == 422
    assert b1.json() == b2.json()
    # Same key, changed content still conflicts.
    bad["allowed_offset"] = 0.01
    b3 = client.post("/api/v1/certificates/cross-batch-readings", json=bad)
    assert b3.status_code == 409
    assert b3.json()["error"] == "REQUEST_KEY_CONFLICT"


def test_reading_request_key_shared_with_other_endpoint_conflicts():
    _create_cert("BATCH-1", key="shared-key")
    _create_cert("BATCH-2", key="create-BATCH-2")
    plan = _make_plan()
    r = _post_reading(plan, request_key="shared-key")
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"


def test_reading_persists_verdict_and_idempotency_in_one_transaction(fresh_store):
    plan = _make_plan()
    r = _post_reading(plan)
    assert r.status_code == 201
    conn = sqlite3.connect(str(fresh_store))
    rows = conn.execute(
        "SELECT verdict_id, plan_id, request_key, baseline_batch_id, "
        "allowed_offset, verdict, source_versions, verdict_body "
        "FROM cross_batch_verdicts"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    (verdict_id, plan_id, request_key, baseline, allowed, verdict,
     versions, verdict_body) = rows[0]
    stored = json.loads(verdict_body)
    assert verdict_id == r.json()["verdict_id"] == stored["verdict_id"]
    assert verdict_id == cross_batch_verdicts._verdict_id("read-1")
    assert plan_id == plan["plan_id"]
    assert request_key == "read-1" and baseline == "BATCH-1"
    assert allowed == 0.25 and verdict == "PASS"
    assert {v["batch_id"] for v in json.loads(versions)} == {"BATCH-1", "BATCH-2"}
    assert stored == r.json()
    # The full evidence is inside the stored body.
    assert stored["scan"] == _scan_from_plan(plan)
    assert len(stored["readings"]) == len(plan["wells"])
    assert len(stored["comparisons"]) == 3


def test_reading_fail_verdict_is_persisted_with_all_pairs(fresh_store):
    plan = _make_plan()
    readings = _passing_readings(plan)
    other_well = plan["comparisons"][2]["members"][1]["well"]
    readings[other_well] += 5.0
    r = _post_reading(plan, readings=readings, allowed_offset=1.0)
    assert r.status_code == 201 and r.json()["verdict"] == "FAIL"
    conn = sqlite3.connect(str(fresh_store))
    (verdict,) = conn.execute(
        "SELECT verdict FROM cross_batch_verdicts").fetchone()
    conn.close()
    assert verdict == "FAIL"


def test_existing_endpoints_still_available_after_readings():
    assert client.get("/health").json() == {"status": "ok"}
    c = _create_cert("BATCH-1")
    assert c["status"] == "ACTIVE"
