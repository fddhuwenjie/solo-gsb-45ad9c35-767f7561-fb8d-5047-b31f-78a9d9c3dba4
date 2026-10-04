"""Cross-batch comparison plate plan tests.

Two layers:

* the pure optimizer (``app.cross_batch`` CP-SAT stages) is checked against a
  brute-force enumeration of every item -> well assignment on random small
  plates (with blocked wells and 2-4 batches): it must first maximise
  same-comparison orthogonal adjacency edges, then maximise rows + columns
  covered per batch and per group; identical inputs/seed yield identical
  plans;
* the endpoint (``POST /api/v1/certificates/cross-batch-plans``) is checked
  for transaction semantics: idempotent replay, same-key different-content
  409, 404 without a certificate, 409 on REVOKED / stale revision, the full
  set of content 422s (duplicated batch/comparison, incomplete comparison,
  sample ineligible from the snapshot, duplicate sample across comparisons,
  unknown batch reference, capacity exceeded, bad blocked well), revision/
  digest provenance and single-transaction persistence.
"""
import itertools
import json
import random
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import certificates, cross_batch
from app.cross_batch import _PlanItem
from app.main import app

client = TestClient(app)

# Reuse the 4x4 certified-batch fixtures from the retest tests.
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


def _post_xcomp(**over):
    return client.post(
        "/api/v1/certificates/cross-batch-plans", json=_xcomp_payload(**over)
    )


# =========================================================== pure optimizer


def _solve_items(rows, cols, items, blocked=None, seed=0):
    """Run the three stages; return (proven flags, (adj, coverage), placement)."""
    blocked = blocked or set()
    pit = [
        _PlanItem(i, b, c, f"s{i}", g, 1, "d")
        for i, (b, g, c) in enumerate(items)
    ]
    bs = sorted({x.batch_id for x in pit})
    gs = sorted({x.group for x in pit})
    cs = sorted({x.comparison_id for x in pit})
    m1 = cross_batch._XCompModel(rows, cols, blocked, pit, bs, gs, cs)
    m1.add_hint(cross_batch._greedy_assignment(rows, cols, blocked, pit))
    m1.maximize_adjacency()
    s1, info1 = cross_batch._solve_stage("1", m1, seed)
    adj = int(s1.objective_value)
    m2 = cross_batch._XCompModel(rows, cols, blocked, pit, bs, gs, cs)
    m2.pin_adjacency(adj)
    m2.maximize_coverage()
    m2.add_hint(m1.assignment_from_solver(s1))
    s2, info2 = cross_batch._solve_stage("2", m2, seed)
    cov = int(s2.objective_value)
    m3 = cross_batch._XCompModel(rows, cols, blocked, pit, bs, gs, cs)
    m3.pin_adjacency(adj)
    m3.pin_coverage(cov)
    m3.add_hint(m2.assignment_from_solver(s2))
    s3, _ = cross_batch._solve_stage("3", m3, seed, first_solution=True)
    placement = frozenset(m3.assignment_from_solver(s3).items())
    return (info1[3], info2[3]), (adj, cov), placement


def _brute_optimum(rows, cols, items, blocked):
    """Enumerate every injective assignment; lex order: max adjacency, then
    max (batch + group) rows/columns coverage. Returns values and all optimum
    placements.
    """
    wells = [(r, c) for r in range(rows) for c in range(cols)
             if (r, c) not in blocked]
    best = None
    optima = set()
    for perm in itertools.permutations(wells, len(items)):
        pos = {i: perm[i] for i in range(len(items))}
        adj = sum(
            1
            for i in range(len(items))
            for j in range(i + 1, len(items))
            if items[i][2] == items[j][2]
            and abs(pos[i][0] - pos[j][0]) + abs(pos[i][1] - pos[j][1]) == 1
        )

        def cover(idx):
            total = 0
            for v in {items[i][idx] for i in range(len(items))}:
                total += len({pos[i][0] for i in range(len(items))
                              if items[i][idx] == v})
                total += len({pos[i][1] for i in range(len(items))
                              if items[i][idx] == v})
            return total

        cov = cover(0) + cover(1)
        key = (-adj, -cov)
        placement = frozenset(pos.items())
        if best is None or key < best:
            best = key
            optima = {placement}
        elif key == best:
            optima.add(placement)
    return (-best[0], -best[1]), optima


def test_optimizer_matches_bruteforce_on_random_small_plates():
    rng = random.Random(20261004)
    for trial in range(60):
        rows, cols = rng.randint(2, 3), rng.randint(2, 3)
        nb = rng.choice([2, 3])
        ncomp = rng.randint(1, 2)
        items = []
        for c in range(ncomp):
            for b in range(nb):
                items.append((f"B{b + 1}", rng.choice(["GA", "GB"]), f"C{c + 1}"))
        rng.shuffle(items)
        blocked = {
            (r, c)
            for r in range(rows) for c in range(cols)
            if rng.random() < 0.1
        }
        if len(items) > rows * cols - len(blocked):
            continue
        (want_adj, want_cov), optima = _brute_optimum(rows, cols, items, blocked)
        (p1, p2), (adj, cov), placement = _solve_items(
            rows, cols, items, blocked, seed=rng.randint(0, 9_999)
        )
        assert adj == want_adj, (trial, adj, want_adj)
        assert cov == want_cov, (trial, cov, want_cov)
        assert placement in optima, trial
        assert p1 and p2, (trial, "both stages must prove optimality")


def test_optimizer_is_deterministic_for_same_seed():
    items = []
    for c in ("C1", "C2", "C3"):
        items.append(("B1", "GA", c))
        items.append(("B2", "GB", c))
    _, _, p1 = _solve_items(4, 6, items, seed=123)
    _, _, p2 = _solve_items(4, 6, items, seed=123)
    assert p1 == p2


def test_optimizer_different_seeds_stay_lex_optimal():
    items = []
    for c in ("C1", "C2"):
        items.append(("B1", "GA", c))
        items.append(("B2", "GB", c))
    (want_adj, want_cov), _ = _brute_optimum(4, 5, items, set())
    placements = set()
    for seed in range(6):
        _, (adj, cov), placement = _solve_items(4, 5, items, seed=seed)
        assert (adj, cov) == (want_adj, want_cov)
        placements.add(placement)
    assert len(placements) >= 1


def test_optimizer_adjacency_has_priority_over_coverage():
    # Two comparisons of two members on a 2x4 plate. Clustering both pairs
    # adjacent (stage 1) is required even when spreading would cover more
    # lines at lower adjacency. Brute force confirms the documented values.
    items = [("B1", "GA", "C1"), ("B2", "GA", "C1"),
             ("B1", "GB", "C2"), ("B2", "GB", "C2")]
    (want_adj, want_cov), _ = _brute_optimum(2, 4, items, set())
    _, (adj, cov), _ = _solve_items(2, 4, items, seed=0)
    assert (adj, cov) == (want_adj, want_cov)
    # Every pair adjacent: at least two same-comparison edges.
    assert adj == 2


# ============================================================ HTTP endpoint


def test_xcomp_success_returns_full_body():
    c1 = _create_cert("BATCH-1")
    c2 = _create_cert("BATCH-2")
    r = _post_xcomp()
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["plan_id"].startswith("xplan-")
    assert body["status"] == "ACTIVE"
    assert body["request_key"] == "xcomp-1"
    assert body["seed"] == 42
    assert body["plate"]["type"] == "24"
    assert body["plate"]["rows"] == 4 and body["plate"]["columns"] == 6

    # Batch provenance: revision + digest for every source.
    sources = {b["batch_id"]: b for b in body["batches"]}
    assert set(sources) == {"BATCH-1", "BATCH-2"}
    for bid, cert in (("BATCH-1", c1), ("BATCH-2", c2)):
        assert sources[bid]["certificate_id"] == cert["certificate_id"]
        assert sources[bid]["status"] == "ACTIVE"
        assert sources[bid]["revision"] == 1
        assert sources[bid]["expected_revision"] == 1
        assert sources[bid]["digest"] == cert["digest"]

    # All six members placed, exactly once, layout == samples (row-major).
    assert len(body["samples"]) == 6
    assert len(body["wells"]) == 6
    assert len(set(body["wells"])) == 6
    for entry in body["samples"]:
        assert body["layout"][entry["qualified_sample_id"]] == entry["well"]
        assert entry["qualified_sample_id"] == f"{entry['batch_id']}:{entry['sample_id']}"
    assert [s["well"] for s in body["samples"]] == body["wells"]

    # Every comparison kept together as an adjacent orthogonal pair.
    lvl1 = body["metrics"]["level1_comparison_adjacency"]
    edges = lvl1["same_comparison_adjacent_edges"]
    assert lvl1["same_comparison_adjacent_edges_count"] == 3 == len(edges)
    assert {e["comparison_id"] for e in edges} == {"C1", "C2", "C3"}
    for e in edges:
        assert len(e["wells"]) == 2 and len(e["members"]) == 2

    # Coverage report is internally consistent.
    lvl2 = body["metrics"]["level2_coverage"]
    by_batch = lvl2["by_batch"]
    by_group = lvl2["by_group"]
    assert set(by_batch) == {"BATCH-1", "BATCH-2"}
    recomputed = 0
    well_pos = {w: cross_batch.parse_well(w, 4, 6) for w in body["wells"]}
    for bid in ("BATCH-1", "BATCH-2"):
        wells = [s["well"] for s in body["samples"] if s["batch_id"] == bid]
        rows = {well_pos[w][0] for w in wells}
        cols = {well_pos[w][1] for w in wells}
        assert by_batch[bid]["rows_covered"] == len(rows)
        assert by_batch[bid]["columns_covered"] == len(cols)
        recomputed += len(rows) + len(cols)
    for g in ("GA", "GB"):
        wells = [s["well"] for s in body["samples"] if s["group"] == g]
        rows = {well_pos[w][0] for w in wells}
        cols = {well_pos[w][1] for w in wells}
        assert by_group[g]["rows_covered"] == len(rows)
        assert by_group[g]["columns_covered"] == len(cols)
        recomputed += len(rows) + len(cols)
    assert lvl2["covered_rows_columns_total"] == recomputed

    stages = body["solver"]["stages"]
    assert [s["stage"] for s in stages] == [
        "1_comparison_adjacency", "2_coverage", "3_tiebreak"]
    assert body["solver"]["status"] in ("OPTIMAL", "FEASIBLE")
    # The objective values are exactly the lex-optimal ones regardless of
    # whether CP-SAT finished proving them inside the deterministic budget.
    assert stages[0]["objective_value"] == 3
    assert stages[1]["objective_value"] == 22


def test_xcomp_persists_plan_and_idempotency_in_one_transaction(fresh_store):
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    r = _post_xcomp()
    assert r.status_code == 201
    conn = sqlite3.connect(str(fresh_store))
    rows = conn.execute(
        "SELECT plan_id, batch_ids, revisions, seed, plan_body "
        "FROM cross_batch_plans"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    plan_id, batch_ids, revisions, seed, plan_body = rows[0]
    stored = json.loads(plan_body)
    assert plan_id == r.json()["plan_id"] == stored["plan_id"]
    assert json.loads(batch_ids) == ["BATCH-1", "BATCH-2"]
    assert json.loads(revisions) == {"BATCH-1": 1, "BATCH-2": 1}
    assert seed == 42
    assert stored == r.json()


def test_xcomp_replay_same_key_same_content_returns_recorded_response():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload()
    b1 = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    b2 = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert b1.status_code == 201 and b2.status_code == 201
    assert b1.json() == b2.json()
    assert b1.json()["plan_id"] == b2.json()["plan_id"]


def test_xcomp_same_key_different_content_is_409_and_saves_nothing(fresh_store):
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    client.post("/api/v1/certificates/cross-batch-plans", json=_xcomp_payload())
    changed = _xcomp_payload(seed=43)
    r = client.post("/api/v1/certificates/cross-batch-plans", json=changed)
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"
    assert r.json()["request_key"] == "xcomp-1"
    conn = sqlite3.connect(str(fresh_store))
    n_plans = conn.execute("SELECT COUNT(*) FROM cross_batch_plans").fetchone()[0]
    n_keys = conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'xcomp-1'"
    ).fetchone()[0]
    conn.close()
    assert n_plans == 1 and n_keys == 1


def test_xcomp_422_is_recorded_and_replays_identically(fresh_store):
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    # Missing one batch member in C3 -> content 422 recorded under the key.
    bad = _xcomp_payload()
    bad["comparisons"][2]["members"] = [_member("BATCH-1", "gA5")]
    b1 = client.post("/api/v1/certificates/cross-batch-plans", json=bad)
    b2 = client.post("/api/v1/certificates/cross-batch-plans", json=bad)
    assert b1.status_code == 422 and b2.status_code == 422
    assert b1.json() == b2.json()
    assert "XCOMP_COMPARISON_INCOMPLETE" in b1.json()["infeasibility_categories"]
    conn = sqlite3.connect(str(fresh_store))
    n_plans = conn.execute("SELECT COUNT(*) FROM cross_batch_plans").fetchone()[0]
    conn.close()
    assert n_plans == 0
    # Same key, different content still conflicts.
    changed = json.loads(json.dumps(bad))
    changed["seed"] = 5
    assert client.post(
        "/api/v1/certificates/cross-batch-plans", json=changed).status_code == 409


def test_xcomp_unknown_batch_is_404():
    _create_cert("BATCH-1")
    r = _post_xcomp()  # BATCH-2 never certified
    assert r.status_code == 404
    assert r.json()["error"] == "CERTIFICATE_NOT_FOUND"
    assert r.json()["batch_id"] in {"BATCH-1", "BATCH-2"}


def test_xcomp_revoked_certificate_is_409():
    cert = _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-1", "request_key": "rv-1", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items() if w != "A1"},
        "readings": _readings(),
    })
    r = _post_xcomp()
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVOKED"
    assert body["certificate_id"] == cert["certificate_id"]
    assert body["status"] == "REVOKED"
    assert body["revoked_reason"] == "SAMPLES_MISSING"


def test_xcomp_stale_expected_revision_is_409_and_saves_nothing(fresh_store):
    c1 = _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    new_readings = dict(_readings())
    new_readings["A1"] = 1.0
    rv = client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-1", "request_key": "rv-1", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": new_readings,
    })
    assert rv.status_code == 200 and rv.json()["revision"] == 2

    r = _post_xcomp()  # still expects revision 1 for BATCH-1
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVISION_MISMATCH"
    assert body["batch_id"] == "BATCH-1"
    assert body["current_revision"] == 2 and body["expected_revision"] == 1
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute("SELECT COUNT(*) FROM cross_batch_plans").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'xcomp-1'"
    ).fetchone()[0] == 0
    conn.close()

    # Correcting the revision succeeds (state 409 does not burn the key).
    ok = _post_xcomp(
        request_key="xcomp-fixed",
        batches=[
            {"batch_id": "BATCH-1", "expected_revision": 2},
            {"batch_id": "BATCH-2", "expected_revision": 1},
        ],
    )
    assert ok.status_code == 201
    sources = {b["batch_id"]: b for b in ok.json()["batches"]}
    assert sources["BATCH-1"]["revision"] == 2
    assert sources["BATCH-1"]["digest"] == rv.json()["digest"] != c1["digest"]


def test_xcomp_replay_after_revision_advances_returns_409():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload()
    first = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert first.status_code == 201
    new_readings = dict(_readings())
    new_readings["A1"] = 1.0
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-2", "request_key": "rv-2", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": new_readings,
    })
    replay = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert replay.status_code == 409
    assert replay.json()["error"] == "CERTIFICATE_REVISION_MISMATCH"


def test_xcomp_duplicated_batch_in_request_is_422():
    _create_cert("BATCH-1")
    payload = _xcomp_payload(batches=[
        {"batch_id": "BATCH-1", "expected_revision": 1},
        {"batch_id": "BATCH-1", "expected_revision": 1},
    ])
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert "XCOMP_BATCH_DUPLICATED" in r.json()["infeasibility_categories"]


def test_xcomp_duplicated_comparison_id_is_422():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload()
    payload["comparisons"][1]["comparison_id"] = "C1"
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert "XCOMP_COMPARISON_DUPLICATED" in r.json()["infeasibility_categories"]


def test_xcomp_member_referencing_unknown_batch_is_422():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload()
    payload["comparisons"][0]["members"][0]["batch_id"] = "BATCH-9"
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert "XCOMP_SAMPLE_UNKNOWN_BATCH" in r.json()["infeasibility_categories"]


def test_xcomp_missing_member_batch_is_422():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload()
    payload["comparisons"][0]["members"] = [_member("BATCH-1", "gA1")]
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "XCOMP_COMPARISON_INCOMPLETE" in cats


def test_xcomp_duplicated_batch_member_in_comparison_is_422():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload()
    payload["comparisons"][0]["members"] = [
        _member("BATCH-1", "gA1"), _member("BATCH-1", "gA3")]
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert "XCOMP_COMPARISON_INCOMPLETE" in r.json()["infeasibility_categories"]


def test_xcomp_same_sample_in_two_comparisons_is_422():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload()
    # BATCH-1 gA1 used by both C1 and C3.
    payload["comparisons"][2]["members"][0]["sample_id"] = "gA1"
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert "XCOMP_SAMPLE_DUPLICATED" in r.json()["infeasibility_categories"]


def test_xcomp_sample_ineligible_from_snapshot_is_422(fresh_store):
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    # Tamper the snapshot: drop the reading at gA1's planned well (A1) in
    # BATCH-1, making gA1 ineligible.
    conn = sqlite3.connect(str(fresh_store))
    row = conn.execute(
        "SELECT snapshot FROM certificates WHERE batch_id = 'BATCH-1'"
    ).fetchone()
    snap = json.loads(row[0])
    snap["readings"]["A1"] = None
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

    r = _post_xcomp(request_key="xcomp-null")
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert "XCOMP_SAMPLE_NOT_ELIGIBLE" in cats
    assert any("gA1" in m and "BATCH-1" in m for m in r.json()["messages"])


def test_xcomp_sample_missing_from_scan_is_422(fresh_store):
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    # Swap scan at A1/A2 in BATCH-1 snapshot: gA1 loses its planned well.
    conn = sqlite3.connect(str(fresh_store))
    row = conn.execute(
        "SELECT snapshot FROM certificates WHERE batch_id = 'BATCH-1'"
    ).fetchone()
    snap = json.loads(row[0])
    snap["scan"]["A1"], snap["scan"]["A2"] = snap["scan"]["A2"], snap["scan"]["A1"]
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

    r = _post_xcomp(request_key="xcomp-scan")
    assert r.status_code == 422
    # gA1 (planned at A1) is now misplaced -> ineligible.
    assert "XCOMP_SAMPLE_NOT_ELIGIBLE" in r.json()["infeasibility_categories"]


def test_xcomp_unknown_sample_id_is_422():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    payload = _xcomp_payload()
    payload["comparisons"][0]["members"][0]["sample_id"] = "nope"
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert "XCOMP_SAMPLE_NOT_ELIGIBLE" in r.json()["infeasibility_categories"]


def test_xcomp_capacity_exceeded_is_422():
    for b in ("BATCH-1", "BATCH-2", "BATCH-3", "BATCH-4"):
        _create_cert(b)
    ga = ["gA1", "gA2", "gA3", "gA4", "gA5", "gA6", "gA7", "gA8"]
    gb = ["gB1", "gB2", "gB3", "gB4", "gB5", "gB6", "gB7", "gB8"]
    comparisons = []
    for i in range(3):
        # Three distinct samples per batch across the three comparisons.
        members = [
            _member("BATCH-1", [ga[0], gb[0], ga[4]][i]),
            _member("BATCH-2", [ga[1], gb[1], gb[5]][i]),
            _member("BATCH-3", [ga[2], gb[2], ga[6]][i]),
            _member("BATCH-4", [ga[3], gb[3], gb[7]][i]),
        ]
        comparisons.append({"comparison_id": f"C{i + 1}", "members": members})
    payload = _xcomp_payload(
        batches=[{"batch_id": f"BATCH-{b}", "expected_revision": 1}
                 for b in range(1, 5)],
        comparisons=comparisons,
        plate={"type": "custom", "rows": 2, "columns": 5},  # 10 wells
    )
    # 3 comparisons * 4 batches = 12 samples > 10 wells.
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert "CAPACITY_EXCEEDED" in r.json()["infeasibility_categories"]


def test_xcomp_bad_blocked_well_is_422():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    r = _post_xcomp(blocked_wells=["Z99"])
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]


def test_xcomp_blocked_wells_are_respected():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    r = _post_xcomp(
        request_key="xcomp-blocked",
        blocked_wells=["A1", "A2", "B1", "B2", "C1", "C2"])
    assert r.status_code == 201, r.text
    assert set(r.json()["wells"]).isdisjoint(
        {"A1", "A2", "B1", "B2", "C1", "C2"})
    assert r.json()["plate"]["blocked_wells"] == ["A1", "A2", "B1", "B2", "C1", "C2"]


def test_xcomp_three_batches_places_three_member_comparisons():
    for b in ("BATCH-1", "BATCH-2", "BATCH-3"):
        _create_cert(b)
    payload = _xcomp_payload(
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
        request_key="xcomp-3",
    )
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert len(body["samples"]) == 6
    comps = {c["comparison_id"]: c for c in body["comparisons"]}
    assert {len(c["members"]) for c in comps.values()} == {3}
    # Each three-member comparison achieves at least two orthogonal edges
    # (a connected triomino), which is the stage-1 maximum possible.
    counts = {}
    for e in body["metrics"]["level1_comparison_adjacency"][
            "same_comparison_adjacent_edges"]:
        counts[e["comparison_id"]] = counts.get(e["comparison_id"], 0) + 1
    assert counts == {"C1": 2, "C2": 2}


def test_xcomp_plan_id_is_deterministic_from_request_key():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    body = _post_xcomp().json()
    assert body["plan_id"] == cross_batch._plan_id("xcomp-1")


def test_xcomp_seed_determinism_across_restarts():
    _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    p1 = _post_xcomp(request_key="seed-a").json()
    certificates.reset_stores()
    # Fresh key, identical content: identical layout even after a restart.
    p2 = _post_xcomp(request_key="seed-b").json()
    assert p1["layout"] == p2["layout"]


def test_xcomp_request_key_shared_with_other_endpoint_conflicts():
    _create_cert("BATCH-1", key="shared-key")
    _create_cert("BATCH-2", key="create-BATCH-2")
    r = _post_xcomp(request_key="shared-key")
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"


# ------------------------------------------------------------ request shape


def test_xcomp_requires_two_to_four_batches():
    payload = _xcomp_payload(
        batches=[{"batch_id": "BATCH-1", "expected_revision": 1}])
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_xcomp_unknown_field_is_validation_error():
    payload = _xcomp_payload()
    payload["extra"] = 1
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_xcomp_missing_request_key_is_validation_error():
    payload = _xcomp_payload()
    del payload["request_key"]
    r = client.post("/api/v1/certificates/cross-batch-plans", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_existing_endpoints_still_available_after_xcomp():
    assert client.get("/health").json() == {"status": "ok"}
    _create_cert("BATCH-1")
    r = client.post(
        "/api/v1/arrangements/batch-release", json=_release_payload())
    assert r.status_code == 200 and r.json()["verdict"] == "RELEASED"
