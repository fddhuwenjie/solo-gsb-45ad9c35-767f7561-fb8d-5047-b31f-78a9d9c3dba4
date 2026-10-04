"""Cross-batch comparison plate plans spanning several ACTIVE batches.

``POST /api/v1/certificates/cross-batch-plans`` takes an idempotency
``request_key``, 2-4 ACTIVE batch certificates (each pinned to an
``expected_revision``), a list of ``comparisons`` — every comparison names
exactly one sample per participating batch — a target plate footprint,
blocked wells and a seed.

For every named batch the service checks, inside one SQLite transaction,
that the certificate is still ACTIVE at the expected current revision, and
confirms from that revision's immutable snapshot that each requested sample
is still on its planned well, that the frozen scan agrees and that the well
carries a valid (finite, non-null) reading. Any stale revision, duplicated or
missing sample, unknown batch reference or infeasibility fails the whole
request and nothing is saved.

On success the samples are placed on the (new) target plate optimising,
lexicographically (CP-SAT):

1. maximise the number of orthogonal adjacency edges (up/down/left/right)
   whose two endpoints belong to the *same comparison*;
2. maximise the rows + columns covered by every batch and by every
   experiment group (a sum of coverage indicators);
3. a deterministic, seed-governed tie-break (the stage-2 optimum is fed back
   as a complete hint and the first solution instantiates it), so identical
   (revision snapshots, comparisons, plate, blocked wells, seed) always
   yields the identical plate.

The certificate state reads, the frozen snapshot lookups, the plan insert and
the idempotency record all commit (or roll back) in one SQLite transaction.
State-dependent rejections (no certificate / REVOKED / stale revision) are not
recorded under the ``request_key``; content-dependent 422 responses are.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

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
from .planner import InfeasibleArrangement, SolverTimeout, _make_solver, _resolve_dimensions
from .retest import _eligible_candidates
from .schemas import (
    CERTIFICATE_NOT_FOUND,
    CERTIFICATE_REVISION_MISMATCH,
    CERTIFICATE_REVOKED,
    VIOLATION_CAPACITY_EXCEEDED,
    VIOLATION_WELL_REFERENCE_INVALID,
    VIOLATION_XCOMP_BATCH_DUPLICATED,
    VIOLATION_XCOMP_COMPARISON_DUPLICATED,
    VIOLATION_XCOMP_COMPARISON_INCOMPLETE,
    VIOLATION_XCOMP_NO_SOLUTION,
    VIOLATION_XCOMP_SAMPLE_DUPLICATED,
    VIOLATION_XCOMP_SAMPLE_NOT_ELIGIBLE,
    VIOLATION_XCOMP_SAMPLE_UNKNOWN_BATCH,
    CrossBatchPlanRequest,
)

_ENDPOINT_XCOMP = "POST /api/v1/certificates/cross-batch-plans"


class CrossBatchPlanInvalid(Exception):
    """The cross-batch comparison request cannot be planned (HTTP 422).

    Raised by the pure builder; no plan row or idempotency record is written
    by the builder itself — the endpoint transaction records the 422 under the
    request key.
    """

    def __init__(self, categories: list[str], messages: list[str]):
        self.categories = categories
        self.messages = messages
        super().__init__("; ".join(messages) if messages else "invalid cross-batch plan request")


def cross_batch_invalid_body(exc: CrossBatchPlanInvalid) -> dict:
    """Canonical 422 envelope for an unplannable cross-batch request."""
    return {
        "detail": "No cross-batch comparison plate plan can be produced; "
        "no plan was saved.",
        "infeasibility_categories": exc.categories,
        "messages": exc.messages,
    }


@dataclass(frozen=True)
class _PlanItem:
    """One placed sample: a (batch, comparison) contribution to the plate."""

    index: int
    batch_id: str
    comparison_id: str
    sample_id: str
    group: str
    source_revision: int
    source_digest: str


@dataclass
class CrossBatchLayout:
    rows: int
    columns: int
    available_wells: list[str]
    items: list[_PlanItem]
    placement: dict[int, tuple[int, int]]  # item index -> (row, col)
    batches: list[str]
    groups: list[str]
    comparison_ids: list[str]
    adjacent_pairs: list[dict]
    adjacent_edge_count: int
    total_orthogonal_edges: int
    batch_coverage: dict[str, dict]
    group_coverage: dict[str, dict]
    coverage_total: int
    stages: list[tuple[str, int, str, bool, float]] = field(default_factory=list)


# -------------------------------------------------------------- model build


def _eligible_pool(snapshot: dict, rows: int, cols: int) -> dict[str, dict]:
    """sample_id -> {well, group} for every eligible sample in a snapshot.

    Eligibility mirrors the re-test plan exactly: the frozen scan must still
    place the sample on its planned well, the well must not be blocked and it
    must carry a valid (present, non-null, finite) reading.
    """
    candidates = _eligible_candidates(snapshot, rows, cols, set())
    return {
        c.sample_id: {"well": c.well, "group": c.group, "pos": (c.row, c.col)}
        for c in candidates
    }


def build_items(
    req: CrossBatchPlanRequest,
    certs: dict[str, dict],
    revisions: dict[str, dict],
) -> tuple[list[_PlanItem], dict[str, dict]]:
    """Validate request content against frozen snapshots and build the pool.

    ``certs[batch_id]`` are ACTIVE certificate rows (state already checked by
    the endpoint transaction); ``revisions[batch_id]`` holds
    ``{"digest": str, "snapshot": dict}`` of every batch's current revision.

    Returns the canonical, sorted item list (one entry per (comparison, batch)
    member) together with per-batch source metadata. Raises
    ``CrossBatchPlanInvalid`` aggregating every content problem found.
    """
    categories: list[str] = []
    messages: list[str] = []

    def add(category: str, msg: str) -> None:
        if category not in categories:
            categories.append(category)
        messages.append(msg)

    batch_refs = req.batches
    batch_ids = [b.batch_id for b in batch_refs]
    expected = {b.batch_id: b.expected_revision for b in batch_refs}

    if len(set(batch_ids)) != len(batch_ids):
        dupes = sorted({b for b in batch_ids if batch_ids.count(b) > 1})
        add(
            VIOLATION_XCOMP_BATCH_DUPLICATED,
            f"batch id(s) listed more than once in batches: {', '.join(dupes)}",
        )

    comp_ids = [c.comparison_id for c in req.comparisons]
    if len(set(comp_ids)) != len(comp_ids):
        dupes = sorted({c for c in comp_ids if comp_ids.count(c) > 1})
        add(
            VIOLATION_XCOMP_COMPARISON_DUPLICATED,
            f"comparison_id(s) listed more than once: {', '.join(dupes)}",
        )

    # One eligible pool per snapshot. Snapshot plate geometries belong to the
    # batches' own certified plates and are used only to verify members; the
    # target plate is validated separately by the caller.
    pools: dict[str, dict[str, dict]] = {}
    for bid in set(batch_ids):
        rev = revisions[bid]
        snap = rev["snapshot"]
        srows = snap["plate"]["rows"]
        scols = snap["plate"]["columns"]
        pools[bid] = _eligible_pool(snap, srows, scols)

    items: list[_PlanItem] = []
    seen_pairs: set[tuple[str, str]] = set()
    for comp in req.comparisons:
        cid = comp.comparison_id
        member_batches = [m.batch_id for m in comp.members]
        # A member referencing a batch not named in `batches` is structurally
        # unknown; duplicate members inside one comparison collapse below.
        unknown = [b for b in member_batches if b not in expected]
        for bid in unknown:
            add(
                VIOLATION_XCOMP_SAMPLE_UNKNOWN_BATCH,
                f"comparison '{cid}' references batch '{bid}', which is not "
                "listed in batches",
            )

        missing = [b for b in batch_ids if b not in member_batches]
        if unknown or missing or len(set(member_batches)) != len(member_batches):
            if len(set(member_batches)) != len(member_batches):
                dupb = sorted({b for b in member_batches if member_batches.count(b) > 1})
                add(
                    VIOLATION_XCOMP_COMPARISON_INCOMPLETE,
                    f"comparison '{cid}' names batch(es) more than once: "
                    f"{', '.join(dupb)}; exactly one sample per batch is required",
                )
            if missing:
                add(
                    VIOLATION_XCOMP_COMPARISON_INCOMPLETE,
                    f"comparison '{cid}' must name exactly one sample for every "
                    f"participating batch; missing batch(es): {', '.join(missing)}",
                )
            # Without a complete, well-formed member set the per-member checks
            # below are meaningless for this comparison.
            continue

        for m in comp.members:
            bid, sid = m.batch_id, m.sample_id
            pair = (bid, sid)
            if pair in seen_pairs:
                add(
                    VIOLATION_XCOMP_SAMPLE_DUPLICATED,
                    f"batch '{bid}' sample '{sid}' is used by more than one "
                    "comparison; a physical sample can occupy only one target well",
                )
                continue
            entry = pools[bid].get(sid)
            if entry is None:
                add(
                    VIOLATION_XCOMP_SAMPLE_NOT_ELIGIBLE,
                    f"batch '{bid}' sample '{sid}' is not eligible from revision "
                    f"{expected[bid]}'s frozen snapshot: it must exist, still occupy "
                    "its planned well in the frozen scan and have a valid finite "
                    "reading at an unblocked well",
                )
                continue
            seen_pairs.add(pair)
            items.append(
                _PlanItem(
                    index=-1,
                    batch_id=bid,
                    comparison_id=cid,
                    sample_id=sid,
                    group=entry["group"],
                    source_revision=revisions[bid]["revision"],
                    source_digest=revisions[bid]["digest"],
                )
            )

    if categories:
        raise CrossBatchPlanInvalid(categories, messages)

    # Canonical order: comparison id first (so a greedy row-major fill clusters
    # each comparison), then batch id, then sample id.
    items.sort(key=lambda it: (it.comparison_id, it.batch_id, it.sample_id))
    items = [
        _PlanItem(i, it.batch_id, it.comparison_id, it.sample_id, it.group,
                  it.source_revision, it.source_digest)
        for i, it in enumerate(items)
    ]
    return items, expected


# ------------------------------------------------------------------- CP-SAT


class _XCompModel:
    """CP-SAT model for one lexicographic stage of the comparison plate."""

    def __init__(self, rows: int, cols: int, blocked: set[tuple[int, int]],
                 items: list[_PlanItem], batches: list[str], groups: list[str],
                 comparisons: list[str]):
        self.rows = rows
        self.cols = cols
        self.items = items
        self.n = len(items)
        self.batches = batches
        self.groups = groups
        self.comparisons = comparisons
        self.batch_idx = {b: i for i, b in enumerate(batches)}
        self.group_idx = {g: i for i, g in enumerate(groups)}
        self.model = cp_model.CpModel()

        self.wells = [
            (r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked
        ]
        self.well_index = {w: i for i, w in enumerate(self.wells)}

        # x[i][j]: item i occupies well j.
        self.x = [
            [self.model.new_bool_var(f"x_{i}_{j}") for j in range(len(self.wells))]
            for i in range(self.n)
        ]
        self.occ = {w: self.model.new_bool_var(f"occ_{w[0]}_{w[1]}") for w in self.wells}
        for i in range(self.n):
            self.model.add_exactly_one(self.x[i])
        for j, w in enumerate(self.wells):
            col = [self.x[i][j] for i in range(self.n)]
            self.model.add(sum(col) <= 1)
            self.model.add(self.occ[w] == sum(col))
        # Note: no "first well occupied" symmetry break is imposed. With
        # blocked wells the top-left well can be an isolated island whose
        # forced occupancy destroys an objective optimum, and even without
        # blocked wells a translated optimal shape need not cover A1. The
        # snake warm start already supplies an adjacency-optimal feasible
        # solution, and CP-SAT symmetry detection handles remaining symmetry.

        self._build_adjacency()
        self._build_coverage()

    # ------------------------------------------------------------ adjacency
    def _build_adjacency(self) -> None:
        # Orthogonal undirected edges between available wells.
        well_set = set(self.wells)
        self.edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r, c in self.wells:
            if (r + 1, c) in well_set:
                self.edges.append(((r, c), (r + 1, c)))
            if (r, c + 1) in well_set:
                self.edges.append(((r, c), (r, c + 1)))

        # y_by_edge[ei][(i,j)] for two distinct items of the same comparison;
        # it is 1 iff items i and j occupy wells u and v (in either order).
        # Four implications linearise this over item pairs (instead of well
        # labels with reification), which keeps the model small and trivially
        # propagable: at most one pair can occupy any one edge.
        self.y_by_edge: list[dict[tuple[int, int], object]] = []
        items_by_comparison: dict[str, list[int]] = {}
        for i, it in enumerate(self.items):
            items_by_comparison.setdefault(it.comparison_id, []).append(i)
        for ei, (u, v) in enumerate(self.edges):
            ju, jv = self.well_index[u], self.well_index[v]
            edge_ys: dict[tuple[int, int], object] = {}
            for iis in items_by_comparison.values():
                for ai in range(len(iis)):
                    for bi in range(ai + 1, len(iis)):
                        i, j = iis[ai], iis[bi]
                        y = self.model.new_bool_var(f"adj_{ei}_{i}_{j}")
                        # y=1 requires one of {i,j} at u and the other at v.
                        self.model.add(y <= self.x[i][ju] + self.x[j][ju])
                        self.model.add(y <= self.x[i][jv] + self.x[j][jv])
                        # Either ordering forces y=1.
                        self.model.add(y >= self.x[i][ju] + self.x[j][jv] - 1)
                        self.model.add(y >= self.x[j][ju] + self.x[i][jv] - 1)
                        edge_ys[(i, j)] = y
            self.y_by_edge.append(edge_ys)
        all_y = [y for edge in self.y_by_edge for y in edge.values()]
        self.adjacency = self.model.new_int_var(0, max(len(self.edges), 0), "adjacency")
        self.model.add(self.adjacency == sum(all_y))

    # ------------------------------------------------------------- coverage
    def _build_coverage(self) -> None:
        # Per-batch and per-group occupancy booleans on each row/column.
        self.batch_row = {b: [] for b in self.batches}
        self.batch_col = {b: [] for b in self.batches}
        self.group_row = {g: [] for g in self.groups}
        self.group_col = {g: [] for g in self.groups}
        br_vars = {b: {r: [] for r in range(self.rows)} for b in self.batches}
        bc_vars = {b: {c: [] for c in range(self.cols)} for b in self.batches}
        gr_vars = {g: {r: [] for r in range(self.rows)} for g in self.groups}
        gc_vars = {g: {c: [] for c in range(self.cols)} for g in self.groups}
        for i, it in enumerate(self.items):
            for j, (r, c) in enumerate(self.wells):
                br_vars[it.batch_id][r].append(self.x[i][j])
                bc_vars[it.batch_id][c].append(self.x[i][j])
                gr_vars[it.group][r].append(self.x[i][j])
                gc_vars[it.group][c].append(self.x[i][j])

        self._cover_vars: list = []
        for b in self.batches:
            for r in range(self.rows):
                v = self.model.new_bool_var(f"br_{self.batch_idx[b]}_{r}")
                xs = br_vars[b][r]
                self.model.add(v <= sum(xs))
                for x in xs:
                    self.model.add(v >= x)
                self.batch_row[b].append(v)
                self._cover_vars.append(v)
            for c in range(self.cols):
                v = self.model.new_bool_var(f"bc_{self.batch_idx[b]}_{c}")
                xs = bc_vars[b][c]
                self.model.add(v <= sum(xs))
                for x in xs:
                    self.model.add(v >= x)
                self.batch_col[b].append(v)
                self._cover_vars.append(v)
        for g in self.groups:
            for r in range(self.rows):
                v = self.model.new_bool_var(f"gr_{self.group_idx[g]}_{r}")
                xs = gr_vars[g][r]
                self.model.add(v <= sum(xs))
                for x in xs:
                    self.model.add(v >= x)
                self.group_row[g].append(v)
                self._cover_vars.append(v)
            for c in range(self.cols):
                v = self.model.new_bool_var(f"gc_{self.group_idx[g]}_{c}")
                xs = gc_vars[g][c]
                self.model.add(v <= sum(xs))
                for x in xs:
                    self.model.add(v >= x)
                self.group_col[g].append(v)
                self._cover_vars.append(v)

        upper = (len(self.batches) + len(self.groups)) * (self.rows + self.cols)
        self.coverage = self.model.new_int_var(0, upper, "coverage")
        self.model.add(self.coverage == sum(self._cover_vars))

    # ------------------------------------------------------------- objectives
    def maximize_adjacency(self) -> None:
        self.model.maximize(self.adjacency)

    def pin_adjacency(self, value: int) -> None:
        self.model.add(self.adjacency == value)

    def maximize_coverage(self) -> None:
        self.model.maximize(self.coverage)

    def pin_coverage(self, value: int) -> None:
        self.model.add(self.coverage == value)

    # ----------------------------------------------------------------- hints
    def add_hint(self, assignment: dict[int, tuple[int, int]]) -> None:
        """Pin every variable from a complete item -> well assignment."""
        occupied = set(assignment.values())
        item_at = {pos: self.items[i] for i, pos in assignment.items()}
        for i, it in enumerate(self.items):
            j = self.well_index[assignment[i]]
            for k in range(len(self.wells)):
                self.model.add_hint(self.x[i][k], 1 if k == j else 0)
        for w in self.wells:
            self.model.add_hint(self.occ[w], 1 if w in occupied else 0)
        # Hint every edge/pair adjacency boolean from the assignment.
        well_to_item = {pos: i for i, pos in assignment.items()}
        adj_count = 0
        for ei, (u, v) in enumerate(self.edges):
            iu, iv = well_to_item.get(u), well_to_item.get(v)
            for (i, j), yvar in self.y_by_edge[ei].items():
                value = int(
                    iu is not None and iv is not None
                    and ((iu == i and iv == j) or (iu == j and iv == i))
                )
                self.model.add_hint(yvar, value)
                adj_count += value
        self.model.add_hint(self.adjacency, adj_count)
        # Hint every coverage indicator with its value under the assignment
        # (self._cover_vars is exactly these batch/group row/column booleans).
        br = {b: set() for b in self.batches}
        bc = {b: set() for b in self.batches}
        grr = {g: set() for g in self.groups}
        gcc = {g: set() for g in self.groups}
        for (r, c), it in item_at.items():
            br[it.batch_id].add(r)
            bc[it.batch_id].add(c)
            grr[it.group].add(r)
            gcc[it.group].add(c)
        for b in self.batches:
            for r in range(self.rows):
                self.model.add_hint(self.batch_row[b][r], 1 if r in br[b] else 0)
            for c in range(self.cols):
                self.model.add_hint(self.batch_col[b][c], 1 if c in bc[b] else 0)
        for g in self.groups:
            for r in range(self.rows):
                self.model.add_hint(self.group_row[g][r], 1 if r in grr[g] else 0)
            for c in range(self.cols):
                self.model.add_hint(self.group_col[g][c], 1 if c in gcc[g] else 0)
        coverage_value = sum(len(s) for s in list(br.values()) + list(bc.values())
                             + list(grr.values()) + list(gcc.values()))
        self.model.add_hint(self.coverage, coverage_value)

    def assignment_from_solver(self, solver) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for i in range(self.n):
            for j, w in enumerate(self.wells):
                if solver.value(self.x[i][j]) == 1:
                    out[i] = w
                    break
        return out


def _snake_wells(rows: int, cols: int,
                 blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Boustrophedon (snake) ordering of the available wells.

    Consecutive positions in this ordering are always orthogonal neighbours
    (including the turn from one row to the next), so handing consecutive
    wells to the members of one comparison gives a maximally connected warm
    start (b members on b-1 adjacent edges).
    """
    out: list[tuple[int, int]] = []
    for r in range(rows):
        ordered = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in ordered:
            if (r, c) not in blocked:
                out.append((r, c))
    return out


def _greedy_assignment(rows: int, cols: int, blocked: set[tuple[int, int]],
                       items: list[_PlanItem]) -> dict[int, tuple[int, int]]:
    """Deterministic feasible warm start: each comparison gets a contiguous
    snake-order block, so every b-member comparison occupies b-1 orthogonal
    edges (the per-comparison maximum); comparisons are packed in canonical
    order. This is an adjacency-optimal (stage-1) feasible assignment.
    """
    snake = _snake_wells(rows, cols, blocked)
    return {it.index: snake[it.index] for it in items}


def _solve_stage(stage: str, model: _XCompModel, seed: int, *,
                 first_solution: bool = False):
    det_limit = float(os.getenv("CP_DETERMINISTIC_TIME_SECONDS", "4"))
    wall_limit = float(os.getenv("CP_WALL_TIME_SECONDS", "12"))
    fast_det = float(os.getenv("CP_FAST_DETERMINISTIC_TIME_SECONDS", "1"))
    fast_wall = float(os.getenv("CP_FAST_WALL_TIME_SECONDS", "8"))
    if first_solution:
        solver = _make_solver(seed, fast_det, fast_wall,
                              stop_at_first=True, fast_presolve=True)
        t0 = time.perf_counter()
        status = solver.solve(model.model)
        elapsed = time.perf_counter() - t0
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return solver, (
                stage, -1, solver.status_name(status),
                status == cp_model.OPTIMAL, elapsed,
            )
        if status == cp_model.INFEASIBLE:
            raise CrossBatchPlanInvalid(
                [VIOLATION_XCOMP_NO_SOLUTION],
                [f"CP-SAT proved the cross-batch constraints infeasible at stage {stage}"],
            )
        raise SolverTimeout(stage, elapsed)

    # Phase A: cheap instantiation of the complete hint (feasible fallback).
    fast = _make_solver(seed, fast_det, fast_wall,
                        stop_at_first=True, fast_presolve=True)
    t0 = time.perf_counter()
    fast_status = fast.solve(model.model)
    fast_elapsed = time.perf_counter() - t0
    if fast_status == cp_model.INFEASIBLE:
        raise CrossBatchPlanInvalid(
            [VIOLATION_XCOMP_NO_SOLUTION],
            [f"CP-SAT proved the cross-batch constraints infeasible at stage {stage}"],
        )
    fallback = fast if fast_status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None

    # Phase B: full single-worker search within the deterministic budget.
    # Symmetry detection is re-enabled here (unlike the arrangement planner,
    # whose highly symmetric balance model needs it off to instantiate the
    # hint): the comparison labels make most symmetries asymmetric, and the
    # coverage stage benefits strongly from detection.
    solver = _make_solver(seed, det_limit, wall_limit)
    solver.parameters.symmetry_level = 2
    solver.parameters.cp_model_probing_level = 1
    t0 = time.perf_counter()
    status = solver.solve(model.model)
    elapsed = time.perf_counter() - t0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver, (
            stage, int(solver.objective_value), solver.status_name(status),
            status == cp_model.OPTIMAL, elapsed,
        )
    if status == cp_model.INFEASIBLE:
        raise CrossBatchPlanInvalid(
            [VIOLATION_XCOMP_NO_SOLUTION],
            [f"CP-SAT proved the cross-batch constraints infeasible at stage {stage}"],
        )
    if fallback is not None:
        return fallback, (
            stage, int(fallback.objective_value), "FEASIBLE_FALLBACK", False,
            fast_elapsed + elapsed,
        )
    raise SolverTimeout(stage, elapsed)


# ------------------------------------------------------------------ planning


def plan_cross_batch(
    req: CrossBatchPlanRequest,
    certs: dict[str, dict],
    revisions: dict[str, dict],
) -> CrossBatchLayout:
    """Validate, build and lexicographically optimise the comparison plate.

    Raises ``InfeasibleArrangement`` (plate geometry/capacity/blocked wells),
    ``CrossBatchPlanInvalid`` (content problems against the snapshots) or
    ``SolverTimeout`` (no solution within the configured budget).
    """
    rows, cols = _resolve_dimensions(req.plate)

    blocked: set[tuple[int, int]] = set()
    geom_categories: list[str] = []
    geom_messages: list[str] = []
    for raw in req.blocked_wells:
        try:
            blocked.add(parse_well(raw, rows, cols))
        except ValueError as exc:
            if VIOLATION_WELL_REFERENCE_INVALID not in geom_categories:
                geom_categories.append(VIOLATION_WELL_REFERENCE_INVALID)
            geom_messages.append(str(exc))
    if geom_categories:
        raise InfeasibleArrangement(geom_categories, geom_messages)

    items, _expected = build_items(req, certs, revisions)

    n_items = len(items)
    available = [
        (r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked
    ]
    if n_items > len(available):
        raise InfeasibleArrangement(
            [VIOLATION_CAPACITY_EXCEEDED],
            [
                f"{n_items} comparison samples but only {len(available)} available "
                f"wells ({rows * cols} total, {len(blocked)} blocked)"
            ],
        )

    batches = sorted({it.batch_id for it in items})
    groups = sorted({it.group for it in items})
    comparisons = sorted({it.comparison_id for it in items})
    seed31 = int(req.seed) % (2**31)
    stages: list[tuple[str, int, str, bool, float]] = []

    # Stage 1: maximise same-comparison orthogonal adjacency edges.
    m1 = _XCompModel(rows, cols, blocked, items, batches, groups, comparisons)
    m1.add_hint(_greedy_assignment(rows, cols, blocked, items))
    m1.maximize_adjacency()
    solver1, info1 = _solve_stage("1_comparison_adjacency", m1, seed31)
    best_adj = int(solver1.objective_value)
    stages.append(info1)

    # Stage 2: maximise rows+columns covered per batch / per group.
    m2 = _XCompModel(rows, cols, blocked, items, batches, groups, comparisons)
    m2.pin_adjacency(best_adj)
    m2.maximize_coverage()
    m2.add_hint(m1.assignment_from_solver(solver1))
    solver2, info2 = _solve_stage("2_coverage", m2, seed31)
    best_cov = int(solver2.objective_value)
    stages.append(info2)

    # Stage 3: deterministic seed tie-break.
    m3 = _XCompModel(rows, cols, blocked, items, batches, groups, comparisons)
    m3.pin_adjacency(best_adj)
    m3.pin_coverage(best_cov)
    m3.add_hint(m2.assignment_from_solver(solver2))
    solver3, info3 = _solve_stage("3_tiebreak", m3, seed31, first_solution=True)
    stages.append(info3)

    placement = m3.assignment_from_solver(solver3)
    return _extract(rows, cols, blocked, items, batches, groups, comparisons,
                    m3, placement, stages)


def _extract(rows, cols, blocked, items, batches, groups, comparisons,
             model: _XCompModel, placement: dict[int, tuple[int, int]],
             stages) -> CrossBatchLayout:
    item_at = {pos: items[i] for i, pos in placement.items()}

    # Same-comparison orthogonal adjacency, recounted authoritatively.
    pairs: list[dict] = []
    edge_count = 0
    well_set = set(model.wells)
    for r, c in model.wells:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) not in well_set:
                continue
            a, b = item_at.get((r, c)), item_at.get((nr, nc))
            if a is not None and b is not None and a.comparison_id == b.comparison_id:
                edge_count += 1
                pairs.append(
                    {
                        "comparison_id": a.comparison_id,
                        "wells": [well_name(r, c), well_name(nr, nc)],
                        "members": [
                            {"batch_id": a.batch_id, "sample_id": a.sample_id},
                            {"batch_id": b.batch_id, "sample_id": b.sample_id},
                        ],
                    }
                )
    pairs.sort(key=lambda p: (p["comparison_id"], p["wells"]))

    def coverage_of(key: str, universe: list[str]):
        out = {}
        total = 0
        for name in universe:
            rowset = {
                placement[it.index][0]
                for it in items
                if getattr(it, key) == name
            }
            colset = {
                placement[it.index][1]
                for it in items
                if getattr(it, key) == name
            }
            entry = {
                "covered_rows": [well_name(r, 0)[0] for r in sorted(rowset)],
                "covered_columns": [c + 1 for c in sorted(colset)],
                "rows_covered": len(rowset),
                "columns_covered": len(colset),
                "total_covered": len(rowset) + len(colset),
            }
            out[name] = entry
            total += entry["total_covered"]
        return out, total

    batch_cov, batch_total = coverage_of("batch_id", batches)
    group_cov, group_total = coverage_of("group", groups)

    available = [
        well_name(r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in blocked
    ]
    return CrossBatchLayout(
        rows=rows,
        columns=cols,
        available_wells=available,
        items=sorted(items, key=lambda it: it.index),
        placement=placement,
        batches=batches,
        groups=groups,
        comparison_ids=comparisons,
        adjacent_pairs=pairs,
        adjacent_edge_count=edge_count,
        total_orthogonal_edges=len(model.edges),
        batch_coverage=batch_cov,
        group_coverage=group_cov,
        coverage_total=batch_total + group_total,
        stages=stages,
    )


# ---------------------------------------------------------------- endpoint


def _plan_id(request_key: str) -> str:
    return "xplan-" + _sha256(
        f"cross-batch-plan\0{request_key}".encode("utf-8")
    )[:24]


def _plan_body(*, req: CrossBatchPlanRequest, certs: dict[str, dict],
               revisions: dict[str, dict], layout: CrossBatchLayout,
               plan_id: str) -> dict:
    row_major = sorted(
        layout.items,
        key=lambda it: layout.placement[it.index],
    )

    batch_sources = []
    for ref in req.batches:
        bid = ref.batch_id
        rev = revisions[bid]
        batch_sources.append(
            {
                "batch_id": bid,
                "certificate_id": certs[bid]["certificate_id"],
                "status": STATUS_ACTIVE,
                "expected_revision": ref.expected_revision,
                "revision": rev["revision"],
                "digest": rev["digest"],
            }
        )

    comparisons_out = []
    for comp in req.comparisons:
        members_out = []
        for m in comp.members:
            it = next(
                it
                for it in layout.items
                if it.comparison_id == comp.comparison_id and it.batch_id == m.batch_id
            )
            r, c = layout.placement[it.index]
            members_out.append(
                {
                    "batch_id": it.batch_id,
                    "sample_id": it.sample_id,
                    "group": it.group,
                    "well": well_name(r, c),
                    "revision": it.source_revision,
                    "digest": it.source_digest,
                }
            )
        comparisons_out.append(
            {"comparison_id": comp.comparison_id, "members": members_out}
        )

    samples_out = []
    for it in row_major:
        r, c = layout.placement[it.index]
        samples_out.append(
            {
                "batch_id": it.batch_id,
                "sample_id": it.sample_id,
                "qualified_sample_id": f"{it.batch_id}:{it.sample_id}",
                "comparison_id": it.comparison_id,
                "group": it.group,
                "well": well_name(r, c),
                "revision": it.source_revision,
                "digest": it.source_digest,
            }
        )

    blocked_canon = sorted(
        {well_name(*parse_well(w, layout.rows, layout.columns)) for w in req.blocked_wells}
    )

    stages_out = [
        {
            "stage": st[0],
            "objective_value": st[1],
            "status": st[2],
            "proven_optimal": st[3],
            "wall_time_seconds": round(st[4], 4),
        }
        for st in layout.stages
    ]
    opt_stages = [st for st in layout.stages if st[0] != "3_tiebreak"]
    overall_status = "OPTIMAL" if all(st[3] for st in opt_stages) else "FEASIBLE"

    return {
        "plan_id": plan_id,
        "status": STATUS_ACTIVE,
        "request_key": req.request_key,
        "seed": req.seed,
        "plate": {
            "type": req.plate.type,
            "rows": layout.rows,
            "columns": layout.columns,
            "blocked_wells": blocked_canon,
            "available_wells": layout.available_wells,
        },
        "batches": batch_sources,
        "comparisons": comparisons_out,
        "layout": {s["qualified_sample_id"]: s["well"] for s in samples_out},
        "samples": samples_out,
        "wells": [s["well"] for s in samples_out],
        "metrics": {
            "level1_comparison_adjacency": {
                "same_comparison_adjacent_edges_count": layout.adjacent_edge_count,
                "same_comparison_adjacent_edges": layout.adjacent_pairs,
                "total_orthogonal_edges": layout.total_orthogonal_edges,
            },
            "level2_coverage": {
                "covered_rows_columns_total": layout.coverage_total,
                "by_batch": layout.batch_coverage,
                "by_group": layout.group_coverage,
            },
        },
        "solver": {"status": overall_status, "stages": stages_out},
    }


def create_cross_batch_plan(req: CrossBatchPlanRequest) -> tuple[int, dict]:
    """Validate all certificates in one transaction and persist the plan."""
    store = get_store()
    content_hash = _content_hash(_ENDPOINT_XCOMP, req)
    with store.transaction() as conn:
        # State is checked BEFORE the idempotency lookup on purpose (same
        # rationale as the re-test plan): once any source certificate advances
        # or is revoked, a request naming the old revision must be refused
        # rather than replaying a plan confirmed against stale evidence.
        certs: dict[str, dict] = {}
        state_error: tuple[int, dict] | None = None
        for ref in req.batches:
            cert = _get_certificate(conn, ref.batch_id)
            if cert is None:
                state_error = 404, {
                    "detail": "no certificate exists for the batch",
                    "error": CERTIFICATE_NOT_FOUND,
                    "batch_id": ref.batch_id,
                }
                break
            if cert["status"] == STATUS_REVOKED:
                state_error = 409, {
                    "detail": "the certificate is REVOKED; a revoked certificate "
                    "can never be restored and cannot feed a cross-batch plan",
                    "error": CERTIFICATE_REVOKED,
                    "certificate_id": cert["certificate_id"],
                    "batch_id": ref.batch_id,
                    "status": STATUS_REVOKED,
                    "revoked_reason": cert["revoked_reason"],
                }
                break
            if ref.expected_revision != cert["revision"]:
                state_error = 409, {
                    "detail": "expected_revision does not match the certificate's "
                    "current revision; cross-batch plans are confirmed only against "
                    "the current revision",
                    "error": CERTIFICATE_REVISION_MISMATCH,
                    "certificate_id": cert["certificate_id"],
                    "batch_id": ref.batch_id,
                    "current_revision": cert["revision"],
                    "expected_revision": ref.expected_revision,
                }
                break
            certs[ref.batch_id] = cert
        if state_error is not None:
            return state_error

        replay = _idem_replay(conn, req.request_key, content_hash)
        if replay is not None:
            return replay

        # Load every batch's pinned current-revision snapshot.
        revisions: dict[str, dict] = {}
        for bid, cert in certs.items():
            row = conn.execute(
                "SELECT digest, snapshot FROM certificate_revisions "
                "WHERE batch_id = ? AND revision = ?",
                (bid, cert["revision"]),
            ).fetchone()
            if row is None:  # pragma: no cover - invariant guard
                raise RuntimeError(
                    f"revision {cert['revision']} snapshot missing for batch {bid!r}"
                )
            revisions[bid] = {
                "revision": cert["revision"],
                "digest": row["digest"],
                "snapshot": json.loads(row["snapshot"]),
            }

        try:
            layout = plan_cross_batch(req, certs, revisions)
        except CrossBatchPlanInvalid as exc:
            body = cross_batch_invalid_body(exc)
            _idem_record(conn, req.request_key, _ENDPOINT_XCOMP,
                         content_hash, 422, body)
            return 422, body
        except InfeasibleArrangement as exc:
            body = {
                "detail": "No cross-batch comparison plate plan can be produced; "
                "no plan was saved.",
                "infeasibility_categories": exc.categories,
                "messages": exc.messages,
            }
            _idem_record(conn, req.request_key, _ENDPOINT_XCOMP,
                         content_hash, 422, body)
            return 422, body

        plan_id = _plan_id(req.request_key)
        body = _plan_body(
            req=req, certs=certs, revisions=revisions,
            layout=layout, plan_id=plan_id,
        )
        now = _utcnow()
        conn.execute(
            "INSERT INTO cross_batch_plans (plan_id, batch_ids, revisions, seed, "
            "plan_body, created_at) VALUES (?,?,?,?,?,?)",
            (
                plan_id,
                json.dumps(layout.batches, ensure_ascii=False),
                json.dumps(
                    {bid: revisions[bid]["revision"] for bid in layout.batches},
                    ensure_ascii=False,
                ),
                req.seed,
                json.dumps(body, ensure_ascii=False),
                now,
            ),
        )
        _idem_record(conn, req.request_key, _ENDPOINT_XCOMP, content_hash, 201, body)
        return 201, body
