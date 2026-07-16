"""
Pairwise-encoding baseline for the request-exclusivity constraint, plus a raw
QUBO-infeasibility measurement helper.

WHY THIS MODULE EXISTS
----------------------
The paper's contribution is that a binary MERGE-TREE encoding replaces the dense
O(k_r^2) pairwise conflict penalties with O(k_r) auxiliary-variable structure.
Reviewer 3 (and, implicitly, Reviewer 1) asked for a direct pairwise-vs-merge-tree
comparison: same problem, same retained trips, same objective, differing ONLY in
how the "at most one selected trip per request" constraint is encoded.

`generate_qubo_pairwise` below is the apples-to-apples counterpart to
`generate_qubo` in quantum_solver.py. It deliberately reuses the SAME trip
retention (cap_per_request scoring), the SAME objective diagonal (-w_t), and the
SAME penalty weight M, so any difference in variable count, coupler count, or
matrix density is attributable purely to the encoding.

DESIGN NOTE (fairness)
----------------------
To keep the comparison honest, this file does NOT re-implement the trip-scoring
and capping logic; that would risk the two encodings operating on different trip
sets. Instead it imports and mirrors the exact retention procedure. If
quantum_solver.generate_qubo changes its retention rule, update the shared helper
`_retain_trips` here to match, or the comparison stops being apples-to-apples.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Shared trip-retention — mirrors quantum_solver.generate_qubo steps 0–1 EXACTLY
# ---------------------------------------------------------------------------
def _retain_trips(
    trips,
    trip_costs,
    ignore_cost,
    cap_per_request,
    cap_total_trips,
    score_mode="benefit",
):
    """Reproduce the trip retention used by generate_qubo so both encodings act
    on an identical set of trip variables. Returns (trip_vars, req_to_trip_idxs,
    idx) where idx maps trip-key -> column index 0..T-1."""
    req_to_trips = defaultdict(list)
    for tkey in trips.keys():
        for r in tkey:
            req_to_trips[r].append(tkey)

    trip_score = {}
    for tkey in trips.keys():
        c = float(trip_costs[tkey])
        benefit = float(ignore_cost) * len(tkey) - c
        trip_score[tkey] = benefit if score_mode == "benefit" else -c

    kept = set()
    if cap_per_request is not None and cap_per_request > 0:
        for r, tlist in req_to_trips.items():
            t_sorted = sorted(
                tlist, key=lambda tk: trip_score.get(tk, -np.inf), reverse=True
            )
            kept.update(t_sorted[:cap_per_request])
    else:
        kept = set(trips.keys())

    if cap_total_trips is not None and len(kept) > cap_total_trips:
        kept = set(
            sorted(
                list(kept),
                key=lambda tk: trip_score.get(tk, -np.inf),
                reverse=True,
            )[:cap_total_trips]
        )

    trip_vars = list(kept)
    idx = {t: i for i, t in enumerate(trip_vars)}
    req_to_trip_idxs = defaultdict(list)
    for t in trip_vars:
        ti = idx[t]
        for r in t:
            req_to_trip_idxs[r].append(ti)

    return trip_vars, req_to_trip_idxs, idx


# ---------------------------------------------------------------------------
# PAIRWISE ENCODING — the baseline the paper compares the merge tree against
# ---------------------------------------------------------------------------
def generate_qubo_pairwise(
    trips,
    trip_costs,
    ignore_cost=10000.0,
    M=20000.0,
    return_numpy=False,
    seed=123,               # accepted for signature parity; pairwise is deterministic
    cap_per_request=30,
    cap_total_trips=None,
    score_mode="benefit",
):
    """
    Standard pairwise (clique) penalty encoding of request exclusivity.

    Identical to generate_qubo EXCEPT the constraint. For each request r with
    retained incident trips S_r, "at most one selected" is enforced by adding a
    penalty M * x_a * x_b for EVERY unordered pair (a, b) in S_r. This is the
    dense O(k_r^2) construction the merge tree is designed to avoid, and it
    introduces NO auxiliary variables.

    Objective (identical to merge-tree version):
        minimize sum_t (-w_t) x_t,   w_t = ignore_cost*|t| - trip_costs[t]

    Returns (Qdict, all_vars) when return_numpy=False, matching generate_qubo's
    contract. all_vars has length T (no aux vars), so decoding works with the
    same downstream code.
    """
    trip_vars, req_to_trip_idxs, idx = _retain_trips(
        trips, trip_costs, ignore_cost, cap_per_request, cap_total_trips, score_mode
    )
    T = len(trip_vars)
    all_vars = list(trip_vars)  # NO auxiliary variables in the pairwise encoding

    Qdict = defaultdict(float)

    def addQ(i, j, v):
        Qdict[(i, j)] += float(v)
        if i != j:
            Qdict[(j, i)] += float(v)

    # Objective on the diagonal — same as generate_qubo step 4.
    for t in trip_vars:
        i = idx[t]
        w_t = float(ignore_cost) * len(t) - float(trip_costs[t])
        addQ(i, i, -w_t)

    # Pairwise exclusivity: M * x_a * x_b for each pair sharing request r.
    # A pair may be shared by more than one request; that naturally accumulates
    # M for each shared request, which is the correct semantics (still one
    # coupler edge, larger weight) — mirroring how the ILP would penalize it.
    for r, tlist in req_to_trip_idxs.items():
        k = len(tlist)
        if k <= 1:
            continue
        for a_pos in range(k):
            a = tlist[a_pos]
            for b_pos in range(a_pos + 1, k):
                b = tlist[b_pos]
                addQ(a, b, M)

    if not return_numpy:
        return dict(Qdict), all_vars

    n = T
    Q = np.zeros((n, n), dtype=float)
    for (i, j), v in Qdict.items():
        Q[i, j] = v
    return Q, all_vars


# ---------------------------------------------------------------------------
# RAW INFEASIBILITY MEASUREMENT
# ---------------------------------------------------------------------------
def raw_infeasibility_from_selection(selected_trip_keys):
    """
    Measure how often the RAW QUBO output violates request exclusivity, BEFORE
    the greedy conflict-cleanup runs.

    In quantum_mwis_run, `selected_trip_keys` is the list of trips the solver set
    to 1 (frozensets of request ids), taken straight from the bitstring. The
    greedy cleanup then keeps only a conflict-free subset. This function quantifies
    what the cleanup had to remove.

    Returns a dict:
      - n_selected:        number of raw selected trips
      - n_conflicting_trips: how many selected trips shared >=1 request with an
                             earlier-kept selected trip (i.e. were dropped by the
                             same greedy rule the pipeline uses)
      - trip_violation_rate: n_conflicting_trips / n_selected  (0 if none selected)
      - n_over_served_requests: number of requests covered by >1 selected trip
      - any_violation:     bool, True if the raw output violated exclusivity at all

    trip_violation_rate is the quantity the penalty weight M is meant to control:
    a well-calibrated M drives the raw QUBO output toward already-feasible
    selections, lowering this rate. Post-cleanup `percent_serviced` is a far less
    sensitive signal for M because the greedy + Hungarian stages repair most
    violations regardless of M.
    """
    n_selected = len(selected_trip_keys)
    if n_selected == 0:
        return {
            "n_selected": 0,
            "n_conflicting_trips": 0,
            "trip_violation_rate": 0.0,
            "n_over_served_requests": 0,
            "any_violation": False,
        }

    # How many requests are covered more than once by the raw selection.
    request_coverage = defaultdict(int)
    for tk in selected_trip_keys:
        for r in tk:
            request_coverage[r] += 1
    n_over_served = sum(1 for c in request_coverage.values() if c > 1)

    # Replay the SAME greedy rule the pipeline uses, and count what it drops.
    covered = set()
    n_conflicting = 0
    for tk in selected_trip_keys:
        if tk & covered:
            n_conflicting += 1
        else:
            covered |= tk

    return {
        "n_selected": n_selected,
        "n_conflicting_trips": n_conflicting,
        "trip_violation_rate": n_conflicting / n_selected,
        "n_over_served_requests": n_over_served,
        "any_violation": (n_conflicting > 0),
    }


# ---------------------------------------------------------------------------
# CONVENIENCE: build both encodings on one instance and compare structure
# ---------------------------------------------------------------------------
def compare_encodings(
    trips,
    trip_costs,
    ignore_cost=10000.0,
    M=20000.0,
    seed=123,
    cap_per_request=30,
    cap_total_trips=None,
):
    """
    Build the merge-tree QUBO (via quantum_solver.generate_qubo) and the pairwise
    QUBO on the SAME instance, and return their structural stats side by side.

    This is what produces the reviewer-requested comparison table:
    variables, couplers, density for each encoding.

    Import is done inside the function to avoid a circular import at module load.
    """
    from quantum_uav_routing.quantum.quantum_solver import (
        generate_qubo,
        qubo_stats_from_dict,
    )

    Q_tree, vars_tree = generate_qubo(
        trips,
        trip_costs=trip_costs,
        ignore_cost=ignore_cost,
        M=M,
        return_numpy=False,
        seed=seed,
        cap_per_request=cap_per_request,
        cap_total_trips=cap_total_trips,
    )
    Q_pair, vars_pair = generate_qubo_pairwise(
        trips,
        trip_costs=trip_costs,
        ignore_cost=ignore_cost,
        M=M,
        return_numpy=False,
        seed=seed,
        cap_per_request=cap_per_request,
        cap_total_trips=cap_total_trips,
    )

    stats_tree = qubo_stats_from_dict(Q_tree)
    stats_pair = qubo_stats_from_dict(Q_pair)

    return {
        "merge_tree": {
            "vars": stats_tree["qubo_vars"],
            "couplers": stats_tree["qubo_couplers"],
            "graph_density": stats_tree["qubo_graph_density"],
            "degree_max": stats_tree["degree_max"],
        },
        "pairwise": {
            "vars": stats_pair["qubo_vars"],
            "couplers": stats_pair["qubo_couplers"],
            "graph_density": stats_pair["qubo_graph_density"],
            "degree_max": stats_pair["degree_max"],
        },
        "coupler_ratio_pairwise_over_tree": (
            stats_pair["qubo_couplers"] / stats_tree["qubo_couplers"]
            if stats_tree["qubo_couplers"] > 0
            else float("inf")
        ),
    }