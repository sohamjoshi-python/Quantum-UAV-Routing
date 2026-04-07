
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import pulp
from scipy.optimize import linear_sum_assignment

from ..rtv.trip_builder import (
    travel,
    trip_name,
    request_lookup,
    vehicle_lookup,
    requests,
    vehicles,
    trips,
    trip_costs,
    trip_to_vehicle,
    node_df,
)
from ..io.save_results import save_metrics_to_csv


def configure_runtime(**kwargs):
    from ..rtv import trip_builder
    trip_builder.configure_runtime(**kwargs)

def ilp_model_stats(model):
    vars_ = model.variables()
    num_vars = len(vars_)
    num_constraints = len(model.constraints)

    # Type counts (PuLP categories: "Binary", "Integer", "Continuous")
    num_bin = sum(1 for v in vars_ if v.cat == "Binary")
    num_int = sum(1 for v in vars_ if v.cat == "Integer")
    num_cont = sum(1 for v in vars_ if v.cat == "Continuous")

    # Nonzeros in constraint matrix (approx): count variable appearances in constraints
    # This uses PuLP's internal constraint representation.
    nnz = 0
    for cname, c in model.constraints.items():
        # c is an LpConstraint, c.items() gives (LpVariable, coeff)
        nnz += len(c.items())

    return {
        "ilp_num_vars": num_vars,
        "ilp_num_constraints": num_constraints,
        "ilp_num_binary_vars": num_bin,
        "ilp_num_integer_vars": num_int,
        "ilp_num_continuous_vars": num_cont,
        "ilp_num_nonzero_coeffs": nnz,
    }

import pulp

def optimal_assignment(
    requests,
    vehicles,
    trips,
    trip_costs,
    trip_to_vehicle,
    baseline_vehicle_costs,
    ignore_cost=10000
):
    """
    PNAS-style ILP for trip-vehicle assignment with a GREEDY WARM START,
    not a fixed greedy solution.

    Returns
    -------
    Sigma_opt : list of (t_id, v_id)
    ignored_requests : list of request IDs
    stats : dict
    """

    # -------------------------------------------------
    # 1. Create ILP model
    # -------------------------------------------------
    model = pulp.LpProblem("OptimalAssignment", pulp.LpMinimize)

    # -------------------------------------------------
    # 2. Decision variables
    # ε_(i,j) and χ_k
    # -------------------------------------------------
    epsilon = {}
    for t_id in trips:
        for v_id in trip_to_vehicle[t_id]:
            if (t_id, v_id) in trip_costs:
                epsilon[(t_id, v_id)] = pulp.LpVariable(
                    f"eps_{t_id}_{v_id}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpBinary
                )

    chi = {
        r_id: pulp.LpVariable(
            f"chi_{r_id}",
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary
        )
        for r_id in requests
    }

    # -------------------------------------------------
    # 3. Objective
    # incremental delay + ignore penalty
    # -------------------------------------------------
    model += (
        pulp.lpSum(
            (trip_costs[(t_id, v_id)] - baseline_vehicle_costs[v_id])
            * epsilon[(t_id, v_id)]
            for (t_id, v_id) in epsilon
        )
        +
        pulp.lpSum(ignore_cost * chi[r_id] for r_id in requests)
    )

    # -------------------------------------------------
    # 4. Constraints
    # -------------------------------------------------

    # (8) Each vehicle assigned at most one trip
    for v_id in vehicles:
        model += (
            pulp.lpSum(
                epsilon[(t_id, v_id)]
                for (t_id, vv) in epsilon
                if vv == v_id
            ) <= 1
        )

    # (9) Each request served exactly once or ignored
    for r_id in requests:
        model += (
            pulp.lpSum(
                epsilon[(t_id, v_id)]
                for (t_id, v_id) in epsilon
                if r_id in trips[t_id]
            ) + chi[r_id] == 1
        )

    # -------------------------------------------------
    # 5. Greedy warm start (Algorithm 2 idea)
    # IMPORTANT: warm start only, DO NOT fix variables
    # -------------------------------------------------

    # Initialize all variables to zero
    for var in epsilon.values():
        var.setInitialValue(0)

    for var in chi.values():
        var.setInitialValue(0)

    sorted_eps = sorted(
        epsilon.keys(),
        key=lambda x: (
            -len(trips[x[0]]),   # trip size descending
            trip_costs[x]        # cost ascending
        )
    )

    assigned_requests = set()
    assigned_vehicles = set()

    for (t_id, v_id) in sorted_eps:
        if v_id in assigned_vehicles:
            continue
        if any(r in assigned_requests for r in trips[t_id]):
            continue

        # Warm start only
        epsilon[(t_id, v_id)].setInitialValue(1)

        assigned_vehicles.add(v_id)
        assigned_requests |= trips[t_id]

    for r_id in requests:
        if r_id not in assigned_requests:
            chi[r_id].setInitialValue(1)

    # -------------------------------------------------
    # 6. Solve
    # -------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True)
    model.solve(solver)

    # -------------------------------------------------
    # 7. Extract solution
    # -------------------------------------------------
    Sigma_opt = [
        (t_id, v_id)
        for (t_id, v_id), var in epsilon.items()
        if var.value() is not None and var.value() > 0.5
    ]

    ignored_requests = [
        r_id
        for r_id, var in chi.items()
        if var.value() is not None and var.value() > 0.5
    ]

    stats = ilp_model_stats(model)

    return Sigma_opt, ignored_requests, stats

import time

import math

import numpy as np

from scipy.optimize import linear_sum_assignment

def greedy_assignment_pnas_instrumented(trips, trip_costs, trip_to_vehicle, nu):
    """
    PNAS Algorithm 2 greedy assignment + instrumentation.

    Returns:
      Sigma: list[(trip_id, vehicle_id)]
      ignored: list[request_id]
      stats: dict with timing + growth counters
    """
    Rok, Vok = set(), set()
    Sigma = []

    # ---- counters ----
    edges_total = 0
    sort_work = 0.0
    accepted = 0

    # ---- timing ----
    t_build = 0.0
    t_sort = 0.0
    t_select = 0.0

    for k in range(int(nu), 0, -1):
        # (A) Build candidate list S_k
        tb0 = time.perf_counter()
        Sk = []
        for t_id in trips.keys():
            if len(t_id) != k:
                continue
            for v_id in trip_to_vehicle.get(t_id, []):
                c = trip_costs.get((t_id, v_id), None)
                if c is not None:
                    Sk.append((float(c), t_id, v_id))
        tb1 = time.perf_counter()
        t_build += (tb1 - tb0)

        m = len(Sk)
        edges_total += m
        if m > 1:
            sort_work += m * math.log2(m)

        # (B) Sort by increasing cost
        ts0 = time.perf_counter()
        Sk.sort(key=lambda x: x[0])
        ts1 = time.perf_counter()
        t_sort += (ts1 - ts0)

        # (C) Select feasible assignments
        tl0 = time.perf_counter()
        for _, t_id, v_id in Sk:
            if v_id in Vok:
                continue
            if any(r in Rok for r in t_id):
                continue
            Sigma.append((t_id, v_id))
            Vok.add(v_id)
            Rok |= set(t_id)
            accepted += 1
        tl1 = time.perf_counter()
        t_select += (tl1 - tl0)

    all_req_ids = [r.id for r in requests]
    ignored = [rid for rid in all_req_ids if rid not in Rok]

    stats = {
        "time_greedy_build_candidates": t_build,
        "time_greedy_sort": t_sort,
        "time_greedy_select": t_select,
        "time_greedy_total": (t_build + t_sort + t_select),
        "greedy_edges_total": int(edges_total),
        "greedy_sort_work": float(sort_work),
        "greedy_assignments": int(accepted),
    }
    return Sigma, ignored, stats

def compute_baseline_vehicle_costs(vehicles):
    costs = {}
    for v in vehicles:
        cost = travel(v, set())
        costs[v.id] = cost if cost is not None else 0
    return costs

import time

import math

def greedy_assignment_pnas(
    requests,
    vehicles,
    trips,
    trip_costs,
    trip_to_vehicle,
    nu
):
    """
    PNAS Algorithm 2 greedy assignment.

    Returns
    -------
    Sigma_greedy : list of (trip_id, vehicle_id)
    ignored_requests : list of request_ids
    greedy_stats : dict
    ilp_stats : dict   # schema-compatible placeholder
    """

    Rok = set()
    Vok = set()
    Sigma_greedy = []

    edges_total = 0
    sort_work   = 0.0
    accepted    = 0

    t_build  = 0.0
    t_sort   = 0.0
    t_select = 0.0

    # (A) Pre-index trip_costs by t_id to avoid repeated (frozenset, v_id) lookups.
    # trip_costs is keyed by (frozenset, v_id), which at large n suffers hash
    # collisions as the dict grows to ~1M entries. Re-indexing as
    # costs_by_trip[t_id][v_id] = cost reduces each lookup to two small-dict
    # accesses and keeps per-edge cost O(1) regardless of total dict size.
    tb0 = time.perf_counter()
    costs_by_trip = {}
    for t_id in trips:
        v_costs = {}
        for v_id in trip_to_vehicle.get(t_id, []):
            c = trip_costs.get((t_id, v_id), None)
            if c is not None:
                v_costs[v_id] = float(c)
        if v_costs:
            costs_by_trip[t_id] = v_costs

    # Build buckets by trip size in the same pass — O(T) total, one scan only.
    buckets = {}
    for t_id, v_costs in costs_by_trip.items():
        k = len(t_id)
        bucket = buckets.setdefault(k, [])
        for v_id, c in v_costs.items():
            bucket.append((c, t_id, v_id))
    tb1 = time.perf_counter()
    t_build = tb1 - tb0

    # (B) Process buckets from largest trip size down to 1
    for k in range(int(nu), 0, -1):
        Sk = buckets.get(k, [])

        m = len(Sk)
        edges_total += m
        if m > 1:
            sort_work += m * math.log2(m)

        ts0 = time.perf_counter()
        Sk.sort(key=lambda x: x[0])
        ts1 = time.perf_counter()
        t_sort += (ts1 - ts0)

        tl0 = time.perf_counter()
        for _, t_id, v_id in Sk:
            if v_id in Vok:
                continue
            if any(r in Rok for r in t_id):
                continue
            Sigma_greedy.append((t_id, v_id))
            Vok.add(v_id)
            Rok.update(t_id)
            accepted += 1
        tl1 = time.perf_counter()
        t_select += (tl1 - tl0)

    ignored_requests = [r_id for r_id in requests if r_id not in Rok]

    greedy_stats = {
        "time_greedy_build_candidates": t_build,
        "time_greedy_sort":             t_sort,
        "time_greedy_select":           t_select,
        "time_greedy_total":            (t_build + t_sort + t_select),
        "greedy_edges_total":           int(edges_total),
        "greedy_sort_work":             float(sort_work),
        "greedy_assignments":           int(accepted),
    }

    ilp_stats = {
        "ilp_num_vars":           None,
        "ilp_num_constraints":    None,
        "ilp_num_integer_vars":   None,
        "ilp_num_nonzero_coeffs": None,
    }

    return Sigma_greedy, ignored_requests, greedy_stats, ilp_stats

import time

def classical_greedy_run(
    metadata,
    baseline_vehicle_costs,
    csv_filename="results.csv",
    stats={},
    seed=123,
    trial=1
):
    """
    Classical greedy run with clean timing semantics:

    - total_run_time: end-to-end time inside THIS function (solver pipeline only)
    - solve_time: time spent inside greedy_assignment_pnas() (greedy solve time)
    - rtv_graph_build_time: scenario precompute time from final_trips (t1 - t0), if available
    - scenario_total_time: scenario pipeline time (if provided via stats), optional
    """

    # Per-solver wall time (does NOT include scenario generation unless you call this inside it)
    run_start = time.perf_counter()

    # ----------------------------
    # GREEDY SOLVE
    # ----------------------------
    t2 = time.perf_counter()

    Sigma_greedy, ignored, greedy_stats, ilp_stats = greedy_assignment_pnas(
        requests=[r.id for r in requests],
        vehicles=[v.id for v in vehicles],
        trips=trips,
        trip_costs=trip_costs,
        trip_to_vehicle=trip_to_vehicle,
        nu=max(len(t) for t in trips) if trips else 0
    )

    t3 = time.perf_counter()

    # ----------------------------
    # POST-PROCESSING
    # ----------------------------
    served_requests = set()
    served_trips = []

    for trip_ids, vid in Sigma_greedy:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        v_obj = vehicle_lookup[vid]
        served_trips.append((trip_objs, v_obj))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = 100.0 * len(served_requests) / len(requests) if requests else 0.0

    # ----------------------------
    # METRICS (RAW)
    # ----------------------------
    detours = []
    waiting = []
    VMT = 0.0

    for trip, v in served_trips:
        total_time = travel(v, trip)
        shortest = sum(r.t_star for r in trip)
        detours.append(total_time / shortest if shortest > 0 else 1.0)
        VMT += float(total_time)

        _, pickups = travel(v, trip, return_timeline=True)
        for r in trip:
            waiting.append(pickups[r] - r.trr)

    avg_detour = float(sum(detours) / len(detours)) if detours else 0.0
    max_detour = float(max(detours)) if detours else 0.0
    avg_wait = float(sum(waiting) / len(waiting)) if waiting else 0.0
    max_wait = float(max(waiting)) if waiting else 0.0

    run_end = time.perf_counter()

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print("Number of nodes in the graph:", node_df["node_id"].nunique())
    print("Number of vehicles:", len(vehicles))
    print("Number of requests:", len(requests))
    print(f"% Serviced Requests: {percent_serviced:.2f}%")

    print("Greedy Assignment:")
    for trip, v in served_trips:
        print(" ", trip_name(trip), "→", v)

    print("\nIgnored Requests:", ignored)

    print("\n--- Performance Metrics ---")
    print(f"Average Waiting Time: {avg_wait:.2f} s")
    print(f"Maximum Waiting Time: {max_wait:.2f} s")
    print(f"Average Detour Factor: {avg_detour:.3f}")
    print(f"Maximum Detour Factor: {max_detour:.3f}")
    print(f"Vehicle Miles Traveled (VMT): {VMT:.2f}")

    # IMPORTANT: do not report "t3 - t0" as solver runtime; that's scenario-cumulative.
    print(f"Total Run Time (solver only): {run_end - run_start:.9f} s")
    print(f"Greedy Solve Time:           {t3 - t2:.9f} s")

    if "t0" in globals() and "t1" in globals():
        print(f"RTV Graph Build Time:        {t1 - t0:.2f} s")

    # ----------------------------
    # SAVE METRICS
    # ----------------------------
    metrics_to_save = {
        "city": metadata.get("city", ""),
        "run_type": "ClassicalGreedy",
        "nodes": node_df["node_id"].nunique(),
        "num_vehicles": len(vehicles),
        "num_requests": len(requests),

        "percent_serviced": percent_serviced,
        "avg_waiting_time": avg_wait,
        "max_waiting_time": max_wait,
        "avg_detour_factor": avg_detour,
        "max_detour_factor": max_detour,
        "vmt": VMT,

        # Clean timing semantics:
        "total_run_time": run_end - run_start,   # solver pipeline wall time
        "solve_time": t3 - t2,                   # greedy solver call time

        # Scenario build time (if available from final_trips):
        "rtv_graph_build_time": (t1 - t0) if ("t0" in globals() and "t1" in globals()) else None,

        "seed": seed,
        "trial": trial,
    }

    # Add scenario-level time if you computed it in the outer loop and stuffed into stats
    # (recommended): stats["scenario_total_time"] = scenario_end - scenario_start

    metrics_to_save.update(stats)
    metrics_to_save.update(ilp_stats)
    metrics_to_save.update(greedy_stats)

    save_metrics_to_csv(csv_filename, metrics_to_save)
    return Sigma_greedy, ignored

from collections import defaultdict

import numpy as np

def classical_ilp_run(
    metadata,
    baseline_vehicle_costs,
    csv_filename="results.csv",
    stats={},
    seed=123,
    trial=1
):
    """
    Classical ILP run with clean timing semantics:

    - total_run_time: end-to-end time inside THIS function (solver pipeline only)
    - solve_time: time spent inside optimal_assignment() (ILP solve/model time)
    - rtv_graph_build_time: scenario precompute time from final_trips (t1 - t0), if available
    - scenario_total_time: scenario pipeline time (if provided via stats), optional
    """

    # Per-solver wall time (does NOT include scenario generation unless you call this inside it)
    run_start = time.perf_counter()

    # ----------------------------
    # ILP SOLVE
    # ----------------------------
    t2 = time.perf_counter()

    Sigma_opt, ignored, ilp_stats = optimal_assignment(
        requests=[r.id for r in requests],
        vehicles=[v.id for v in vehicles],
        trips=trips,
        trip_costs=trip_costs,
        trip_to_vehicle=trip_to_vehicle,
        baseline_vehicle_costs=baseline_vehicle_costs,
        ignore_cost=10000
    )

    t3 = time.perf_counter()

    # ----------------------------
    # POST-PROCESSING
    # ----------------------------
    served_requests = set()
    served_trips = []

    for trip_ids, vid in Sigma_opt:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        v_obj = vehicle_lookup[vid]
        served_trips.append((trip_objs, v_obj))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = 100.0 * len(served_requests) / len(requests) if requests else 0.0

    # ----------------------------
    # METRICS (RAW)
    # ----------------------------
    detours = []
    waiting = []
    VMT = 0.0

    for trip, v in served_trips:
        total_time = travel(v, trip)
        shortest = sum(r.t_star for r in trip)
        detours.append(total_time / shortest if shortest > 0 else 1.0)
        VMT += float(total_time)

        _, pickups = travel(v, trip, return_timeline=True)
        for r in trip:
            waiting.append(pickups[r] - r.trr)

    avg_detour = float(sum(detours) / len(detours)) if detours else 0.0
    max_detour = float(max(detours)) if detours else 0.0
    avg_wait = float(sum(waiting) / len(waiting)) if waiting else 0.0
    max_wait = float(max(waiting)) if waiting else 0.0

    run_end = time.perf_counter()

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print("Number of nodes in the graph:", node_df["node_id"].nunique())
    print("Number of vehicles:", len(vehicles))
    print("Number of requests:", len(requests))
    print(f"% Serviced Requests: {percent_serviced:.2f}%")

    print("Optimal Assignment:")
    for trip, v in served_trips:
        print(" ", trip_name(trip), "→", v)

    print("\nIgnored Requests:", ignored)

    print("\n--- Performance Metrics ---")
    print(f"Average Waiting Time: {avg_wait:.2f} s")
    print(f"Maximum Waiting Time: {max_wait:.2f} s")
    print(f"Average Detour Factor: {avg_detour:.3f}")
    print(f"Maximum Detour Factor: {max_detour:.3f}")
    print(f"Vehicle Miles Traveled (VMT): {VMT:.2f}")

    # IMPORTANT: do not report "t3 - t0" as solver runtime; that's scenario-cumulative.
    print(f"Total Run Time (solver only): {run_end - run_start:.9f} s")
    print(f"ILP Solve Time:              {t3 - t2:.9f} s")

    if "t0" in globals() and "t1" in globals():
        print(f"RTV Graph Build Time:        {t1 - t0:.2f} s")

    # ----------------------------
    # SAVE METRICS
    # ----------------------------
    metrics_to_save = {
        "city": metadata.get("city", ""),
        "run_type": "Classical",
        "nodes": node_df["node_id"].nunique(),
        "num_vehicles": len(vehicles),
        "num_requests": len(requests),

        "percent_serviced": percent_serviced,
        "avg_waiting_time": avg_wait,
        "max_waiting_time": max_wait,
        "avg_detour_factor": avg_detour,
        "max_detour_factor": max_detour,
        "vmt": VMT,

        # Clean timing semantics:
        "total_run_time": run_end - run_start,   # solver pipeline wall time
        "solve_time": t3 - t2,                   # ILP solver/model call time

        # Scenario build time (if available from final_trips):
        "rtv_graph_build_time": (t1 - t0) if ("t0" in globals() and "t1" in globals()) else None,

        "seed": seed,
        "trial": trial,
    }

    # Add scenario-level time if you computed it in the outer loop and stuffed into stats
    # (recommended): stats["scenario_total_time"] = scenario_end - scenario_start

    metrics_to_save.update(stats)
    metrics_to_save.update(ilp_stats)

    save_metrics_to_csv(csv_filename, metrics_to_save)
    return Sigma_opt, ignored
