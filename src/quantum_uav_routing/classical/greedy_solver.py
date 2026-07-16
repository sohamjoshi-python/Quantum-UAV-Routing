import time
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..io.save_results import save_metrics_to_csv

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