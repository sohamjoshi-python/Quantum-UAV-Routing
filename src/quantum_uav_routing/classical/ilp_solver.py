from __future__ import annotations

import math
import time

import numpy as np

from ..models import trip_name
from ..io.save_results import save_metrics_to_csv


def greedy_assignment_pnas(
    requests,
    vehicles,
    trips,
    trip_costs,
    trip_to_vehicle,
    nu,
):
    Rok = set()
    Vok = set()
    Sigma_greedy = []

    edges_total = 0
    sort_work = 0.0
    accepted = 0

    t_build = 0.0
    t_sort = 0.0
    t_select = 0.0

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

    buckets = {}
    for t_id, v_costs in costs_by_trip.items():
        k = len(t_id)
        bucket = buckets.setdefault(k, [])
        for v_id, c in v_costs.items():
            bucket.append((c, t_id, v_id))
    tb1 = time.perf_counter()
    t_build = tb1 - tb0

    for k in range(int(nu), 0, -1):
        Sk = buckets.get(k, [])

        m = len(Sk)
        edges_total += m
        if m > 1:
            sort_work += m * math.log2(m)

        ts0 = time.perf_counter()
        Sk.sort(key=lambda x: x[0])
        ts1 = time.perf_counter()
        t_sort += ts1 - ts0

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
        t_select += tl1 - tl0

    ignored_requests = [r_id for r_id in requests if r_id not in Rok]

    greedy_stats = {
        "time_greedy_build_candidates": t_build,
        "time_greedy_sort": t_sort,
        "time_greedy_select": t_select,
        "time_greedy_total": t_build + t_sort + t_select,
        "greedy_edges_total": int(edges_total),
        "greedy_sort_work": float(sort_work),
        "greedy_assignments": int(accepted),
    }

    ilp_stats = {
        "ilp_num_vars": None,
        "ilp_num_constraints": None,
        "ilp_num_integer_vars": None,
        "ilp_num_nonzero_coeffs": None,
    }

    return Sigma_greedy, ignored_requests, greedy_stats, ilp_stats


def classical_greedy_run(
    metadata,
    baseline_vehicle_costs,
    requests,
    vehicles,
    request_lookup,
    vehicle_lookup,
    trips,
    trip_to_vehicle,
    trip_costs,
    node_df,
    rtv_graph_build_time,
    travel_fn,
    csv_filename="results.csv",
    stats=None,
    seed=123,
    trial=1,
):
    if stats is None:
        stats = {}

    run_start = time.perf_counter()

    t2 = time.perf_counter()
    Sigma_greedy, ignored, greedy_stats, ilp_stats = greedy_assignment_pnas(
        requests=[r.id for r in requests],
        vehicles=[v.id for v in vehicles],
        trips=trips,
        trip_costs=trip_costs,
        trip_to_vehicle=trip_to_vehicle,
        nu=max((len(t) for t in trips), default=0),
    )
    t3 = time.perf_counter()

    served_requests = set()
    served_trips = []

    for trip_ids, vid in Sigma_greedy:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        v_obj = vehicle_lookup[vid]
        served_trips.append((trip_objs, v_obj))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = 100.0 * len(served_requests) / len(requests) if requests else 0.0

    detours = []
    waiting = []
    vmt = 0.0

    for trip, v in served_trips:
        total_time = travel_fn(v, trip)
        shortest = sum(r.t_star for r in trip)
        detours.append(total_time / shortest if shortest > 0 else 1.0)
        vmt += float(total_time)

        _, pickups = travel_fn(v, trip, return_timeline=True)
        for r in trip:
            waiting.append(pickups[r] - r.trr)

    avg_detour = float(np.mean(detours)) if detours else 0.0
    max_detour = float(np.max(detours)) if detours else 0.0
    avg_wait = float(np.mean(waiting)) if waiting else 0.0
    max_wait = float(np.max(waiting)) if waiting else 0.0

    run_end = time.perf_counter()

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
    print(f"Vehicle Miles Traveled (VMT): {vmt:.2f}")
    print(f"Total Run Time (solver only): {run_end - run_start:.9f} s")
    print(f"Greedy Solve Time:           {t3 - t2:.9f} s")
    print(f"RTV Graph Build Time:        {rtv_graph_build_time:.9f} s")

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
        "vmt": vmt,
        "total_run_time": run_end - run_start,
        "solve_time": t3 - t2,
        "rtv_graph_build_time": rtv_graph_build_time,
        "seed": seed,
        "trial": trial,
    }

    metrics_to_save.update(stats)
    metrics_to_save.update(ilp_stats)
    metrics_to_save.update(greedy_stats)

    save_metrics_to_csv(csv_filename, metrics_to_save)
    return Sigma_greedy, ignored