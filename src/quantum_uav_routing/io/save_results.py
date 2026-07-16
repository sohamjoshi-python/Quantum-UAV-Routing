from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

CSV_COLUMNS = [
    "city", "run_type", "nodes", "num_vehicles", "num_requests", "seed", "trial",
    "rtv_graph_build_time", "percent_serviced", "avg_waiting_time", "max_waiting_time",
    "avg_detour_factor", "max_detour_factor", "vmt", "total_run_time", "solve_time",
    "qubo_build_time", "time_min_cost_prep", "time_qubo_gen", "time_compress",
    "time_quantum_solve", "time_decode", "time_vehicle_assignment", "time_metrics_calc",
    "time_struct_stats", "time_total_quantum_block", "ilp_num_vars", "ilp_num_constraints",
    "ilp_num_integer_vars", "ilp_num_nonzero_coeffs", "energy_gap", "argmin_preservation",
    "condition_number", "real_quantum_hardware", "K", "infeasible_windows",
    "arrival_violations", "mean_slack", "min_slack", "max_slack", "p25_slack",
    "median_slack", "p75_slack", "base_qubo_vars", "base_qubo_couplers",
    "base_qubo_graph_density", "base_node_density_avg", "base_node_density_max",
    "base_node_density_min", "base_degree_avg", "base_degree_max", "base_degree_min",
    "base_qubo_max_abs", "base_qubo_min_nonzero_abs", "base_qubo_dynamic_range",
    "comp_qubo_vars", "comp_qubo_couplers", "comp_qubo_graph_density",
    "comp_node_density_avg", "comp_node_density_max", "comp_node_density_min",
    "comp_degree_avg", "comp_degree_max", "comp_degree_min", "comp_qubo_max_abs",
    "comp_qubo_min_nonzero_abs", "comp_qubo_dynamic_range", "qubo_logical_qubits",
    "time_greedy_build_candidates", "greedy_edges_total", "time_greedy_sort",
    "time_greedy_select", "time_greedy_total", "greedy_sort_work", "greedy_assignments",
    "lambda_val", "M_val", "num_reads", "num_sweeps", "cost_alpha",
    # Raw pre-cleanup QUBO infeasibility instrumentation (quantum rows only).
    "raw_selected_trips", "raw_kept_trips", "raw_dropped_trips",
    "raw_violation_rate", "raw_infeasible_instance",
]

# Legacy notebook globals expected by final_trips.
requests = []
vehicles = []
vehicle_lookup = {}
request_lookup = {}
trips = {}
trip_to_vehicle = {}
trip_costs = {}
node2d_to_3d = {}
node_df = None
T_max = 3 * 3600
shortest_path_cached = None
travel_cached = None
build_trips = None
compute_baseline_vehicle_costs = None
generate_requests_and_vehicles = None


def configure_runtime(**kwargs):
    globals().update(kwargs)


def save_metrics_to_csv(filename, metrics_dict):
    # Preserve any columns the metrics dict carries that are not in the known
    # CSV_COLUMNS list, appended after the known columns in a stable order. This
    # prevents silent data loss: previously, reindex(columns=CSV_COLUMNS) DROPPED
    # any metric whose key was not whitelisted (e.g. new instrumentation columns),
    # which is why raw_* infeasibility fields never reached the CSV.
    extra_cols = [k for k in metrics_dict.keys() if k not in CSV_COLUMNS]
    target_columns = list(CSV_COLUMNS) + extra_cols

    df_new = pd.DataFrame([metrics_dict])
    df_new = df_new.reindex(columns=target_columns, fill_value=np.nan)
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        # Union the existing file's columns too, so older rows that lack the new
        # columns (or new rows that lack an old one) all align without dropping.
        union_columns = list(dict.fromkeys(target_columns + list(df_existing.columns)))
        df_existing = df_existing.reindex(columns=union_columns, fill_value=np.nan)
        df_new = df_new.reindex(columns=union_columns, fill_value=np.nan)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(filename, index=False)
    print(f"Data successfully saved to {filename}")


def final_trips(num_requests: int, num_vehicles: int, seed=123):
    global requests, vehicles, vehicle_lookup, request_lookup, trips, trip_to_vehicle, trip_costs, t0, t1

    if travel_cached is not None and hasattr(travel_cached, "cache_clear"):
        travel_cached.cache_clear()

    requests, vehicles = generate_requests_and_vehicles(
        num_requests=int(num_requests),
        num_vehicles=int(num_vehicles),
        node2d_to_3d=node2d_to_3d,
        shortest_path_cached=shortest_path_cached,
        node_df=node_df,
        T_max=T_max,
        seed=seed,
    )

    vehicle_lookup = {v.id: v for v in vehicles}
    request_lookup = {r.id: r for r in requests}

    t0 = time.perf_counter()
    for req_obj in requests:
        req_obj.origin_3d = node2d_to_3d[req_obj.origin]
        req_obj.dest_3d = node2d_to_3d[req_obj.destination]
    all_trips_with_costs = build_trips(requests=requests, vehicles=vehicles, nu=4)
    t1 = time.perf_counter()

    trips = {}
    trip_to_vehicle = {}
    trip_costs = {}

    for t_key, v_costs_dict in all_trips_with_costs.items():
        trips[t_key] = t_key
        trip_to_vehicle[t_key] = list(v_costs_dict.keys())
        for v_id, cost in v_costs_dict.items():
            trip_costs[(t_key, v_id)] = cost

    baseline_vehicle_costs = compute_baseline_vehicle_costs(vehicles)

    return (
        requests,
        vehicles,
        vehicle_lookup,
        request_lookup,
        trips,
        trip_to_vehicle,
        trip_costs,
        baseline_vehicle_costs,
        t0,
        t1,
    )


def summarize_requests(requests):
    trr = np.array([r.trr for r in requests])
    tplr = np.array([r.tplr for r in requests])
    t_star = np.array([r.t_star for r in requests])
    slack = tplr - trr
    return {
        "num_requests": len(requests),
        "infeasible_windows": int(np.sum(slack < 0)),
        "arrival_violations": int(np.sum(t_star < trr)),
        "mean_slack": float(np.mean(slack)),
        "min_slack": float(np.min(slack)),
        "p25_slack": float(np.percentile(slack, 25)),
        "median_slack": float(np.median(slack)),
        "p75_slack": float(np.percentile(slack, 75)),
        "max_slack": float(np.max(slack)),
    }