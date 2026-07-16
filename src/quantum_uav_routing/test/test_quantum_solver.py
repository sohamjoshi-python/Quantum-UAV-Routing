"""
Integration tests for the quantum MWIS solver on Phoenix scenarios.

Uses simulated annealing (real=False) for scenarios up to 20 requests,
matching the notebook's smaller quantum benchmark range.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_uav_routing.io import save_results
from quantum_uav_routing.network.load_network import (
    REPO_DIR,
    clone_dataset_repo,
    convert_to_3d,
    create_graph,
    parse_graph,
)
import quantum_uav_routing.network.shortest_path as sp_mod
from quantum_uav_routing.network.shortest_path import build_3d_graph, shortest_path_cached
from quantum_uav_routing.quantum import quantum_solver
from quantum_uav_routing.rtv import trip_builder
from quantum_uav_routing.rtv import build_rtv_graph as build_rtv
from quantum_uav_routing.rtv.data_structure import Request, Vehicle, trip_name

NETWORK_NAME = "32_Phoenix_City"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
NODE_3D_PATH = RAW_DATA_DIR / "node_3d.csv"
LINK_3D_PATH = RAW_DATA_DIR / "link_3d.csv"
MAPPING_PATH = RAW_DATA_DIR / "node2d_to_3d_mapping.csv"

BASE_SEED = 42
# Quantum tests capped at 20 requests (simulated annealing is slower than classical).
SCENARIOS_BY_VEHICLES = {
    5: [5, 10],
    10: [10, 20],
}
SCENARIOS = [
    (num_vehicles, num_requests)
    for num_vehicles, request_counts in SCENARIOS_BY_VEHICLES.items()
    for num_requests in request_counts
]

RESULTS_DIR = PROJECT_ROOT / "results" / "quantum_solver_tests"
QUANTUM_RESULTS_CSV = RESULTS_DIR / "quantum_results.csv"

# Notebook single-run quantum parameters.
LAMBDA_VAL = 5000.0
M_VAL = 25000.0

COMPARISON_COLUMNS = [
    "run_type",
    "city",
    "num_vehicles",
    "num_requests",
    "seed",
    "trial",
    "lambda_val",
    "M_val",
    "real_quantum_hardware",
    "percent_serviced",
    "avg_waiting_time",
    "max_waiting_time",
    "avg_detour_factor",
    "max_detour_factor",
    "vmt",
    "rtv_graph_build_time",
    "total_run_time",
    "solve_time",
    "qubo_build_time",
    "time_quantum_solve",
    "base_qubo_vars",
    "base_qubo_couplers",
]


def _scenario_seed(num_vehicles: int, num_requests: int, city_index: int = 0, trial: int = 0) -> int:
    return BASE_SEED + trial + 100 * num_vehicles + 10_000 * num_requests + 1_000_000 * city_index


def _generate_requests_and_vehicles(
    num_requests,
    num_vehicles,
    node2d_to_3d,
    shortest_path_cached,
    node_df,
    T_max=3 * 3600,
    seed=42,
    min_slack=1200,
    max_slack=4800,
):
    random.seed(seed)
    np.random.seed(seed)

    nodes_2d = list(node2d_to_3d.keys())
    node_coords_map = node_df.set_index("node_id")[["x_coord", "y_coord"]].to_dict("index")

    vehicles = [
        Vehicle(
            random.choice(nodes_2d),
            random.randint(0, T_max // 2),
            [],
            vid + 1,
        )
        for vid in range(num_vehicles)
    ]

    requests = []
    for rid in range(1, num_requests + 1):
        origin, destination = random.sample(nodes_2d, 2)
        origin_coords = node_coords_map.get(origin)
        destination_coords = node_coords_map.get(destination)
        if origin_coords is None or destination_coords is None:
            continue

        release_time = random.randint(0, T_max - max_slack)
        travel_time = shortest_path_cached(
            node2d_to_3d[origin],
            node2d_to_3d[destination],
        )
        if not np.isfinite(travel_time):
            continue

        slack = random.randint(min_slack, max_slack)
        requests.append(
            Request(
                origin,
                destination,
                release_time,
                release_time + slack,
                travel_time,
                rid,
                origin_coords["x_coord"],
                origin_coords["y_coord"],
                destination_coords["x_coord"],
                destination_coords["y_coord"],
            )
        )

    return requests, vehicles


def _compute_baseline_vehicle_costs(vehicles):
    costs = {}
    for vehicle in vehicles:
        cost = trip_builder.travel(vehicle, set())
        costs[vehicle.id] = cost if cost is not None else 0
    return costs


def _load_or_build_3d_network(network_name: str):
    clone_dataset_repo()
    create_graph(network_name)
    _, node_df, _, _ = parse_graph(network_name)

    if not (NODE_3D_PATH.exists() and LINK_3D_PATH.exists() and MAPPING_PATH.exists()):
        convert_to_3d(network_name, str(PROJECT_ROOT / REPO_DIR / network_name))

    node_3d_df = pd.read_csv(NODE_3D_PATH, low_memory=False)
    link_3d_df = pd.read_csv(LINK_3D_PATH, low_memory=False)
    link_3d_df["length"] = link_3d_df["length"].fillna(1e-3).replace(0, 1e-3)

    mapping_df = pd.read_csv(MAPPING_PATH)
    node2d_to_3d = dict(zip(mapping_df["original_node_id"], mapping_df["node_3d_id"]))

    return node_df, node_3d_df, link_3d_df, node2d_to_3d


def _configure_shortest_path_oracle(node_3d_df, link_3d_df):
    graph = build_3d_graph(node_3d_df, link_3d_df, alpha=1.0, beta=0.005)
    node_list = list(graph.nodes())
    sp_mod.NODE_TO_IDX = {node_id: idx for idx, node_id in enumerate(node_list)}

    rows, cols, data = [], [], []
    for source, target, attrs in graph.edges(data=True):
        weight = float(attrs.get("weight", 1.0))
        source_idx = sp_mod.NODE_TO_IDX[source]
        target_idx = sp_mod.NODE_TO_IDX[target]
        rows.extend([source_idx, target_idx])
        cols.extend([target_idx, source_idx])
        data.extend([weight, weight])

    sp_mod.CSR = csr_matrix(
        (data, (rows, cols)),
        shape=(len(node_list), len(node_list)),
        dtype=float,
    )
    sp_mod._dist_from_src.cache_clear()


def _inject_runtime_globals(node_df, node2d_to_3d):
    trip_builder.node2d_to_3d = node2d_to_3d
    build_rtv.node2d_to_3d = node2d_to_3d
    build_rtv.shortest_path_cached = shortest_path_cached

    save_results.configure_runtime(
        node2d_to_3d=node2d_to_3d,
        node_df=node_df,
        shortest_path_cached=shortest_path_cached,
        travel_cached=trip_builder.travel_cached,
        build_trips=build_rtv.build_trips,
        compute_baseline_vehicle_costs=_compute_baseline_vehicle_costs,
        generate_requests_and_vehicles=_generate_requests_and_vehicles,
    )


def _inject_scenario_globals(node_df, scenario_result):
    (
        requests,
        vehicles,
        vehicle_lookup,
        request_lookup,
        trips,
        trip_to_vehicle,
        trip_costs,
        baseline,
        t0,
        t1,
    ) = scenario_result

    trip_builder.vehicle_lookup = vehicle_lookup
    trip_builder.request_lookup = request_lookup
    trip_builder.requests = requests
    trip_builder.vehicles = vehicles

    quantum_solver.requests = requests
    quantum_solver.vehicles = vehicles
    quantum_solver.vehicle_lookup = vehicle_lookup
    quantum_solver.request_lookup = request_lookup
    quantum_solver.trips = trips
    quantum_solver.trip_to_vehicle = trip_to_vehicle
    quantum_solver.trip_costs = trip_costs
    quantum_solver.node_df = node_df
    quantum_solver.travel = trip_builder.travel
    quantum_solver.travel_cached = trip_builder.travel_cached
    quantum_solver.t0 = t0
    quantum_solver.t1 = t1

    return requests, vehicles, baseline


def _validate_assignment(assignment, requests, vehicles):
    served_requests = set()
    used_vehicles = set()

    for trip_ids, vehicle_id in assignment:
        assert vehicle_id not in used_vehicles, "vehicle assigned more than once"
        used_vehicles.add(vehicle_id)
        assert vehicle_id in {v.id for v in vehicles}, "unknown vehicle in assignment"

        for request_id in trip_ids:
            assert request_id not in served_requests, "request served more than once"
            served_requests.add(request_id)
            assert request_id in {r.id for r in requests}, "unknown request in assignment"

    return served_requests


def _prepare_scenario(node_df, num_vehicles, num_requests):
    seed = _scenario_seed(num_vehicles, num_requests)
    scenario = save_results.final_trips(
        num_requests=num_requests,
        num_vehicles=num_vehicles,
        seed=seed,
    )
    requests, vehicles, baseline = _inject_scenario_globals(node_df, scenario)
    stats = save_results.summarize_requests(requests)
    metadata = {"city": NETWORK_NAME, "v": num_vehicles, "r": num_requests}
    return seed, requests, vehicles, baseline, stats, metadata


@pytest.fixture(scope="session")
def results_csv():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if QUANTUM_RESULTS_CSV.exists():
        QUANTUM_RESULTS_CSV.unlink()
    return QUANTUM_RESULTS_CSV


@pytest.fixture(scope="module")
def phoenix_network():
    node_df, node_3d_df, link_3d_df, node2d_to_3d = _load_or_build_3d_network(NETWORK_NAME)
    _configure_shortest_path_oracle(node_3d_df, link_3d_df)
    _inject_runtime_globals(node_df, node2d_to_3d)
    return node_df


@pytest.mark.parametrize("num_vehicles,num_requests", SCENARIOS)
def test_quantum_solver(phoenix_network, results_csv, num_vehicles, num_requests):
    """Run quantum MWIS (simulated annealing), persist metrics, validate assignment."""
    node_df = phoenix_network
    seed, requests, vehicles, baseline, stats, metadata = _prepare_scenario(
        node_df, num_vehicles, num_requests
    )
    assert len(requests) > 0
    assert len(vehicles) == num_vehicles

    assignment = quantum_solver.quantum_mwis_run(
        metadata,
        baseline,
        str(results_csv),
        real=False,
        request_stats=stats,
        seed=seed,
        trial=1,
        ignore_costs=LAMBDA_VAL,
        M_val=M_VAL,
    )

    served = _validate_assignment(assignment, requests, vehicles)
    assert len(served) <= len(requests)
    assert 0.0 <= 100.0 * len(served) / len(requests) <= 100.0


def _print_results_summary():
    if not QUANTUM_RESULTS_CSV.exists():
        print("No results file found.")
        return

    df = pd.read_csv(QUANTUM_RESULTS_CSV)
    cols = [c for c in COMPARISON_COLUMNS if c in df.columns]
    summary = df[cols].sort_values(["num_vehicles", "num_requests"])

    print(f"\nResults saved to: {QUANTUM_RESULTS_CSV}")
    print("Compare these rows to the notebook's quantum results:\n")
    print(summary.to_string(index=False))
    print(f"\nFull metrics ({len(df.columns)} columns) in: {QUANTUM_RESULTS_CSV}")


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v"])
    _print_results_summary()
    raise SystemExit(exit_code)
