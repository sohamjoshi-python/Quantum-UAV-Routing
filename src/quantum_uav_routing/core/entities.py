from __future__ import annotations

import argparse
from pathlib import Path

from quantum_uav_routing.network import (
    load_gmns_network,
    convert_to_3d,
    build_3d_graph,
    ShortestPathOracle,
)
from quantum_uav_routing.scenario import build_scenario_artifacts, summarize_requests

# replace these imports with your actual refactored solver locations
from quantum_uav_routing.classical.ilp_solver import classical_ilp_run
from quantum_uav_routing.classical.greedy_solver import classical_greedy_run
from quantum_uav_routing.quantum.quantum_solver import quantum_mwis_run
from quantum_uav_routing.rtv.trip_builder import build_trips


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--network-dir", type=str, required=True)
    p.add_argument("--num-requests", type=int, required=True)
    p.add_argument("--num-vehicles", type=int, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--solver", choices=["greedy", "classical", "quantum"], required=True)
    p.add_argument("--results-csv", type=str, default="results/results.csv")
    p.add_argument("--real-quantum", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    network_dir = Path(args.network_dir)
    node_df, link_df, zone_df = load_gmns_network(network_dir)

    node_3d_df, link_3d_df, node2d_to_3d = convert_to_3d(node_df, link_df)
    G = build_3d_graph(node_3d_df, link_3d_df)
    oracle = ShortestPathOracle(G)

    scenario = build_scenario_artifacts(
        num_requests=args.num_requests,
        num_vehicles=args.num_vehicles,
        node2d_to_3d=node2d_to_3d,
        shortest_path_fn=oracle.shortest_path,
        node_df=node_df,
        build_trips_fn=build_trips,
        seed=args.seed,
        nu=4,
    )

    request_stats = summarize_requests(scenario.requests)
    metadata = {"city": network_dir.name}

    if args.solver == "greedy":
        classical_greedy_run(
            metadata=metadata,
            baseline_vehicle_costs=scenario.baseline_vehicle_costs,
            requests=scenario.requests,
            vehicles=scenario.vehicles,
            request_lookup=scenario.request_lookup,
            vehicle_lookup=scenario.vehicle_lookup,
            trips=scenario.trips,
            trip_to_vehicle=scenario.trip_to_vehicle,
            trip_costs=scenario.trip_costs,
            node_df=node_df,
            rtv_graph_build_time=scenario.rtv_graph_build_time,
            csv_filename=args.results_csv,
            stats=request_stats,
            seed=args.seed,
            trial=1,
        )

    elif args.solver == "classical":
        classical_ilp_run(
            metadata=metadata,
            baseline_vehicle_costs=scenario.baseline_vehicle_costs,
            requests=scenario.requests,
            vehicles=scenario.vehicles,
            request_lookup=scenario.request_lookup,
            vehicle_lookup=scenario.vehicle_lookup,
            trips=scenario.trips,
            trip_to_vehicle=scenario.trip_to_vehicle,
            trip_costs=scenario.trip_costs,
            node_df=node_df,
            rtv_graph_build_time=scenario.rtv_graph_build_time,
            csv_filename=args.results_csv,
            stats=request_stats,
            seed=args.seed,
            trial=1,
        )

    else:
        quantum_mwis_run(
            metadata=metadata,
            baseline_vehicle_costs=scenario.baseline_vehicle_costs,
            requests=scenario.requests,
            vehicles=scenario.vehicles,
            request_lookup=scenario.request_lookup,
            vehicle_lookup=scenario.vehicle_lookup,
            trips=scenario.trips,
            trip_to_vehicle=scenario.trip_to_vehicle,
            trip_costs=scenario.trip_costs,
            node_df=node_df,
            rtv_graph_build_time=scenario.rtv_graph_build_time,
            csv_filename=args.results_csv,
            request_stats=request_stats,
            real=args.real_quantum,
            seed=args.seed,
            trial=1,
        )


if __name__ == "__main__":
    main()