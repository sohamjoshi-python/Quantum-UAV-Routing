"""
Run the full Phoenix experiment pipeline across notebook scenarios.

For every scenario and trial:
  - classical greedy
  - classical ILP

For every scenario (additionally):
  - quantum simulated annealing (real=False)  [main quality solver, all sizes]
  - quantum IBM hardware (real=True), if QISKIT_IBM_TOKEN is set
    [small-instance QAOA compatibility check only; see QAOA_SCENARIOS]

This script may be launched from ANY working directory: main() changes the
working directory to the project root at startup so that all relative paths
(dataset clone dir, 3D-conversion output dir) resolve consistently with the
absolute paths this script reads from.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_uav_routing.classical import greedy_solver, ilp_solver
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

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
NODE_3D_PATH = RAW_DATA_DIR / "node_3d.csv"
LINK_3D_PATH = RAW_DATA_DIR / "link_3d.csv"
MAPPING_PATH = RAW_DATA_DIR / "node2d_to_3d_mapping.csv"

BASE_SEED = 42
DEFAULT_TRIALS = 5
LAMBDA_VAL = 5000.0
M_VAL = 25000.0

SCENARIOS_BY_VEHICLES = {
    5: [5, 10],
    10: [10, 20],
    20: [20, 30, 40],
    30: [30, 60],
    40: [40, 80],
    50: [50, 100],
    60: [60, 120],
    70: [70, 140],
    80: [80, 160],
    90: [90, 180],
}

# Quantum (D-Wave simulated annealing, real=False) runs on EVERY scenario so the
# quantum quality/waiting-time curves span the full size range, matching the paper.
# Set to None to mean "all scenarios".
QUANTUM_SCENARIOS = None

# Gate-based QAOA on real IBM hardware (real=True) is only a small-instance
# compatibility check; it requires QISKIT_IBM_TOKEN and is expensive/limited in
# qubit count, so it runs only on these small scenarios.
QAOA_SCENARIOS = {(5, 5), (5, 10)}


def scenario_seed(
    num_vehicles: int,
    num_requests: int,
    city_index: int,
    trial: int,
) -> int:
    return (
        BASE_SEED
        + trial
        + 100 * num_vehicles
        + 10_000 * num_requests
        + 1_000_000 * city_index
    )


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


def load_or_build_network(city: str):
    clone_dataset_repo()
    create_graph(city)
    _, node_df, _, _ = parse_graph(city)

    if not (NODE_3D_PATH.exists() and LINK_3D_PATH.exists() and MAPPING_PATH.exists()):
        convert_to_3d(city, str(PROJECT_ROOT / REPO_DIR / city))

    # convert_to_3d writes to <cwd>/data/raw. Because main() chdir's to
    # PROJECT_ROOT, <cwd>/data/raw == RAW_DATA_DIR. Verify that before reading so
    # a path mismatch surfaces as a clear error instead of a bare FileNotFoundError.
    missing = [
        str(path)
        for path in (NODE_3D_PATH, LINK_3D_PATH, MAPPING_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "3D network files were not found where this script reads them:\n  "
            + "\n  ".join(missing)
            + f"\nExpected them under {RAW_DATA_DIR}.\n"
            "convert_to_3d() writes to <cwd>/data/raw; ensure the working "
            "directory is the project root (main() sets this automatically)."
        )

    node_3d_df = pd.read_csv(NODE_3D_PATH, low_memory=False)
    link_3d_df = pd.read_csv(LINK_3D_PATH, low_memory=False)
    link_3d_df["length"] = link_3d_df["length"].fillna(1e-3).replace(0, 1e-3)

    mapping_df = pd.read_csv(MAPPING_PATH)
    node2d_to_3d = dict(zip(mapping_df["original_node_id"], mapping_df["node_3d_id"]))

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

    return node_df


def inject_scenario_globals(node_df, scenario_result):
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

    solver_modules = (greedy_solver, ilp_solver, quantum_solver)
    for module in solver_modules:
        module.requests = requests
        module.vehicles = vehicles
        module.vehicle_lookup = vehicle_lookup
        module.request_lookup = request_lookup
        module.trips = trips
        module.trip_to_vehicle = trip_to_vehicle
        module.trip_costs = trip_costs
        module.node_df = node_df
        module.travel = trip_builder.travel
        module.trip_name = trip_name
        module.t0 = t0
        module.t1 = t1

    quantum_solver.travel_cached = trip_builder.travel_cached

    return requests, vehicles, baseline


def prepare_scenario(node_df, city, num_vehicles, num_requests, city_index, trial):
    seed = scenario_seed(num_vehicles, num_requests, city_index, trial)
    scenario = save_results.final_trips(
        num_requests=num_requests,
        num_vehicles=num_vehicles,
        seed=seed,
    )
    requests, vehicles, baseline = inject_scenario_globals(node_df, scenario)
    stats = save_results.summarize_requests(requests)
    metadata = {"city": city, "v": num_vehicles, "r": num_requests}
    return seed, requests, vehicles, baseline, stats, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phoenix routing experiment suite.")
    parser.add_argument(
        "--city",
        type=str,
        default="32_Phoenix_City",
        help="GMNS network folder name under data/GMNS_Plus_Dataset/",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results/experiment_results.csv",
        help="Output CSV path (appends rows across runs).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of trials per scenario (seeds use trial=0..trials-1).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the results CSV before starting (disables resume).",
    )
    parser.add_argument(
        "--skip-real-quantum",
        action="store_true",
        help="Skip quantum runs with real=True even for 5/5 and 5/10.",
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=LAMBDA_VAL,
        help="Quantum ignore cost (lambda).",
    )
    parser.add_argument(
        "--m-val",
        type=float,
        default=M_VAL,
        help="Quantum penalty M.",
    )
    parser.add_argument(
        "--auto-m",
        action="store_true",
        help="Derive M per scenario as 5*lambda*nu_max (fixes capacity-3 infeasibility).",
    )
    parser.add_argument(
        "--cap-per-request",
        type=_parse_cap_per_request,
        default=30,
        help="Max trip vars per request when building the QUBO. "
             "Pass 'none' to disable pruning (keep all incident trips).",
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=1000,
        help="Simulated Annealing num_reads (ignored for real QAOA).",
    )
    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=1000,
        help="Simulated Annealing num_sweeps (ignored for real QAOA).",
    )
    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Skip classical greedy/ILP (quantum-only; useful for SA sweeps).",
    )
    parser.add_argument(
        "--cost-alpha",
        type=float,
        default=1.0,
        help="QUBO objective: w_t = lambda*|t| - cost_alpha*trip_cost (default 1.0).",
    )
    parser.add_argument(
        "--ilp-time-limit",
        type=float,
        default=ilp_solver.DEFAULT_ILP_TIME_LIMIT_S,
        help="Wall-clock limit (seconds) for each classical ILP solve. "
             "At cutoff, report best incumbent and optimality gap (default 7200 = 2h).",
    )
    return parser.parse_args()


def _parse_cap_per_request(value: str):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("none", "null", "off"):
        return None
    return int(text)


def load_completed_keys(results_csv: Path):
    """Return a set of (run_type, num_vehicles, num_requests, seed, trial) tuples
    already present in the results CSV, so a resumed run can skip them.

    Rows are per-solver, so this keys on run_type as well: a scenario whose
    ClassicalGreedy/Classical rows exist but whose Quantum row is missing will
    correctly re-run ONLY the missing Quantum row.

    For Quantum rows, entries with real_quantum_hardware are ALSO stored as
    (run_type, v, r, seed, trial, real_flag) so annealing and QAOA completions
    are distinguishable.
    """
    if not results_csv.exists():
        return set()
    try:
        done = pd.read_csv(results_csv, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read existing results ({exc}); treating as empty.")
        return set()
    keys = set()
    required = {"run_type", "num_vehicles", "num_requests", "seed", "trial"}
    if not required.issubset(done.columns):
        print("Existing CSV missing key columns; cannot resume, will append.")
        return set()
    has_real_col = "real_quantum_hardware" in done.columns
    for _, row in done.iterrows():
        try:
            rt = str(row["run_type"])
            base = (
                rt,
                int(row["num_vehicles"]),
                int(row["num_requests"]),
                int(row["seed"]),
                int(row["trial"]),
            )
            keys.add(base)
            if rt == "Quantum" and has_real_col:
                try:
                    real_flag = int(float(row["real_quantum_hardware"]))
                    keys.add(base + (real_flag,))
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError):
            continue
    return keys


def pending_solvers_for_trial(
    completed_keys: set,
    num_vehicles: int,
    num_requests: int,
    seed: int,
    trial_num: int,
    *,
    skip_classical: bool,
    run_qubo_sa: bool,
    run_real_quantum: bool,
    qaoa_scenario: bool,
) -> list[str]:
    """Return solver names still needed for this scenario/trial (empty => skip)."""
    pending: list[str] = []
    if not skip_classical:
        if ("ClassicalGreedy", num_vehicles, num_requests, seed, trial_num) not in completed_keys:
            pending.append("ClassicalGreedy")
        if ("Classical", num_vehicles, num_requests, seed, trial_num) not in completed_keys:
            pending.append("Classical")
    if run_qubo_sa and (
        "Quantum", num_vehicles, num_requests, seed, trial_num, 0
    ) not in completed_keys:
        pending.append("Quantum")
    if run_real_quantum and qaoa_scenario and (
        "Quantum", num_vehicles, num_requests, seed, trial_num, 1
    ) not in completed_keys:
        pending.append("QuantumReal")
    return pending


def find_resume_point(
    scenarios: dict,
    trials: int,
    city_index: int,
    completed_keys: set,
    *,
    skip_classical: bool,
    quantum_scenarios,
    run_real_quantum: bool,
    qaoa_scenarios: set,
) -> dict | None:
    """First (v, r, trial) with any solver still pending, or None if all done."""
    for num_vehicles, request_list in scenarios.items():
        for num_requests in request_list:
            run_qubo_sa = (
                quantum_scenarios is None
                or (num_vehicles, num_requests) in quantum_scenarios
            )
            qaoa_scenario = (num_vehicles, num_requests) in qaoa_scenarios
            for trial in range(trials):
                seed = scenario_seed(num_vehicles, num_requests, city_index, trial)
                trial_num = trial + 1
                pending = pending_solvers_for_trial(
                    completed_keys,
                    num_vehicles,
                    num_requests,
                    seed,
                    trial_num,
                    skip_classical=skip_classical,
                    run_qubo_sa=run_qubo_sa,
                    run_real_quantum=run_real_quantum,
                    qaoa_scenario=qaoa_scenario,
                )
                if pending:
                    return {
                        "num_vehicles": num_vehicles,
                        "num_requests": num_requests,
                        "trial": trial_num,
                        "seed": seed,
                        "pending": pending,
                    }
    return None


def count_pending_work(
    scenarios: dict,
    trials: int,
    city_index: int,
    completed_keys: set,
    *,
    skip_classical: bool,
    quantum_scenarios,
    run_real_quantum: bool,
    qaoa_scenarios: set,
) -> int:
    """How many scenario-trials still need at least one solver run."""
    n = 0
    for num_vehicles, request_list in scenarios.items():
        for num_requests in request_list:
            run_qubo_sa = (
                quantum_scenarios is None
                or (num_vehicles, num_requests) in quantum_scenarios
            )
            qaoa_scenario = (num_vehicles, num_requests) in qaoa_scenarios
            for trial in range(trials):
                seed = scenario_seed(num_vehicles, num_requests, city_index, trial)
                trial_num = trial + 1
                if pending_solvers_for_trial(
                    completed_keys,
                    num_vehicles,
                    num_requests,
                    seed,
                    trial_num,
                    skip_classical=skip_classical,
                    run_qubo_sa=run_qubo_sa,
                    run_real_quantum=run_real_quantum,
                    qaoa_scenario=qaoa_scenario,
                ):
                    n += 1
    return n


def main():
    # Make the script runnable from ANY working directory. Several helpers use
    # relative paths (REPO_DIR = "data/GMNS_Plus_Dataset") and os.getcwd()-based
    # output (convert_to_3d writes to <cwd>/data/raw), while this script reads via
    # absolute PROJECT_ROOT paths. Anchoring the cwd to PROJECT_ROOT makes both
    # sides agree no matter where python was launched from.
    os.chdir(PROJECT_ROOT)

    args = parse_args()
    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and results_csv.exists():
        results_csv.unlink()

    # Resume support: rows already in the CSV are skipped so an interrupted run
    # can continue without redoing work or duplicating rows. --fresh (above)
    # wipes the file first, so completed_keys is empty in that case.
    completed_keys = load_completed_keys(results_csv)
    if completed_keys and not args.fresh:
        max_trial_in_csv = max(
            key[4] for key in completed_keys if len(key) >= 5
        )
        if max_trial_in_csv > args.trials:
            print(
                f"Warning: results CSV contains trials up to {max_trial_in_csv}, "
                f"but --trials={args.trials}. Pass --trials {max_trial_in_csv} "
                "to resume unfinished larger trial indices."
            )
        pending_trials = count_pending_work(
            SCENARIOS_BY_VEHICLES,
            args.trials,
            city_index=0,
            completed_keys=completed_keys,
            skip_classical=args.skip_classical,
            quantum_scenarios=QUANTUM_SCENARIOS,
            run_real_quantum=not args.skip_real_quantum and bool(os.getenv("QISKIT_IBM_TOKEN")),
            qaoa_scenarios=QAOA_SCENARIOS,
        )
        resume = find_resume_point(
            SCENARIOS_BY_VEHICLES,
            args.trials,
            city_index=0,
            completed_keys=completed_keys,
            skip_classical=args.skip_classical,
            quantum_scenarios=QUANTUM_SCENARIOS,
            run_real_quantum=not args.skip_real_quantum and bool(os.getenv("QISKIT_IBM_TOKEN")),
            qaoa_scenarios=QAOA_SCENARIOS,
        )
        print(
            f"Resuming: {len(completed_keys)} solver-rows on disk; "
            f"{pending_trials} scenario-trial(s) still need work."
        )
        if resume:
            print(
                f"  Next up: v={resume['num_vehicles']} r={resume['num_requests']} "
                f"trial={resume['trial']} seed={resume['seed']} "
                f"pending={','.join(resume['pending'])}"
            )
        else:
            print("  All configured scenario-trials are already complete.")
            print(f"Experiment complete. Results saved to: {results_csv.resolve()}")
            return
    elif completed_keys:
        print(f"Fresh run requested; ignoring {len(completed_keys)} existing rows.")

    cities = [args.city]
    has_ibm_token = bool(os.getenv("QISKIT_IBM_TOKEN"))
    run_real_quantum = not args.skip_real_quantum and has_ibm_token

    if not args.skip_real_quantum and not has_ibm_token:
        print("QISKIT_IBM_TOKEN not set; skipping real quantum runs.")

    print(f"Loading network: {args.city}")
    node_df = load_or_build_network(args.city)
    print(f"Results file: {results_csv.resolve()}")

    skipped_complete = 0

    for city_index, city in enumerate(cities):
        trip_builder.travel_cached.cache_clear()

        for num_vehicles, request_list in SCENARIOS_BY_VEHICLES.items():
            for num_requests in request_list:
                run_qubo_sa = (
                    QUANTUM_SCENARIOS is None
                    or (num_vehicles, num_requests) in QUANTUM_SCENARIOS
                )
                qaoa_scenario = (num_vehicles, num_requests) in QAOA_SCENARIOS

                for trial in range(args.trials):
                    seed = scenario_seed(num_vehicles, num_requests, city_index, trial)
                    trial_num = trial + 1

                    pending = pending_solvers_for_trial(
                        completed_keys,
                        num_vehicles,
                        num_requests,
                        seed,
                        trial_num,
                        skip_classical=args.skip_classical,
                        run_qubo_sa=run_qubo_sa,
                        run_real_quantum=run_real_quantum,
                        qaoa_scenario=qaoa_scenario,
                    )
                    if not pending:
                        skipped_complete += 1
                        continue

                    print(
                        f"\n--- {city} | Vehicles: {num_vehicles} | "
                        f"Requests: {num_requests} | Trial: {trial_num} | "
                        f"SEED: {seed} | pending: {', '.join(pending)} ---"
                    )

                    _seed, requests, vehicles, baseline, stats, metadata = prepare_scenario(
                        node_df,
                        city,
                        num_vehicles,
                        num_requests,
                        city_index,
                        trial,
                    )

                    if not requests:
                        print("No feasible requests generated; skipping scenario.")
                        continue

                    print(
                        f"Request diagnostics | "
                        f"Infeasible windows: {stats['infeasible_windows']} | "
                        f"Slack (min/mean/median): "
                        f"{stats['min_slack']:.1f} / {stats['mean_slack']:.1f} / "
                        f"{stats['median_slack']:.1f} | "
                        f"Arrival violations: {stats['arrival_violations']}"
                    )

                    csv_path = str(results_csv)

                    if not args.skip_classical:
                        if "ClassicalGreedy" in pending:
                            greedy_solver.classical_greedy_run(
                                metadata,
                                baseline,
                                csv_path,
                                stats,
                                seed=seed,
                                trial=trial_num,
                            )
                            completed_keys.add(
                                ("ClassicalGreedy", num_vehicles, num_requests, seed, trial_num)
                            )
                        if "Classical" in pending:
                            ilp_solver.classical_ilp_run(
                                metadata,
                                baseline,
                                csv_path,
                                stats,
                                seed=seed,
                                trial=trial_num,
                                time_limit_s=args.ilp_time_limit,
                            )
                            completed_keys.add(
                                ("Classical", num_vehicles, num_requests, seed, trial_num)
                            )
                    else:
                        print("  skip classical ( --skip-classical )")

                    if "Quantum" in pending:
                        quantum_solver.quantum_mwis_run(
                            metadata,
                            baseline,
                            csv_path,
                            real=False,
                            request_stats=stats,
                            seed=seed,
                            trial=trial_num,
                            ignore_costs=args.lambda_val,
                            M_val=args.m_val,
                            auto_M=args.auto_m,
                            cap_per_request=args.cap_per_request,
                            num_reads=args.num_reads,
                            num_sweeps=args.num_sweeps,
                            cost_alpha=args.cost_alpha,
                        )
                        completed_keys.add(
                            ("Quantum", num_vehicles, num_requests, seed, trial_num, 0)
                        )

                    if "QuantumReal" in pending:
                        quantum_solver.quantum_mwis_run(
                            metadata,
                            baseline,
                            csv_path,
                            real=True,
                            request_stats=stats,
                            seed=seed,
                            trial=trial_num,
                            ignore_costs=args.lambda_val,
                            M_val=args.m_val,
                            auto_M=args.auto_m,
                            cap_per_request=args.cap_per_request,
                            num_reads=args.num_reads,
                            num_sweeps=args.num_sweeps,
                            cost_alpha=args.cost_alpha,
                        )
                        completed_keys.add(
                            ("Quantum", num_vehicles, num_requests, seed, trial_num, 1)
                        )

                    gc.collect()

    if skipped_complete:
        print(f"\nSkipped {skipped_complete} already-complete scenario-trial(s).")

    print(f"\nExperiment complete. Results saved to: {results_csv.resolve()}")


if __name__ == "__main__":
    main()