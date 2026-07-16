"""
Minor-embedding feasibility analysis: merge-tree vs pairwise QUBO on real D-Wave
hardware topology (Pegasus / Advantage).

For each scenario/seed/trial, builds BOTH QUBO encodings (structure only), then
attempts to minor-embed each onto a Pegasus graph using minorminer. Records:
  - logical qubit count (QUBO variables)
  - physical qubit count (sum of chain lengths) -> logical-to-physical overhead
  - max / mean chain length
  - embedding success or FAILURE (failure is itself a result: the QUBO does not
    fit on the target hardware)

This is OFFLINE and does NOT run the solver or consume quantum time. It reuses the
experiment pipeline only to build the RTV graph (trips/trip_costs injected as
globals on quantum_solver), exactly like compare_encodings_sweep.py.

minorminer is a heuristic and can be slow or fail on large dense sources. Each
embedding attempt is bounded by --timeout seconds; a timeout/failure is recorded
as embed_success=0 (meaningful: pairwise is expected to fail to embed at sizes
where merge-tree still succeeds).

Outputs under results/embedding/:
  - embedding_results.csv        one row per (scenario, seed, trial, encoding)
  - embedding_summary.csv        mean over seeds per (request size, encoding)

Usage:
  python scripts/embedding_analysis.py
  python scripts/embedding_analysis.py --quick
  python scripts/embedding_analysis.py --max-requests 60   # skip huge instances
  python scripts/embedding_analysis.py --pegasus-size 16 --timeout 60 --tries 5
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_experiment import (  # noqa: E402
    DEFAULT_TRIALS,
    SCENARIOS_BY_VEHICLES,
    load_or_build_network,
    prepare_scenario,
    scenario_seed,
)
from quantum_uav_routing.quantum import quantum_solver  # noqa: E402

try:
    from quantum_uav_routing.quantum.quantum_solver import (  # noqa: E402
        generate_qubo,
        generate_qubo_pairwise,
    )
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(
        "Could not import generate_qubo_pairwise from quantum_solver. Paste "
        "pairwise_encoding.py into quantum_solver.py first.\nError: " + str(exc)
    )

try:
    import dwave_networkx as dnx
    import minorminer
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(
        "Embedding libraries missing. Install with:\n"
        "  pip install dwave-system minorminer dwave-networkx\n"
        "Error: " + str(exc)
    )

RESULTS_SUBDIR = PROJECT_ROOT / "results" / "embedding"
QUICK_SCENARIOS = {5: [5, 10], 10: [10, 20]}


def minimal_trip_costs(module_trip_costs):
    minimal = defaultdict(lambda: float("inf"))
    for (tkey, vid), cost in module_trip_costs.items():
        if cost < minimal[tkey]:
            minimal[tkey] = cost
    return minimal


def qubo_edges(qdict):
    """Undirected edge list from a symmetric QUBO dict (skip diagonal)."""
    edges = set()
    for (i, j) in qdict.keys():
        if i != j:
            edges.add((i, j) if i < j else (j, i))
    return list(edges)


def n_vars(qdict):
    m = 0
    for (i, j) in qdict.keys():
        if i > m:
            m = i
        if j > m:
            m = j
    return m + 1


def try_embed(edges, target, tries, timeout, seed):
    """Attempt a minor-embedding. Returns dict of stats; embed_success=0 on failure."""
    t0 = time.time()
    try:
        emb = minorminer.find_embedding(
            edges, target, random_seed=seed, tries=tries, timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        emb = None
    dt = time.time() - t0

    if not emb:
        return {
            "embed_success": 0,
            "physical_qubits": None,
            "max_chain": None,
            "mean_chain": None,
            "embed_time_s": dt,
        }
    chains = [len(v) for v in emb.values()]
    phys = sum(chains)
    return {
        "embed_success": 1,
        "physical_qubits": int(phys),
        "max_chain": int(max(chains)) if chains else 0,
        "mean_chain": (phys / len(chains)) if chains else 0.0,
        "embed_time_s": dt,
    }


def load_completed(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:  # noqa: BLE001
        return set()
    need = {"num_vehicles", "num_requests", "seed", "trial", "encoding"}
    if not need.issubset(df.columns):
        return set()
    keys = set()
    for _, r in df.iterrows():
        try:
            keys.add((int(r["num_vehicles"]), int(r["num_requests"]),
                      int(r["seed"]), int(r["trial"]), str(r["encoding"])))
        except (ValueError, TypeError):
            continue
    return keys


def append_row(row: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def summarize(full_csv: Path, summary_csv: Path):
    if not full_csv.exists():
        return
    df = pd.read_csv(full_csv, low_memory=False)
    if df.empty:
        return
    for c in ("logical_qubits", "physical_qubits", "max_chain", "mean_chain",
              "embed_success"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    summary = (
        df.groupby(["num_requests", "encoding"], as_index=False)
        .agg(
            n_seeds=("embed_success", "count"),
            embed_success_rate=("embed_success", "mean"),
            logical_qubits=("logical_qubits", "mean"),
            physical_qubits=("physical_qubits", "mean"),
            max_chain=("max_chain", "mean"),
            mean_chain=("mean_chain", "mean"),
        )
        .sort_values(["num_requests", "encoding"])
    )
    # logical-to-physical overhead ratio
    summary["phys_over_logical"] = summary.apply(
        lambda r: (r["physical_qubits"] / r["logical_qubits"])
        if r["logical_qubits"] and r["physical_qubits"] and r["physical_qubits"] == r["physical_qubits"]
        else float("nan"),
        axis=1,
    )
    summary.to_csv(summary_csv, index=False)
    print("\n=== Embedding summary (mean over seeds) ===")
    show = ["num_requests", "encoding", "embed_success_rate", "logical_qubits",
            "physical_qubits", "max_chain", "phys_over_logical"]
    print(summary[show].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="Minor-embedding analysis on Pegasus.")
    p.add_argument("--city", type=str, default="32_Phoenix_City")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--pegasus-size", type=int, default=16,
                   help="Pegasus P_n. 16 = Advantage-scale (5640 qubits).")
    p.add_argument("--timeout", type=int, default=60,
                   help="Seconds per embedding attempt before giving up (=failure).")
    p.add_argument("--tries", type=int, default=5,
                   help="minorminer restart attempts per embedding.")
    p.add_argument("--max-requests", type=int, default=None,
                   help="Skip scenarios larger than this (embedding gets slow/fails).")
    p.add_argument("--lambda-val", type=float, default=5000.0)
    p.add_argument("--m-val", type=float, default=25000.0)
    p.add_argument("--cap-per-request", type=int, default=30)
    p.add_argument("--out-dir", type=str, default=str(RESULTS_SUBDIR))
    return p.parse_args()


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / "embedding_results.csv"
    summary_csv = out_dir / "embedding_summary.csv"

    if args.fresh:
        for path in (full_csv, summary_csv):
            if path.exists():
                path.unlink()

    print(f"Building Pegasus P{args.pegasus_size} target graph...")
    target = dnx.pegasus_graph(args.pegasus_size)
    print(f"  target: {target.number_of_nodes()} qubits, "
          f"{target.number_of_edges()} couplers")

    scenarios = QUICK_SCENARIOS if args.quick else SCENARIOS_BY_VEHICLES
    completed = load_completed(full_csv)
    if completed:
        print(f"Resuming: {len(completed)} embedding rows already present.")

    node_df = load_or_build_network(args.city)

    for city_index, city in enumerate([args.city]):
        for num_vehicles, request_list in scenarios.items():
            for num_requests in request_list:
                if args.max_requests is not None and num_requests > args.max_requests:
                    continue
                for trial in range(args.trials):
                    seed = scenario_seed(num_vehicles, num_requests, city_index, trial)
                    trial_num = trial + 1

                    # Skip only if BOTH encodings already done for this instance.
                    if all(
                        (num_vehicles, num_requests, seed, trial_num, enc) in completed
                        for enc in ("merge_tree", "pairwise")
                    ):
                        continue

                    print(f"\n--- {city} | v={num_vehicles} r={num_requests} "
                          f"trial={trial_num} seed={seed} ---")
                    _s, requests, vehicles, baseline, stats, metadata = prepare_scenario(
                        node_df, city, num_vehicles, num_requests, city_index, trial
                    )
                    if not requests:
                        print("  no feasible requests; skip")
                        continue

                    trips = quantum_solver.trips
                    tcosts = minimal_trip_costs(quantum_solver.trip_costs)

                    builders = {
                        "merge_tree": generate_qubo,
                        "pairwise": generate_qubo_pairwise,
                    }
                    for enc_name, builder in builders.items():
                        if (num_vehicles, num_requests, seed, trial_num, enc_name) in completed:
                            continue
                        q, _ = builder(
                            trips, tcosts, ignore_cost=args.lambda_val, M=args.m_val,
                            return_numpy=False, seed=seed,
                            cap_per_request=args.cap_per_request,
                        )
                        nv = n_vars(q)
                        edges = qubo_edges(q)
                        stats_e = try_embed(
                            edges, target, tries=args.tries,
                            timeout=args.timeout, seed=seed,
                        )
                        row = {
                            "city": city,
                            "encoding": enc_name,
                            "num_vehicles": num_vehicles,
                            "num_requests": num_requests,
                            "seed": seed,
                            "trial": trial_num,
                            "pegasus_size": args.pegasus_size,
                            "logical_qubits": nv,
                            "n_couplers": len(edges),
                            **stats_e,
                        }
                        append_row(row, full_csv)
                        ok = "OK" if stats_e["embed_success"] else "FAILED"
                        pq = stats_e["physical_qubits"]
                        mc = stats_e["max_chain"]
                        print(f"  {enc_name:10s}: logical={nv} embed={ok} "
                              f"physical={pq} max_chain={mc} "
                              f"({stats_e['embed_time_s']:.1f}s)")
                        gc.collect()

    summarize(full_csv, summary_csv)
    print(f"\nDone. Full: {full_csv}  Summary: {summary_csv}")


if __name__ == "__main__":
    main()