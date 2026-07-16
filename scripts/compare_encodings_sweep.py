"""
Sweep the merge-tree vs pairwise QUBO encodings across the full scenario range
and record a side-by-side structural comparison (Reviewer 3's requested table).

This is a STRUCTURAL sweep only: it builds both QUBOs and measures variables,
couplers, and density. It does NOT solve them (no annealing), so it is fast.

For each scenario/seed/trial it reuses the experiment pipeline to build the RTV
graph once (via prepare_scenario, which injects trips/trip_costs as module globals
on quantum_solver), then constructs both encodings on the SAME pruned trip set.

Outputs under results/encoding_comparison/:
  - encoding_comparison.csv   one row per (scenario, seed, trial): vars/couplers/
                              density for both encodings + ratios
  - encoding_comparison_summary.csv   mean over trials per request size
  - encoding_crossover.png    couplers vs #requests for both encodings (log-log)

Usage:
  python scripts/compare_encodings_sweep.py
  python scripts/compare_encodings_sweep.py --quick
  python scripts/compare_encodings_sweep.py --fresh
  python scripts/compare_encodings_sweep.py --cap-per-request 30
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
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

# The encoding functions live in quantum_solver after you paste pairwise_encoding.py
# into it. Import defensively so this script gives a clear error if they're missing.
try:
    from quantum_uav_routing.quantum.quantum_solver import (  # noqa: E402
        generate_qubo,
        generate_qubo_pairwise,
        qubo_stats_from_dict,
    )
except ImportError as exc:  # noqa: BLE001
    raise SystemExit(
        "Could not import generate_qubo_pairwise / qubo_stats_from_dict from "
        "quantum_solver. Paste the contents of pairwise_encoding.py into "
        "quantum_solver.py first.\nOriginal error: " + str(exc)
    )

RESULTS_SUBDIR = PROJECT_ROOT / "results" / "encoding_comparison"

QUICK_SCENARIOS = {5: [5, 10], 10: [10, 20]}


def minimal_trip_costs(module_trip_costs):
    """Reproduce the per-trip minimal cost reduction that quantum_mwis_run does,
    so the encodings are built on the same objective the solver would use."""
    minimal = defaultdict(lambda: float("inf"))
    for (tkey, vid), cost in module_trip_costs.items():
        if cost < minimal[tkey]:
            minimal[tkey] = cost
    return minimal


def load_completed_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:  # noqa: BLE001
        return set()
    req = {"num_vehicles", "num_requests", "seed", "trial"}
    if not req.issubset(df.columns):
        return set()
    keys = set()
    for _, r in df.iterrows():
        try:
            keys.add((int(r["num_vehicles"]), int(r["num_requests"]),
                      int(r["seed"]), int(r["trial"])))
        except (ValueError, TypeError):
            continue
    return keys


def append_row(row: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)


def make_summary_and_plot(full_csv: Path, summary_csv: Path, plot_path: Path):
    if not full_csv.exists():
        print("No results to summarize.")
        return
    df = pd.read_csv(full_csv, low_memory=False)
    if df.empty:
        print("Results file is empty.")
        return

    for c in ("tree_couplers", "pair_couplers", "tree_vars", "pair_vars",
              "tree_density", "pair_density", "couplers_pair_over_tree"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    summary = (
        df.groupby("num_requests", as_index=False)
        .agg(
            tree_vars=("tree_vars", "mean"),
            pair_vars=("pair_vars", "mean"),
            tree_couplers=("tree_couplers", "mean"),
            pair_couplers=("pair_couplers", "mean"),
            tree_density=("tree_density", "mean"),
            pair_density=("pair_density", "mean"),
            couplers_pair_over_tree=("couplers_pair_over_tree", "mean"),
            n_trials=("tree_couplers", "count"),
        )
        .sort_values("num_requests")
    )
    summary.to_csv(summary_csv, index=False)
    print("\n=== Encoding comparison (mean over trials) ===")
    print(
        summary[
            ["num_requests", "tree_vars", "pair_vars",
             "tree_couplers", "pair_couplers", "couplers_pair_over_tree"]
        ].to_string(index=False)
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(summary["num_requests"], summary["tree_couplers"],
                "o-", label="Merge-tree couplers")
        ax.plot(summary["num_requests"], summary["pair_couplers"],
                "s-", label="Pairwise couplers")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Requests")
        ax.set_ylabel("QUBO Couplers")
        ax.set_title("Encoding Comparison: Merge-Tree vs Pairwise Couplers")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"\nCrossover plot saved to {plot_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"(Plot skipped: {exc})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Structural sweep: merge-tree vs pairwise QUBO encoding."
    )
    p.add_argument("--city", type=str, default="32_Phoenix_City")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--quick", action="store_true",
                   help="Small scenario set for a fast sanity sweep.")
    p.add_argument("--fresh", action="store_true",
                   help="Delete existing comparison results before starting.")
    p.add_argument("--lambda-val", type=float, default=5000.0,
                   help="ignore_cost used when scoring/pruning trips (match your run).")
    p.add_argument("--m-val", type=float, default=25000.0,
                   help="Penalty weight M (structure is independent of M, but kept "
                        "for parity with the solver).")
    p.add_argument("--cap-per-request", type=int, default=30,
                   help="Per-request trip cap used in BOTH encodings. This bounds "
                        "k_r and therefore the pairwise blowup; report it in the paper.")
    p.add_argument("--out-dir", type=str, default=str(RESULTS_SUBDIR))
    return p.parse_args()


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / "encoding_comparison.csv"
    summary_csv = out_dir / "encoding_comparison_summary.csv"
    plot_path = out_dir / "encoding_crossover.png"

    if args.fresh:
        for path in (full_csv, summary_csv, plot_path):
            if path.exists():
                path.unlink()

    scenarios = QUICK_SCENARIOS if args.quick else SCENARIOS_BY_VEHICLES
    completed = load_completed_keys(full_csv)

    print(f"City: {args.city}")
    print(f"Cap per request: {args.cap_per_request} (bounds pairwise blowup; disclose in paper)")
    print(f"Scenarios: {sum(len(v) for v in scenarios.values())} request sizes, "
          f"{args.trials} trials each")
    if completed:
        print(f"Resuming: {len(completed)} rows already present will be skipped.")

    node_df = load_or_build_network(args.city)

    for city_index, city in enumerate([args.city]):
        for num_vehicles, request_list in scenarios.items():
            for num_requests in request_list:
                for trial in range(args.trials):
                    seed = scenario_seed(num_vehicles, num_requests, city_index, trial)
                    trial_num = trial + 1
                    if (num_vehicles, num_requests, seed, trial_num) in completed:
                        continue

                    print(f"\n--- {city} | v={num_vehicles} | r={num_requests} | "
                          f"trial={trial_num} | seed={seed} ---")

                    _seed, requests, vehicles, baseline, stats, metadata = prepare_scenario(
                        node_df, city, num_vehicles, num_requests, city_index, trial
                    )
                    if not requests:
                        print("  No feasible requests; skipping.")
                        continue

                    # trips/trip_costs were injected onto quantum_solver as globals.
                    trips = quantum_solver.trips
                    tcosts = minimal_trip_costs(quantum_solver.trip_costs)

                    # Build BOTH encodings on the SAME pruned trip set.
                    q_tree, _ = generate_qubo(
                        trips, tcosts, ignore_cost=args.lambda_val, M=args.m_val,
                        return_numpy=False, seed=seed,
                        cap_per_request=args.cap_per_request,
                    )
                    q_pair, _ = generate_qubo_pairwise(
                        trips, tcosts, ignore_cost=args.lambda_val, M=args.m_val,
                        return_numpy=False, seed=seed,
                        cap_per_request=args.cap_per_request,
                    )
                    s_tree = qubo_stats_from_dict(q_tree)
                    s_pair = qubo_stats_from_dict(q_pair)

                    tree_cpl = s_tree["qubo_couplers"]
                    pair_cpl = s_pair["qubo_couplers"]
                    ratio = (pair_cpl / tree_cpl) if tree_cpl else float("inf")

                    row = {
                        "city": city,
                        "num_vehicles": num_vehicles,
                        "num_requests": num_requests,
                        "seed": seed,
                        "trial": trial_num,
                        "cap_per_request": args.cap_per_request,
                        "n_trips_pruned": s_pair["qubo_vars"],  # pairwise vars == #trips
                        "tree_vars": s_tree["qubo_vars"],
                        "pair_vars": s_pair["qubo_vars"],
                        "tree_couplers": tree_cpl,
                        "pair_couplers": pair_cpl,
                        "tree_density": s_tree["qubo_graph_density"],
                        "pair_density": s_pair["qubo_graph_density"],
                        "tree_degree_max": s_tree["degree_max"],
                        "pair_degree_max": s_pair["degree_max"],
                        "couplers_pair_over_tree": ratio,
                        "aux_vars_tree": s_tree["qubo_vars"] - s_pair["qubo_vars"],
                    }
                    append_row(row, full_csv)
                    print(f"  trips={row['n_trips_pruned']}  "
                          f"tree: {s_tree['qubo_vars']}v/{tree_cpl}c  "
                          f"pair: {s_pair['qubo_vars']}v/{pair_cpl}c  "
                          f"pair/tree couplers={ratio:.2f}x")
                    gc.collect()

    make_summary_and_plot(full_csv, summary_csv, plot_path)
    print("\nEncoding comparison sweep complete.")
    print(f"  Full:    {full_csv}")
    print(f"  Summary: {summary_csv}")
    print(f"  Plot:    {plot_path}")


if __name__ == "__main__":
    main()