"""
Grid-search quantum hyperparameters (lambda_val, M_val) using simulated annealing.

For each scenario/seed/trial, builds the RTV graph once, then runs quantum_mwis_run
(real=False) across all (lambda, M) pairs on the same instance.

IMPORTANT — statistical robustness:
  Simulated annealing is stochastic and the solver does NOT seed the annealer, so
  repeated calls on the same instance and same (lambda, M) give DIFFERENT accuracy.
  A single draw per grid cell therefore produces an unstable ranking in which
  annealing noise can outrank the true effect of lambda/M. To control for this,
  each grid cell is evaluated REPS_PER_CELL times (independent annealing draws) per
  instance, and configurations are ranked by MEAN percent serviced with its spread
  (std / standard error). Configs whose means overlap within one standard error
  should be treated as tied — "insensitive to lambda/M" is a valid, reportable
  conclusion.

Ranking metrics:
  1. percent_serviced (post-cleanup accuracy)  -- always available
  2. raw_violation_rate (pre-cleanup infeasibility of the QUBO output) -- ranked
     ONLY if the solver exports it (see note below). This is the quantity M truly
     controls; post-cleanup accuracy is partly rectified by the classical
     conflict-cleanup + Hungarian steps and is a less sensitive signal for M.

     To enable metric 2, add this logic-free export inside quantum_mwis_run, right
     after final_trips_list is built (around line 553 of quantum_solver.py):

         raw_n = len(selected_trip_keys)
         kept_n = len(final_trips_list)
         # fraction of raw QUBO-selected trips dropped as request-conflicting:
         raw_violation_rate = (raw_n - kept_n) / raw_n if raw_n else 0.0
     and include in the output dict:
         "raw_violation_rate": raw_violation_rate,
         "raw_selected_trips": raw_n,
         "raw_kept_trips": kept_n,
     If absent, this script simply omits metric 2 and ranks on accuracy only.

Results are written under results/quantum_tuning/:
  - quantum_tuning_results.csv     full metrics (append/resume), one row PER repeat
  - accuracy_by_run.csv            focused accuracy columns per repeat
  - hyperparameter_ranking.csv     mean/std/SEM accuracy grouped by (lambda, M)
  - best_hyperparameters.json      top-ranked configuration(s), with tie flagging

Usage:
  python scripts/optimal_tuning_params.py
  python scripts/optimal_tuning_params.py --quick
  python scripts/optimal_tuning_params.py --fresh
  python scripts/optimal_tuning_params.py --reps-per-cell 5 --trials 3
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse experiment pipeline helpers (same seeds, network load, scenario prep).
from run_experiment import (  # noqa: E402
    BASE_SEED,
    DEFAULT_TRIALS,
    SCENARIOS_BY_VEHICLES,
    load_or_build_network,
    prepare_scenario,
    scenario_seed,
)
from quantum_uav_routing.quantum import quantum_solver  # noqa: E402

RESULTS_SUBDIR = PROJECT_ROOT / "results" / "quantum_tuning"
FULL_RESULTS_CSV = RESULTS_SUBDIR / "quantum_tuning_results.csv"
ACCURACY_CSV = RESULTS_SUBDIR / "accuracy_by_run.csv"
RANKING_CSV = RESULTS_SUBDIR / "hyperparameter_ranking.csv"
BEST_PARAMS_JSON = RESULTS_SUBDIR / "best_hyperparameters.json"

# Notebook grid from FINALAllPhoenix.ipynb
DEFAULT_LAMBDA_VALUES = [2500, 5000, 10000, 20000, 40000]
DEFAULT_M_MULTIPLIERS = [2, 5, 10]

# Number of independent annealing draws per (instance, lambda, M) cell.
# >1 is what makes the ranking trustworthy; 1 reproduces the old (noisy) behavior.
DEFAULT_REPS_PER_CELL = 3

# Smaller scenario set for faster sweeps (--quick).
QUICK_SCENARIOS = {
    90: [180]
}

ACCURACY_COLUMNS = [
    "city",
    "num_vehicles",
    "num_requests",
    "seed",
    "trial",
    "rep",
    "lambda_val",
    "M_val",
    "percent_serviced",
    "raw_violation_rate",
    "avg_waiting_time",
    "max_waiting_time",
    "avg_detour_factor",
    "max_detour_factor",
    "vmt",
    "real_quantum_hardware",
]


def build_hyperparameter_grid(lambda_values, m_multipliers):
    grid = []
    for lam in lambda_values:
        for mult in m_multipliers:
            grid.append((float(lam), float(mult * lam)))
    return grid


def tuning_key(num_vehicles, num_requests, seed, trial, lambda_val, m_val, rep):
    # rep is part of the key so each independent annealing draw is tracked and
    # resume skips only the specific draws already completed.
    return (
        int(num_vehicles),
        int(num_requests),
        int(seed),
        int(trial),
        float(lambda_val),
        float(m_val),
        int(rep),
    )


def load_completed_tuning_keys(results_csv: Path) -> set:
    if not results_csv.exists():
        return set()
    try:
        df = pd.read_csv(results_csv, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read {results_csv} ({exc}); starting fresh.")
        return set()

    required = {"num_vehicles", "num_requests", "seed", "trial", "lambda_val", "M_val"}
    if not required.issubset(df.columns):
        return set()

    # 'rep' may not exist in older result files; default missing rep to 0 so old
    # single-draw rows still count as (at least) rep 0.
    has_rep = "rep" in df.columns
    keys = set()
    for _, row in df.iterrows():
        try:
            rep = int(row["rep"]) if has_rep and not pd.isna(row["rep"]) else 0
            keys.add(
                tuning_key(
                    row["num_vehicles"],
                    row["num_requests"],
                    row["seed"],
                    row["trial"],
                    row["lambda_val"],
                    row["M_val"],
                    rep,
                )
            )
        except (ValueError, TypeError):
            continue
    return keys


def append_accuracy_row(row: dict, accuracy_csv: Path):
    accuracy_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    # Append in a single write when possible to avoid re-reading a growing file
    # on every cell (the old version re-read + rewrote the whole CSV each time).
    if accuracy_csv.exists():
        df_new.to_csv(accuracy_csv, mode="a", header=False, index=False)
    else:
        df_new.to_csv(accuracy_csv, index=False)


def summarize_and_rank(full_csv: Path, ranking_csv: Path, best_json: Path):
    if not full_csv.exists():
        print("No results to summarize.")
        return

    df = pd.read_csv(full_csv, low_memory=False)
    if df.empty or "percent_serviced" not in df.columns:
        print("Results file has no percent_serviced column.")
        return

    df["percent_serviced"] = pd.to_numeric(df["percent_serviced"], errors="coerce")
    df["lambda_val"] = pd.to_numeric(df["lambda_val"], errors="coerce")
    df["M_val"] = pd.to_numeric(df["M_val"], errors="coerce")

    has_violation = "raw_violation_rate" in df.columns
    if has_violation:
        df["raw_violation_rate"] = pd.to_numeric(df["raw_violation_rate"], errors="coerce")

    agg = {
        "mean_percent_serviced": ("percent_serviced", "mean"),
        "std_percent_serviced": ("percent_serviced", "std"),
        "median_percent_serviced": ("percent_serviced", "median"),
        "min_percent_serviced": ("percent_serviced", "min"),
        "max_percent_serviced": ("percent_serviced", "max"),
        "n_runs": ("percent_serviced", "count"),
    }
    if has_violation:
        agg["mean_raw_violation_rate"] = ("raw_violation_rate", "mean")
        agg["std_raw_violation_rate"] = ("raw_violation_rate", "std")

    ranking = (
        df.groupby(["lambda_val", "M_val"], as_index=False).agg(**agg)
    )

    # Standard error of the mean accuracy = std / sqrt(n).
    ranking["sem_percent_serviced"] = ranking.apply(
        lambda r: (r["std_percent_serviced"] / math.sqrt(r["n_runs"]))
        if r["n_runs"] and r["n_runs"] > 0 and not pd.isna(r["std_percent_serviced"])
        else float("nan"),
        axis=1,
    )

    # Primary ranking: accuracy (higher is better).
    ranking_acc = ranking.sort_values(
        ["mean_percent_serviced", "median_percent_serviced", "max_percent_serviced"],
        ascending=False,
    ).reset_index(drop=True)
    ranking_acc.to_csv(ranking_csv, index=False)

    best = ranking_acc.iloc[0]
    best_mean = best["mean_percent_serviced"]
    best_sem = best["sem_percent_serviced"]

    # Flag configs statistically tied with the best (means within 1 SEM of best).
    tie_threshold = best_sem if (best_sem and not math.isnan(best_sem)) else 0.0
    tied = ranking_acc[
        ranking_acc["mean_percent_serviced"] >= (best_mean - tie_threshold)
    ]

    payload = {
        "ranked_by": "mean_percent_serviced",
        "best_lambda_val": float(best["lambda_val"]),
        "best_M_val": float(best["M_val"]),
        "best_mean_percent_serviced": float(best_mean),
        "best_sem_percent_serviced": (
            float(best_sem) if best_sem and not math.isnan(best_sem) else None
        ),
        "best_n_runs": int(best["n_runs"]),
        "n_configs_tied_within_1sem": int(len(tied)),
        "tied_configs": [
            {
                "lambda_val": float(r["lambda_val"]),
                "M_val": float(r["M_val"]),
                "mean_percent_serviced": float(r["mean_percent_serviced"]),
            }
            for _, r in tied.iterrows()
        ],
    }

    # Secondary ranking on raw violation rate (lower is better), if available.
    if has_violation and "mean_raw_violation_rate" in ranking_acc.columns:
        ranking_viol = ranking_acc.sort_values(
            "mean_raw_violation_rate", ascending=True
        ).reset_index(drop=True)
        vbest = ranking_viol.iloc[0]
        payload["violation_optimal"] = {
            "lambda_val": float(vbest["lambda_val"]),
            "M_val": float(vbest["M_val"]),
            "mean_raw_violation_rate": float(vbest["mean_raw_violation_rate"]),
            "note": (
                "This is the config that MINIMIZES pre-cleanup QUBO infeasibility, "
                "i.e. what M actually controls. May differ from the accuracy-optimal "
                "config because the classical cleanup + Hungarian steps rectify much "
                "of the raw infeasibility before accuracy is measured."
            ),
        }

    best_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== Top configs by MEAN % serviced (with spread) ===")
    show_cols = [
        "lambda_val", "M_val", "mean_percent_serviced",
        "sem_percent_serviced", "std_percent_serviced", "n_runs",
    ]
    print(ranking_acc[show_cols].head(6).to_string(index=False))

    if payload["n_configs_tied_within_1sem"] > 1:
        print(
            f"\nNOTE: {payload['n_configs_tied_within_1sem']} configs are within 1 SEM "
            f"of the best (mean {best_mean:.2f} ± {tie_threshold:.2f}). Treat these as "
            f"statistically TIED — accuracy is largely insensitive to (lambda, M) here."
        )
    else:
        print(
            f"\nBest config is separated from the rest by more than 1 SEM: "
            f"lambda={payload['best_lambda_val']}, M={payload['best_M_val']}, "
            f"mean {best_mean:.2f} ± {tie_threshold:.2f}."
        )

    if "violation_optimal" in payload:
        vo = payload["violation_optimal"]
        print(
            f"\nViolation-optimal (minimizes raw QUBO infeasibility): "
            f"lambda={vo['lambda_val']}, M={vo['M_val']}, "
            f"mean raw violation rate={vo['mean_raw_violation_rate']:.3f}"
        )
    else:
        print(
            "\n(raw_violation_rate not found in results; ranked on accuracy only. "
            "Add the optional solver export described in this file's docstring to "
            "also rank on the metric M truly controls.)"
        )

    print(f"\nBest config(s) saved to {best_json}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Grid-search quantum lambda/M for maximum accuracy (simulator only)."
    )
    p.add_argument("--city", type=str, default="32_Phoenix_City")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument(
        "--reps-per-cell",
        type=int,
        default=DEFAULT_REPS_PER_CELL,
        help=(
            "Independent annealing draws per (instance, lambda, M) cell. >1 averages "
            "out annealing noise so the ranking is trustworthy. Default 3."
        ),
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller scenario set (5/5,5/10 and 10/10,10/20) for faster sweeps.",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing tuning results before starting.",
    )
    p.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDA_VALUES,
        help="Ignore-cost values to sweep.",
    )
    p.add_argument(
        "--m-multipliers",
        type=int,
        nargs="+",
        default=DEFAULT_M_MULTIPLIERS,
        help="M_val = multiplier * lambda_val for each multiplier.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(RESULTS_SUBDIR),
        help="Output directory under results/.",
    )
    return p.parse_args()


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    if args.reps_per_cell < 1:
        raise SystemExit("--reps-per-cell must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / "quantum_tuning_results.csv"
    accuracy_csv = out_dir / "accuracy_by_run.csv"
    ranking_csv = out_dir / "hyperparameter_ranking.csv"
    best_json = out_dir / "best_hyperparameters.json"

    if args.fresh:
        for path in (full_csv, accuracy_csv, ranking_csv, best_json):
            if path.exists():
                path.unlink()

    scenarios = QUICK_SCENARIOS if args.quick else SCENARIOS_BY_VEHICLES
    grid = build_hyperparameter_grid(args.lambda_values, args.m_multipliers)
    # Resume keys must come from the rep-aware accuracy CSV, because the solver
    # writes full_csv rows WITHOUT a 'rep' column. accuracy_csv is stamped with
    # rep by this script, so it correctly distinguishes individual draws.
    completed = load_completed_tuning_keys(accuracy_csv)

    total_runs = (
        sum(len(rs) for rs in scenarios.values())
        * args.trials
        * len(grid)
        * args.reps_per_cell
    )
    print(f"City: {args.city}")
    print(f"Scenarios: {sum(len(v) for v in scenarios.values())} request sizes")
    print(f"Trials per scenario: {args.trials}")
    print(f"Hyperparameter grid: {len(grid)} (lambda x M) combinations")
    print(f"Reps per cell (annealing draws): {args.reps_per_cell}")
    print(f"Planned quantum runs (max): {total_runs}")
    print(f"Output directory: {out_dir.resolve()}")
    if completed:
        print(f"Resuming: {len(completed)} runs already completed will be skipped.")

    node_df = load_or_build_network(args.city)
    cities = [args.city]
    runs_done = 0
    runs_skipped = 0

    for city_index, city in enumerate(cities):
        for num_vehicles, request_list in scenarios.items():
            for num_requests in request_list:
                for trial in range(args.trials):
                    seed = scenario_seed(num_vehicles, num_requests, city_index, trial)
                    trial_num = trial + 1

                    print(
                        f"\n--- {city} | v={num_vehicles} | r={num_requests} | "
                        f"trial={trial_num} | seed={seed} ---"
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
                        print("  No feasible requests; skipping scenario.")
                        continue

                    for lambda_val, m_val in grid:
                        for rep in range(args.reps_per_cell):
                            key = tuning_key(
                                num_vehicles, num_requests, seed,
                                trial_num, lambda_val, m_val, rep,
                            )
                            if key in completed:
                                runs_skipped += 1
                                continue

                            print(
                                f"  quantum sim | lambda={lambda_val} M={m_val} "
                                f"rep={rep + 1}/{args.reps_per_cell}"
                            )
                            quantum_solver.quantum_mwis_run(
                                metadata,
                                baseline,
                                str(full_csv),
                                real=False,
                                request_stats=stats,
                                seed=seed,
                                trial=trial_num,
                                ignore_costs=lambda_val,
                                M_val=m_val,
                            )

                            # Mirror key accuracy fields into a dedicated summary
                            # file, tagging the rep index. Note: the solver writes
                            # its own row to full_csv WITHOUT a 'rep' column, so we
                            # stamp rep here for the accuracy summary and rely on
                            # (…,rep) resume keys derived below.
                            if full_csv.exists():
                                last = pd.read_csv(full_csv, low_memory=False).iloc[-1]
                                acc_row = {
                                    col: last.get(col)
                                    for col in ACCURACY_COLUMNS
                                    if col in last.index
                                }
                                acc_row["rep"] = rep
                                append_accuracy_row(acc_row, accuracy_csv)

                            completed.add(key)
                            runs_done += 1

                    gc.collect()

    summarize_and_rank(full_csv, ranking_csv, best_json)

    print(f"\nTuning sweep complete.")
    print(f"  New runs:     {runs_done}")
    print(f"  Skipped:      {runs_skipped}")
    print(f"  Full results: {full_csv}")
    print(f"  Accuracy:     {accuracy_csv}")
    print(f"  Ranking:      {ranking_csv}")


if __name__ == "__main__":
    main()