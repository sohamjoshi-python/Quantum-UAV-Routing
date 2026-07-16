"""
Rank (lambda_val, M_val) configs SEPARATELY for every (num_requests, num_vehicles)
scenario size present in accuracy_by_run.csv, using the same mean/std/SEM +
1-SEM tie-flagging logic as summarize_and_rank() in optimal_tuning_params.py.

This is the multi-size companion to rank_90v180.py: instead of isolating just
90/180, it produces one ranking table per scenario size, plus a combined
cross-size summary showing how each config's rank shifts across sizes. Use
this to check whether a chosen config (e.g. the 90/180 winner) is a genuine
consistent performer or a size-specific fluke, before writing up hyperparameter
selection in the paper's Methods section.

Usage:
    python rank_all_scenarios.py --csv results/quantum_tuning/accuracy_by_run.csv
    python rank_all_scenarios.py --csv accuracy_by_run.csv --out-dir per_size_rankings
"""

import argparse
import math
from pathlib import Path

import pandas as pd


def rank_one_scenario(sub: pd.DataFrame) -> pd.DataFrame:
    grouped = sub.groupby(["lambda_val", "M_val"]).agg(
        mean_percent_serviced=("percent_serviced", "mean"),
        std_percent_serviced=("percent_serviced", "std"),
        median_percent_serviced=("percent_serviced", "median"),
        min_percent_serviced=("percent_serviced", "min"),
        max_percent_serviced=("percent_serviced", "max"),
        n_runs=("percent_serviced", "count"),
    ).reset_index()

    grouped["sem_percent_serviced"] = grouped.apply(
        lambda r: (r["std_percent_serviced"] / math.sqrt(r["n_runs"]))
        if r["n_runs"] and r["n_runs"] > 0 and not pd.isna(r["std_percent_serviced"])
        else float("nan"),
        axis=1,
    )

    ranked = grouped.sort_values(
        ["mean_percent_serviced", "median_percent_serviced", "max_percent_serviced"],
        ascending=False,
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    best_mean = ranked.iloc[0]["mean_percent_serviced"]
    best_sem = ranked.iloc[0]["sem_percent_serviced"]
    tie_threshold = best_sem if (best_sem and not math.isnan(best_sem)) else 0.0
    ranked["tied_with_best"] = ranked["mean_percent_serviced"] >= (best_mean - tie_threshold)

    return ranked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to accuracy_by_run.csv")
    p.add_argument("--out-dir", type=str, default="per_size_rankings")
    p.add_argument("--min-runs", type=int, default=3,
                    help="Skip scenario sizes with fewer than this many total runs.")
    args = p.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)

    required_cols = {"num_requests", "num_vehicles", "lambda_val", "M_val", "percent_serviced"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    for col in ("num_requests", "num_vehicles", "lambda_val", "M_val", "percent_serviced"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_sizes = (
        df[["num_requests", "num_vehicles"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["num_requests", "num_vehicles"])
        .itertuples(index=False)
    )

    combined_rows = []
    per_size_tables = {}

    for num_requests, num_vehicles in scenario_sizes:
        sub = df[(df["num_requests"] == num_requests) & (df["num_vehicles"] == num_vehicles)]
        if len(sub) < args.min_runs:
            print(f"Skipping {num_requests}/{num_vehicles}: only {len(sub)} runs (< --min-runs).")
            continue

        ranked = rank_one_scenario(sub)
        label = f"{int(num_requests)}r_{int(num_vehicles)}v"
        per_size_tables[label] = ranked

        out_path = out_dir / f"ranking_{label}.csv"
        ranked.to_csv(out_path, index=False)

        best = ranked.iloc[0]
        n_tied = int(ranked["tied_with_best"].sum())
        print(f"\n=== {int(num_requests)} requests / {int(num_vehicles)} vehicles "
              f"(n_runs total={len(sub)}) ===")
        print(f"Best: lambda={best['lambda_val']}, M={best['M_val']}, "
              f"mean={best['mean_percent_serviced']:.2f} "
              f"+/- {best['sem_percent_serviced']:.2f} SEM  "
              f"(n_configs_tied_within_1sem={n_tied})")
        print(f"Saved: {out_path}")

        for _, row in ranked.iterrows():
            combined_rows.append({
                "num_requests": num_requests,
                "num_vehicles": num_vehicles,
                "lambda_val": row["lambda_val"],
                "M_val": row["M_val"],
                "mean_percent_serviced": row["mean_percent_serviced"],
                "sem_percent_serviced": row["sem_percent_serviced"],
                "n_runs": row["n_runs"],
                "rank": row["rank"],
                "tied_with_best": row["tied_with_best"],
            })

    if not combined_rows:
        print("\nNo scenario sizes had enough runs to rank. Nothing to summarize.")
        return

    combined = pd.DataFrame(combined_rows)
    combined_path = out_dir / "combined_cross_size_ranks.csv"
    combined.to_csv(combined_path, index=False)

    # Cross-size consistency summary: for each config, how does its rank
    # behave across all scenario sizes it appeared in?
    consistency = combined.groupby(["lambda_val", "M_val"]).agg(
        mean_rank=("rank", "mean"),
        best_rank=("rank", "min"),
        worst_rank=("rank", "max"),
        n_sizes_seen=("rank", "count"),
        n_sizes_tied_with_best=("tied_with_best", "sum"),
    ).reset_index().sort_values("mean_rank")

    consistency_path = out_dir / "cross_size_consistency_summary.csv"
    consistency.to_csv(consistency_path, index=False)

    print("\n=== Cross-size consistency summary (lower mean_rank = more consistently good) ===")
    print(consistency.to_string(index=False))
    print(f"\nCombined per-size table: {combined_path}")
    print(f"Consistency summary:     {consistency_path}")
    print(f"Individual size tables:  {out_dir}/ranking_<requests>r_<vehicles>v.csv")


if __name__ == "__main__":
    main()