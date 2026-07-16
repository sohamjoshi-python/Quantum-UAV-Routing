"""
Analyze experiment_results.csv with the same plots/tables as
AnalyzeOutput_with_greedy.ipynb.

Usage:
  python scripts/compare_results.py
  python scripts/compare_results.py --csv results/experiment_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "results" / "experiment_results.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "analysis_figures"
TABLES_DIR = RESULTS_DIR / "analysis_tables"

# Notebook-style D-Wave QPU timing assumptions (for modeled quantum time).
NUM_READS = 2000
ANNEAL_US = 50
READOUT_US = 120
QPU_DELAY_US = 20
PROGRAMMING_MS = 10
OTHER_OVERHEAD_MS = 5
QUEUE_OVERHEAD_S = 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Analyze experiment_results.csv")
    p.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help="Path to experiment results CSV",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Results directory (figures go in analysis_figures/, tables in analysis_tables/)",
    )
    p.add_argument(
        "--per-request-csv",
        type=str,
        default=None,
        help="Per-request side file. Defaults to <csv stem>_per_request.csv next to --csv.",
    )
    return p.parse_args()


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def avg_by_requests(d: pd.DataFrame, col: str) -> pd.DataFrame:
    t = d[["num_requests", col]].dropna()
    return t.groupby("num_requests", as_index=False)[col].mean()


def avg_by_requests_sem(d: pd.DataFrame, col: str) -> pd.DataFrame:
    t = d[["num_requests", col]].dropna()
    grouped = t.groupby("num_requests")[col].agg(["mean", "sem", "count"]).reset_index()
    grouped.columns = ["num_requests", f"{col}_mean", f"{col}_sem", f"{col}_count"]
    return grouped


def fit_powerlaw(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    lx = np.log10(x)
    ly = np.log10(y)
    a, b = np.polyfit(lx, ly, 1)
    ly_hat = a * lx + b
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - np.sum((ly - ly_hat) ** 2) / ss_tot if ss_tot > 0 else np.nan
    return a, 10**b, r2


def first_present(frame: pd.DataFrame, cols):
    for c in cols:
        if c in frame.columns:
            return c
    return None


def safe_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def dwave_qpu_time_seconds(
    num_reads=NUM_READS,
    anneal_us=ANNEAL_US,
    readout_us=READOUT_US,
    qpu_delay_us=QPU_DELAY_US,
    programming_ms=PROGRAMMING_MS,
    other_overhead_ms=OTHER_OVERHEAD_MS,
    queue_overhead_s=QUEUE_OVERHEAD_S,
) -> float:
    per_sample_us = anneal_us + readout_us + qpu_delay_us
    sampling_s = (num_reads * per_sample_us) * 1e-6
    fixed_s = (programming_ms + other_overhead_ms) * 1e-3
    return queue_overhead_s + fixed_s + sampling_s


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["run_type"] = df["run_type"].astype(str).str.strip()

    numeric_cols = [
        "num_requests",
        "num_vehicles",
        "seed",
        "trial",
        "percent_serviced",
        "avg_waiting_time",
        "max_waiting_time",
        "avg_detour_factor",
        "max_detour_factor",
        "vmt",
        "total_run_time",
        "solve_time",
        "rtv_graph_build_time",
        "qubo_build_time",
        "time_min_cost_prep",
        "time_qubo_gen",
        "time_compress",
        "time_quantum_solve",
        "time_decode",
        "time_vehicle_assignment",
        "time_metrics_calc",
        "time_struct_stats",
        "time_total_quantum_block",
        "time_greedy_total",
        "base_qubo_vars",
        "base_qubo_couplers",
        "ilp_num_integer_vars",
        "ilp_num_nonzero_coeffs",
        "greedy_sort_work",
        "arrival_violations",
        "infeasible_windows",
        "real_quantum_hardware",
        "lambda_val",
        "M_val",
        "raw_selected_trips",
        "raw_kept_trips",
        "raw_dropped_trips",
        "raw_violation_rate",
        "raw_infeasible_instance",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "arrival_violations" in df.columns and "num_requests" in df.columns:
        df["violation_rate"] = df["arrival_violations"] / df["num_requests"]
    elif "infeasible_windows" in df.columns and "num_requests" in df.columns:
        df["violation_rate"] = df["infeasible_windows"] / df["num_requests"]

    # Friendly method label for quantum hardware vs simulator
    def method_label(row):
        rt = row["run_type"]
        if rt != "Quantum":
            return rt
        hw = row.get("real_quantum_hardware", 0)
        if pd.notna(hw) and float(hw) >= 0.5:
            return "Quantum (real hardware)"
        return "Quantum (simulator)"

    df["method"] = df.apply(method_label, axis=1)
    return df


def split_solvers(df: pd.DataFrame):
    d_class = df[df["run_type"] == "Classical"].copy()
    d_greedy = df[df["run_type"] == "ClassicalGreedy"].copy()
    d_quant = df[df["run_type"] == "Quantum"].copy()
    d_quant_sim = d_quant[d_quant["real_quantum_hardware"].fillna(0) < 0.5].copy()
    d_quant_real = d_quant[d_quant["real_quantum_hardware"].fillna(0) >= 0.5].copy()
    return d_class, d_greedy, d_quant, d_quant_sim, d_quant_real


def plot_decision_dimensions(d_class, d_greedy, d_quant, figures_dir: Path, tables_dir: Path):
    print("\n[1/8] Decision variable / interaction density scaling")
    qubo_src = d_quant if not d_quant.empty else d_quant
    # Prefer simulator rows for QUBO structure (same QUBO for both modes).
    if "real_quantum_hardware" in qubo_src.columns:
        sim = qubo_src[qubo_src["real_quantum_hardware"].fillna(0) < 0.5]
        if not sim.empty:
            qubo_src = sim

    qubo_vars = avg_by_requests(qubo_src, "base_qubo_vars")
    qubo_coup = avg_by_requests(qubo_src, "base_qubo_couplers")
    ilp_int = avg_by_requests(d_class, "ilp_num_integer_vars")
    ilp_nnz = avg_by_requests(d_class, "ilp_num_nonzero_coeffs")
    greedy_work = avg_by_requests(d_greedy, "greedy_sort_work")

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.set_title("Decision Variable Count vs. Problem Size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Variable / Work Unit Count")
    ax.grid(alpha=0.25)

    rows = []
    for label, frame, col, marker in [
        ("QUBO vars", qubo_vars, "base_qubo_vars", "o"),
        ("ILP integer vars", ilp_int, "ilp_num_integer_vars", "s"),
        ("Greedy sort work", greedy_work, "greedy_sort_work", "^"),
    ]:
        if frame.empty:
            continue
        x = frame["num_requests"].to_numpy()
        y = frame[col].to_numpy()
        a, C, r2 = fit_powerlaw(x, y)
        ax.scatter(x, y, marker=marker, label=label)
        if np.isfinite(a):
            xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            ax.plot(xfit, C * (xfit**a), label=f"{label} fit ~ n^{a:.2f} (R²={r2:.3f})")
            rows.append({"series": label, "exponent": a, "C": C, "R2": r2})
    ax.legend(fontsize=8)
    savefig(figures_dir / "01_decision_variable_count.png")

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.set_title("Interaction Density Scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    for label, frame, col in [
        ("QUBO couplers", qubo_coup, "base_qubo_couplers"),
        ("ILP nonzeros", ilp_nnz, "ilp_num_nonzero_coeffs"),
    ]:
        if frame.empty:
            continue
        x = frame["num_requests"].to_numpy()
        y = frame[col].to_numpy()
        a, C, r2 = fit_powerlaw(x, y)
        ax.scatter(x, y, label=label)
        if np.isfinite(a):
            xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            ax.plot(xfit, C * (xfit**a), label=f"{label} fit ~ n^{a:.2f} (R²={r2:.3f})")
            rows.append({"series": label, "exponent": a, "C": C, "R2": r2})
    ax.legend(fontsize=8)
    savefig(figures_dir / "02_interaction_density.png")

    if rows:
        pd.DataFrame(rows).to_csv(tables_dir / "scaling_exponents_structure.csv", index=False)


def plot_end_to_end_runtime(d_class, d_greedy, d_quant_sim, d_quant_real, figures_dir, tables_dir):
    print("\n[2/8] End-to-end runtime scaling")
    plt.figure(figsize=(8, 6))
    plt.title("End-to-End Runtime vs. Problem Size")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Requests")
    plt.ylabel("Runtime (s)")
    plt.grid(alpha=0.25)

    fit_rows = []
    series = [
        ("Classical ILP", d_class, ["total_run_time", "solve_time"]),
        ("ClassicalGreedy", d_greedy, ["time_greedy_total", "total_run_time"]),
        ("Quantum (simulator)", d_quant_sim, ["total_run_time", "solve_time"]),
        ("Quantum (real hardware)", d_quant_real, ["total_run_time", "solve_time"]),
    ]
    for label, frame, cols in series:
        if frame.empty:
            continue
        col = first_present(frame, cols)
        if col is None:
            continue
        avg = avg_by_requests(frame, col)
        if avg.empty:
            continue
        x = avg["num_requests"].to_numpy()
        y = avg[col].to_numpy()
        plt.scatter(x, y, label=f"{label} ({col})")
        a, C, r2 = fit_powerlaw(x, y)
        if np.isfinite(a):
            xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            plt.plot(xfit, C * (xfit**a), label=f"{label} fit ~ n^{a:.2f} (R²={r2:.3f})")
            fit_rows.append({"method": label, "column": col, "exponent": a, "C": C, "R2": r2})

    # Modeled D-Wave end-to-end (simulator overhead + modeled QPU) when simulator data exists
    if not d_quant_sim.empty:
        q = d_quant_sim.copy()
        qpu = dwave_qpu_time_seconds()
        overhead = (
            safe_series(q, "rtv_graph_build_time")
            + safe_series(q, "time_min_cost_prep")
            + safe_series(q, "time_qubo_gen")
            + safe_series(q, "qubo_build_time")
            + safe_series(q, "time_compress")
            + safe_series(q, "time_decode")
            + safe_series(q, "time_vehicle_assignment")
            + safe_series(q, "time_metrics_calc")
            + safe_series(q, "time_struct_stats")
        )
        q["modeled_dwave_e2e"] = overhead + qpu
        avg = avg_by_requests(q, "modeled_dwave_e2e")
        if not avg.empty:
            x = avg["num_requests"].to_numpy()
            y = avg["modeled_dwave_e2e"].to_numpy()
            plt.scatter(x, y, marker="D", label="Quantum (modeled D-Wave e2e)")
            a, C, r2 = fit_powerlaw(x, y)
            if np.isfinite(a):
                xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
                plt.plot(
                    xfit,
                    C * (xfit**a),
                    linestyle="--",
                    label=f"Modeled D-Wave fit ~ n^{a:.2f} (R²={r2:.3f})",
                )
                fit_rows.append(
                    {
                        "method": "Quantum (modeled D-Wave e2e)",
                        "column": "modeled_dwave_e2e",
                        "exponent": a,
                        "C": C,
                        "R2": r2,
                    }
                )

    plt.legend(fontsize=7)
    savefig(figures_dir / "03_end_to_end_runtime.png")
    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(tables_dir / "scaling_exponents_runtime.csv", index=False)


def plot_build_and_pipeline_times(df, d_quant, figures_dir, tables_dir):
    print("\n[3/8] RTV / QUBO build / decode / output pipeline times")
    # RTV
    if "rtv_graph_build_time" in df.columns:
        avg = avg_by_requests(df, "rtv_graph_build_time")
        plt.figure(figsize=(8, 6))
        plt.title("RTV Graph Construction Time vs. Problem Size")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of Requests")
        plt.ylabel("Graph Build Time (s)")
        plt.grid(alpha=0.25)
        x = avg["num_requests"].to_numpy()
        y = avg["rtv_graph_build_time"].to_numpy()
        if len(x):
            a, C, r2 = fit_powerlaw(x, y)
            plt.scatter(x, y, label="Combined RTV build")
            if np.isfinite(a):
                xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
                plt.plot(xfit, C * (xfit**a), label=f"fit ~ n^{a:.2f} (R²={r2:.3f})")
        plt.legend()
        savefig(figures_dir / "04_rtv_graph_build_time.png")

    # QUBO build (quantum rows only)
    if not d_quant.empty and "qubo_build_time" in d_quant.columns:
        qsrc = d_quant[d_quant["real_quantum_hardware"].fillna(0) < 0.5]
        if qsrc.empty:
            qsrc = d_quant
        avg = avg_by_requests(qsrc, "qubo_build_time")
        plt.figure(figsize=(8, 6))
        plt.title("QUBO Construction Time vs. Problem Size")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of Requests")
        plt.ylabel("QUBO Construction Time (s)")
        plt.grid(alpha=0.25)
        x = avg["num_requests"].to_numpy()
        y = avg["qubo_build_time"].to_numpy()
        if len(x):
            a, C, r2 = fit_powerlaw(x, y)
            plt.scatter(x, y, label="QUBO build")
            if np.isfinite(a):
                xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
                plt.plot(xfit, C * (xfit**a), label=f"fit ~ n^{a:.2f} (R²={r2:.3f})")
        plt.legend()
        savefig(figures_dir / "05_qubo_build_time.png")

    # Decode + vehicle assignment
    if not d_quant.empty:
        q = d_quant.copy()
        if "real_quantum_hardware" in q.columns:
            sim = q[q["real_quantum_hardware"].fillna(0) < 0.5]
            if not sim.empty:
                q = sim
        q["decode_pipeline"] = safe_series(q, "time_decode") + safe_series(q, "time_vehicle_assignment")
        q["output_pipeline"] = (
            safe_series(q, "time_decode")
            + safe_series(q, "time_vehicle_assignment")
            + safe_series(q, "time_metrics_calc")
            + safe_series(q, "time_struct_stats")
        )
        for col, title, fname in [
            ("decode_pipeline", "Decode + Vehicle Assignment Time", "06_decode_vehicle_assignment.png"),
            ("output_pipeline", "Quantum Output / Postprocessing Time", "07_quantum_output_pipeline.png"),
        ]:
            avg = avg_by_requests(q, col)
            if avg.empty:
                continue
            plt.figure(figsize=(8, 6))
            plt.title(f"{title} vs. Problem Size")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Number of Requests")
            plt.ylabel("Time (s)")
            plt.grid(alpha=0.25)
            x = avg["num_requests"].to_numpy()
            y = avg[col].to_numpy()
            a, C, r2 = fit_powerlaw(x, y)
            plt.scatter(x, y, label=col)
            if np.isfinite(a):
                xfit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
                plt.plot(xfit, C * (xfit**a), label=f"fit ~ n^{a:.2f} (R²={r2:.3f})")
            plt.legend()
            savefig(figures_dir / fname)


def plot_quantum_runtime_breakdown(d_quant_sim, figures_dir, tables_dir):
    print("\n[4/8] Quantum runtime breakdown")
    if d_quant_sim.empty:
        print("  (no simulator quantum rows; skipping)")
        return

    q = d_quant_sim.copy()
    q["dwave_qpu_time"] = dwave_qpu_time_seconds()
    q["prep_build_time"] = (
        safe_series(q, "rtv_graph_build_time")
        + safe_series(q, "time_min_cost_prep")
        + safe_series(q, "time_qubo_gen")
        + safe_series(q, "qubo_build_time")
        + safe_series(q, "time_compress")
    )
    q["output_time"] = (
        safe_series(q, "time_decode")
        + safe_series(q, "time_vehicle_assignment")
        + safe_series(q, "time_metrics_calc")
        + safe_series(q, "time_struct_stats")
    )
    q["quantum_end_to_end"] = q["prep_build_time"] + q["dwave_qpu_time"] + q["output_time"]

    avg = (
        q.groupby("num_requests", as_index=False)[
            ["prep_build_time", "dwave_qpu_time", "output_time", "quantum_end_to_end"]
        ]
        .mean()
        .sort_values("num_requests")
    )
    avg["output_fraction"] = avg["output_time"] / avg["quantum_end_to_end"]
    avg.to_csv(tables_dir / "quantum_runtime_breakdown.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.stackplot(
        avg["num_requests"].to_numpy(),
        avg["prep_build_time"].to_numpy(),
        avg["dwave_qpu_time"].to_numpy(),
        avg["output_time"].to_numpy(),
        labels=["Prep + QUBO build", "Modeled D-Wave solve", "Output/postprocessing"],
        alpha=0.85,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Requests")
    plt.ylabel("Runtime (s)")
    plt.title("Quantum End-to-End Runtime Breakdown vs. Problem Size")
    plt.legend(loc="upper left")
    savefig(figures_dir / "08_quantum_runtime_breakdown.png")

    plt.figure(figsize=(8, 5))
    plt.plot(avg["num_requests"], avg["output_fraction"], marker="o")
    plt.xscale("log")
    plt.xlabel("Number of Requests")
    plt.ylabel("Output fraction of total quantum runtime")
    plt.title("Contribution of output stage to quantum end-to-end runtime")
    plt.grid(alpha=0.3)
    savefig(figures_dir / "09_quantum_output_fraction.png")


def plot_service_quality(d_class, d_greedy, d_quant_sim, d_quant_real, figures_dir, tables_dir):
    print("\n[5/8] Service quality (% serviced, waiting, violations)")
    series = [
        ("Classical", d_class),
        ("ClassicalGreedy", d_greedy),
        ("Quantum (simulator)", d_quant_sim),
        ("Quantum (real hardware)", d_quant_real),
    ]

    # Summary table of means by requests
    summary_parts = []
    for name, frame in series:
        if frame.empty or "percent_serviced" not in frame.columns:
            continue
        g = (
            frame.groupby("num_requests", as_index=False)
            .agg(
                percent_serviced=("percent_serviced", "mean"),
                avg_waiting_time=("avg_waiting_time", "mean"),
                vmt=("vmt", "mean"),
                violation_rate=("violation_rate", "mean")
                if "violation_rate" in frame.columns
                else ("percent_serviced", "count"),
                n=("percent_serviced", "count"),
            )
        )
        g["method"] = name
        summary_parts.append(g)
    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
        summary.to_csv(tables_dir / "service_quality_by_requests.csv", index=False)
        # Also pivot for readability
        piv = summary.pivot(index="num_requests", columns="method", values="percent_serviced")
        piv.to_csv(tables_dir / "percent_serviced_pivot.csv")

    # % serviced with SEM
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_ylim(0, 110)
    ax.set_title("Requests vs % Serviced (mean ± SEM)")
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("% Serviced")
    ax.grid(alpha=0.25)
    for name, frame in series:
        if frame.empty or "percent_serviced" not in frame.columns:
            continue
        data = avg_by_requests_sem(frame, "percent_serviced")
        ax.errorbar(
            data["num_requests"],
            data["percent_serviced_mean"],
            yerr=data["percent_serviced_sem"],
            fmt="-o",
            capsize=3,
            label=name,
        )
    ax.legend(fontsize=8)
    savefig(figures_dir / "10_percent_serviced.png")

    # Waiting time
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_title("Requests vs Mean Waiting Time (mean ± SEM)")
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Mean Waiting Time (s)")
    ax.grid(alpha=0.25)
    for name, frame in series:
        if frame.empty or "avg_waiting_time" not in frame.columns:
            continue
        data = avg_by_requests_sem(frame, "avg_waiting_time")
        ax.errorbar(
            data["num_requests"],
            data["avg_waiting_time_mean"],
            yerr=data["avg_waiting_time_sem"],
            fmt="-o",
            capsize=3,
            label=name,
        )
    ax.legend(fontsize=8)
    savefig(figures_dir / "11_avg_waiting_time.png")

    # VMT
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_title("Requests vs VMT (mean ± SEM)")
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("VMT")
    ax.grid(alpha=0.25)
    for name, frame in series:
        if frame.empty or "vmt" not in frame.columns:
            continue
        data = avg_by_requests_sem(frame, "vmt")
        ax.errorbar(
            data["num_requests"],
            data["vmt_mean"],
            yerr=data["vmt_sem"],
            fmt="-o",
            capsize=3,
            label=name,
        )
    ax.legend(fontsize=8)
    savefig(figures_dir / "12_vmt.png")

    # Violation rate
    if any("violation_rate" in frame.columns and not frame.empty for _, frame in series):
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.set_title("Requests vs Constraint Violation Rate (mean ± SEM)")
        ax.set_xlabel("Number of Requests")
        ax.set_ylabel("Violation Rate")
        ax.grid(alpha=0.25)
        for name, frame in series:
            if frame.empty or "violation_rate" not in frame.columns:
                continue
            data = avg_by_requests_sem(frame, "violation_rate")
            ax.errorbar(
                data["num_requests"],
                data["violation_rate_mean"],
                yerr=data["violation_rate_sem"],
                fmt="-o",
                capsize=3,
                label=name,
            )
        ax.legend(fontsize=8)
        savefig(figures_dir / "13_violation_rate.png")


def plot_accuracy_line_comparison(df, figures_dir, tables_dir):
    print("\n[6/8] Accuracy line comparison (all methods)")
    means = (
        df.groupby(["num_requests", "method"], as_index=False)["percent_serviced"]
        .mean()
        .sort_values(["num_requests", "method"])
    )
    means.to_csv(tables_dir / "accuracy_by_method.csv", index=False)

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=means, x="num_requests", y="percent_serviced", hue="method", marker="o")
    plt.title("Accuracy Comparison Across Solvers")
    plt.ylabel("Average Percent Serviced (%)")
    plt.xlabel("Number of Requests")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    savefig(figures_dir / "14_accuracy_all_methods.png")


def paired_statistical_tests(d_class, d_greedy, d_quant_sim, d_quant_real, tables_dir):
    print("\n[7/8] Paired statistical tests")
    instance_cols = [c for c in ["city", "num_requests", "num_vehicles", "seed", "trial"] if c]
    # keep only columns that exist in all used frames
    frames = [d for d in (d_class, d_greedy, d_quant_sim, d_quant_real) if not d.empty]
    if not frames:
        return
    instance_cols = [c for c in instance_cols if all(c in f.columns for f in frames)]
    if not instance_cols:
        print("  (no shared instance columns; skipping)")
        return

    metrics = [m for m in ["percent_serviced", "avg_waiting_time", "vmt", "violation_rate"]]
    results = []

    def run_pair(label_a, d_a, label_b, d_b, metric):
        if d_a.empty or d_b.empty or metric not in d_a.columns or metric not in d_b.columns:
            return
        merged = pd.merge(
            d_a[instance_cols + [metric]],
            d_b[instance_cols + [metric]],
            on=instance_cols,
            suffixes=("_a", "_b"),
        ).dropna()
        if len(merged) < 5:
            results.append(
                {
                    "metric": metric,
                    "comparison": f"{label_b} - {label_a}",
                    "n": len(merged),
                    "note": "not enough paired samples",
                }
            )
            return
        x = merged[f"{metric}_a"].to_numpy()
        y = merged[f"{metric}_b"].to_numpy()
        diff = y - x
        t_stat, p_val = stats.ttest_rel(y, x)
        try:
            _, p_w = stats.wilcoxon(diff)
        except Exception:
            p_w = np.nan
        d_cohen = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan
        ci = stats.t.interval(0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff))
        results.append(
            {
                "metric": metric,
                "comparison": f"{label_b} - {label_a}",
                "n": len(diff),
                "mean_diff": float(np.mean(diff)),
                "paired_t_pvalue": float(p_val),
                "wilcoxon_pvalue": float(p_w) if pd.notna(p_w) else np.nan,
                "cohens_d": float(d_cohen) if pd.notna(d_cohen) else np.nan,
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
            }
        )

    pairs = [
        ("Classical", d_class, "ClassicalGreedy", d_greedy),
        ("Classical", d_class, "Quantum (simulator)", d_quant_sim),
        ("ClassicalGreedy", d_greedy, "Quantum (simulator)", d_quant_sim),
        ("Quantum (simulator)", d_quant_sim, "Quantum (real hardware)", d_quant_real),
        ("Classical", d_class, "Quantum (real hardware)", d_quant_real),
    ]
    for la, da, lb, db in pairs:
        for metric in metrics:
            if metric == "violation_rate" and (
                metric not in da.columns or metric not in db.columns
            ):
                continue
            run_pair(la, da, lb, db, metric)

    if results:
        out = pd.DataFrame(results)
        out.to_csv(tables_dir / "paired_statistical_tests.csv", index=False)
        print(out.to_string(index=False))


def hyperparameter_and_real_sim_summary(d_quant, d_quant_sim, d_quant_real, tables_dir, figures_dir):
    print("\n[8/8] Quantum hyperparams + real vs simulator summary")
    if not d_quant.empty and {"lambda_val", "M_val", "percent_serviced"}.issubset(d_quant.columns):
        opt = (
            d_quant.groupby(["lambda_val", "M_val"], as_index=False)["percent_serviced"]
            .mean()
            .sort_values("percent_serviced", ascending=False)
        )
        opt.to_csv(tables_dir / "quantum_hyperparameter_ranking.csv", index=False)
        if not opt.empty:
            best = opt.iloc[0]
            print(
                f"  Best config: λ={best['lambda_val']}, M={best['M_val']}, "
                f"avg % serviced={best['percent_serviced']:.2f}"
            )

    # Direct real vs sim quality comparison on overlapping instances
    keys = [c for c in ["city", "num_vehicles", "num_requests", "seed", "trial"] if c]
    if d_quant_sim.empty or d_quant_real.empty:
        print("  (insufficient real/sim rows for paired comparison)")
        return
    keys = [c for c in keys if c in d_quant_sim.columns and c in d_quant_real.columns]
    metrics = [m for m in ["percent_serviced", "avg_waiting_time", "avg_detour_factor", "vmt"] if m]
    merged = pd.merge(
        d_quant_sim[keys + metrics],
        d_quant_real[keys + metrics],
        on=keys,
        suffixes=("_sim", "_real"),
    )
    if merged.empty:
        print("  (no overlapping real/sim instances)")
        return
    merged.to_csv(tables_dir / "quantum_real_vs_sim_paired.csv", index=False)

    summary = []
    for m in metrics:
        summary.append(
            {
                "metric": m,
                "n_pairs": len(merged),
                "mean_sim": float(merged[f"{m}_sim"].mean()),
                "mean_real": float(merged[f"{m}_real"].mean()),
                "mean_diff_real_minus_sim": float(
                    (merged[f"{m}_real"] - merged[f"{m}_sim"]).mean()
                ),
            }
        )
    pd.DataFrame(summary).to_csv(tables_dir / "quantum_real_vs_sim_summary.csv", index=False)

    # Bar chart of mean % serviced
    plt.figure(figsize=(6, 5))
    means = [
        merged["percent_serviced_sim"].mean(),
        merged["percent_serviced_real"].mean(),
    ]
    plt.bar(["Simulator", "Real hardware"], means, color=["#4C78A8", "#F58518"])
    plt.ylabel("Mean % Serviced (paired instances)")
    plt.title(f"Quantum Real vs Simulator (% serviced)\nn={len(merged)} paired runs")
    plt.ylim(0, 110)
    for i, v in enumerate(means):
        plt.text(i, v + 1, f"{v:.1f}%", ha="center")
    savefig(figures_dir / "15_quantum_real_vs_sim_percent_serviced.png")


MAX_PLAUSIBLE_DETOUR = 50.0  # a detour factor above this is a corrupted row


def filter_corrupted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with physically impossible metric values.

    A few Classical rows in the raw sweep carry corrupted avg/max detour and VMT
    (e.g. avg_detour_factor in the tens of thousands). These are data-recording
    glitches, not real results; left in, they destroy any Classical VMT/detour
    mean. We null out the offending metric cells (not the whole row, so the row's
    valid metrics like percent_serviced survive).
    """
    df = df.copy()
    n_bad = 0
    bad_mask = pd.Series(False, index=df.index)
    for col in ("avg_detour_factor", "max_detour_factor"):
        if col in df.columns:
            bad = pd.to_numeric(df[col], errors="coerce") > MAX_PLAUSIBLE_DETOUR
            bad_mask = bad_mask | bad
            n_bad += int(bad.sum())
            df.loc[bad, col] = np.nan
    # VMT co-corrupts with detour on the same rows; null VMT where detour was absurd.
    if "vmt" in df.columns and bad_mask.any():
        df.loc[bad_mask, "vmt"] = np.nan
    if n_bad:
        print(f"  [robustness] nulled {n_bad} corrupted detour cells (>{MAX_PLAUSIBLE_DETOUR})")
    return df


def variance_table(df: pd.DataFrame, tables_dir: Path):
    """Numeric spread table per method x size for the key quality metrics.

    Reports n, mean, std, sem, min, max, and coefficient of variation so the
    paper can state error bars as numbers, and show that the quantum solver's
    run-to-run spread is small (stability across seeds), not just plot whiskers.
    """
    print("\n[+] Variance / stability table")
    metrics = [m for m in ["percent_serviced", "avg_waiting_time", "vmt",
                           "avg_detour_factor"] if m in df.columns]
    rows = []
    for (method, n), sub in df.groupby(["method", "num_requests"]):
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            sem = std / np.sqrt(len(vals)) if len(vals) > 0 else 0.0
            cv = (std / mean) if mean not in (0, np.nan) else np.nan
            rows.append({
                "method": method, "num_requests": int(n), "metric": m,
                "n": len(vals), "mean": round(mean, 4), "std": round(std, 4),
                "sem": round(sem, 4), "cv": round(cv, 4) if pd.notna(cv) else np.nan,
                "min": round(vals.min(), 4), "max": round(vals.max(), 4),
            })
    if rows:
        out = pd.DataFrame(rows).sort_values(["metric", "num_requests", "method"])
        out.to_csv(tables_dir / "variance_by_method_size.csv", index=False)
        # Compact headline: quantum % serviced spread at each size.
        q = out[(out["method"].str.contains("simulator")) &
                (out["metric"] == "percent_serviced")]
        if not q.empty:
            print("  Quantum (sim) % serviced spread by size:")
            for _, r in q.iterrows():
                print(f"    n={r['num_requests']:>4}: {r['mean']:.1f} ± {r['sem']:.1f} "
                      f"(std {r['std']:.1f}, cv {r['cv']:.3f}, n={r['n']})")
        print(f"  saved {tables_dir / 'variance_by_method_size.csv'}")


def decision_space_breakdown(d_quant, d_class, d_greedy, figures_dir, tables_dir):
    """Per-request decision-space scaling: variables and couplers with power-law
    exponents, reported as a clean table plus a two-panel figure. Strengthens the
    existing plot by (a) using simulator rows for QUBO structure, (b) reporting
    vars-per-request and couplers-per-request to expose the linear-in-n structure,
    and (c) emitting the exponents as a paper table.
    """
    print("\n[+] Decision-space breakdown (vars/couplers scaling)")
    q = d_quant.copy()
    if "real_quantum_hardware" in q.columns:
        sim = q[q["real_quantum_hardware"].fillna(0) < 0.5]
        if not sim.empty:
            q = sim
    if q.empty:
        print("  (no quantum rows)"); return

    vars_by = avg_by_requests(q, "base_qubo_vars")
    coup_by = avg_by_requests(q, "base_qubo_couplers")
    ilp_by = avg_by_requests(d_class, "ilp_num_integer_vars")

    rows = []
    for label, frame, col in [("QUBO vars", vars_by, "base_qubo_vars"),
                              ("QUBO couplers", coup_by, "base_qubo_couplers"),
                              ("ILP integer vars", ilp_by, "ilp_num_integer_vars")]:
        if frame.empty:
            continue
        x = frame["num_requests"].to_numpy()
        y = frame[col].to_numpy()
        a, C, r2 = fit_powerlaw(x, y)
        rows.append({"series": label, "exponent": round(a, 3),
                     "prefactor": round(C, 3), "R2": round(r2, 4)})
    if rows:
        pd.DataFrame(rows).to_csv(tables_dir / "decision_space_exponents.csv", index=False)
        for r in rows:
            print(f"    {r['series']:20s} ~ n^{r['exponent']} (R²={r['R2']})")

    # per-request normalized view (exposes O(n) vars, ~O(n) couplers under the cap)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if not vars_by.empty:
        ax1.plot(vars_by["num_requests"],
                 vars_by["base_qubo_vars"] / vars_by["num_requests"],
                 "o-", label="QUBO vars / request")
    if not coup_by.empty:
        ax1.plot(coup_by["num_requests"],
                 coup_by["base_qubo_couplers"] / coup_by["num_requests"],
                 "s-", label="QUBO couplers / request")
    ax1.set_xlabel("Number of Requests"); ax1.set_ylabel("Count per request")
    ax1.set_title("(a) Per-request decision-space (bounded growth)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_xscale("log"); ax2.set_yscale("log")
    if not vars_by.empty:
        ax2.scatter(vars_by["num_requests"], vars_by["base_qubo_vars"], label="QUBO vars")
    if not coup_by.empty:
        ax2.scatter(coup_by["num_requests"], coup_by["base_qubo_couplers"], label="QUBO couplers")
    if not ilp_by.empty:
        ax2.scatter(ilp_by["num_requests"], ilp_by["ilp_num_integer_vars"], label="ILP int vars", marker="^")
    ax2.set_xlabel("Number of Requests"); ax2.set_ylabel("Count")
    ax2.set_title("(b) Decision-space scaling (log-log)")
    ax2.legend(); ax2.grid(alpha=0.3, which="both")
    plt.tight_layout()
    savefig(figures_dir / "16_decision_space_breakdown.png")


def qaoa_vs_annealing(df, figures_dir, tables_dir):
    """HONEST QAOA (real hardware) vs annealing (simulator) comparison.

    Real QAOA runs only exist at small sizes (here n=5). We therefore compare the
    two solvers ONLY on the sizes where BOTH ran, paired by instance, and label
    it explicitly as a small-instance agreement check -- NOT a claim that QAOA
    was run at scale. This is the correct framing: annealing is the main quality
    solver; QAOA is a small-instance compatibility check that agrees with it.
    """
    print("\n[+] QAOA (real) vs annealing (sim) -- small-instance agreement check")
    q = df[df["run_type"] == "Quantum"].copy()
    real = q[q["real_quantum_hardware"].fillna(0) >= 0.5]
    sim = q[q["real_quantum_hardware"].fillna(0) < 0.5]
    if real.empty:
        print("  (no real-hardware QAOA rows; skipping)"); return

    overlap_sizes = sorted(set(real["num_requests"]).intersection(sim["num_requests"]))
    if not overlap_sizes:
        print("  (no size overlap between QAOA and annealing; skipping)"); return
    print(f"  QAOA ran at sizes: {sorted(real['num_requests'].unique())}")
    print(f"  Overlap with annealing: {overlap_sizes}  (comparison scoped to these)")

    keys = [c for c in ["city", "num_vehicles", "num_requests", "seed"] if c in real.columns and c in sim.columns]
    metrics = [m for m in ["percent_serviced", "avg_waiting_time", "avg_detour_factor", "vmt"] if m in q.columns]
    # collapse sim duplicates (e.g. multiple trials) to per-instance mean first
    sim_small = (sim[sim["num_requests"].isin(overlap_sizes)]
                 .groupby(keys, as_index=False)[metrics].mean())
    real_small = (real[real["num_requests"].isin(overlap_sizes)]
                  .groupby(keys, as_index=False)[metrics].mean())
    merged = pd.merge(sim_small, real_small, on=keys, suffixes=("_anneal", "_qaoa"))
    if merged.empty:
        print("  (no paired QAOA/annealing instances)"); return
    merged.to_csv(tables_dir / "qaoa_vs_annealing_paired.csv", index=False)

    summary = []
    for m in metrics:
        a = merged[f"{m}_anneal"]; r = merged[f"{m}_qaoa"]
        summary.append({
            "metric": m, "n_pairs": len(merged),
            "mean_annealing": round(a.mean(), 3),
            "mean_qaoa": round(r.mean(), 3),
            "mean_abs_diff": round((r - a).abs().mean(), 3),
            "max_abs_diff": round((r - a).abs().max(), 3),
        })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(tables_dir / "qaoa_vs_annealing_summary.csv", index=False)
    print(sdf.to_string(index=False))
    print("  Interpretation: small mean_abs_diff => annealing reproduces QAOA's "
          "solution quality on the instances QAOA can handle, supporting annealing "
          "as the scalable stand-in. Report ONLY for sizes in the overlap above.")

    # simple paired bar for % serviced
    if "percent_serviced_anneal" in merged.columns:
        plt.figure(figsize=(6, 5))
        means = [merged["percent_serviced_anneal"].mean(), merged["percent_serviced_qaoa"].mean()]
        plt.bar(["Annealing (sim)", "QAOA (real HW)"], means, color=["#4C78A8", "#F58518"])
        plt.ylabel("Mean % Serviced (paired, overlap sizes only)")
        plt.title(f"QAOA vs Annealing agreement\nn={len(merged)} paired instances, "
                  f"sizes={overlap_sizes}")
        plt.ylim(0, 110)
        for i, v in enumerate(means):
            plt.text(i, v + 1, f"{v:.1f}%", ha="center")
        savefig(figures_dir / "17_qaoa_vs_annealing.png")


def write_overview(df, tables_dir):
    counts = (
        df.groupby(["run_type", "method"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["run_type", "method"])
    )
    counts.to_csv(tables_dir / "run_counts.csv", index=False)
    overview = {
        "n_rows": len(df),
        "n_classical": int((df["run_type"] == "Classical").sum()),
        "n_greedy": int((df["run_type"] == "ClassicalGreedy").sum()),
        "n_quantum_sim": int(
            ((df["run_type"] == "Quantum") & (df.get("real_quantum_hardware", 0).fillna(0) < 0.5)).sum()
        ),
        "n_quantum_real": int(
            ((df["run_type"] == "Quantum") & (df.get("real_quantum_hardware", 0).fillna(0) >= 0.5)).sum()
        ),
        "request_sizes": sorted(df["num_requests"].dropna().unique().tolist()),
        "vehicle_sizes": sorted(df["num_vehicles"].dropna().unique().tolist())
        if "num_vehicles" in df.columns
        else [],
    }
    pd.Series(overview).to_json(tables_dir / "overview.json")
    print("\n=== Overview ===")
    print(counts.to_string(index=False))


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "analysis_figures"
    tables_dir = out_dir / "analysis_tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    print(f"Loading {csv_path}")
    df = load_and_prepare(csv_path)
    df = filter_corrupted_rows(df)
    d_class, d_greedy, d_quant, d_quant_sim, d_quant_real = split_solvers(df)
    write_overview(df, tables_dir)

    sns.set_theme(style="whitegrid")
    plot_decision_dimensions(d_class, d_greedy, d_quant, figures_dir, tables_dir)
    plot_end_to_end_runtime(d_class, d_greedy, d_quant_sim, d_quant_real, figures_dir, tables_dir)
    plot_build_and_pipeline_times(df, d_quant, figures_dir, tables_dir)
    plot_quantum_runtime_breakdown(d_quant_sim, figures_dir, tables_dir)
    plot_service_quality(d_class, d_greedy, d_quant_sim, d_quant_real, figures_dir, tables_dir)
    plot_accuracy_line_comparison(df, figures_dir, tables_dir)
    paired_statistical_tests(d_class, d_greedy, d_quant_sim, d_quant_real, tables_dir)
    hyperparameter_and_real_sim_summary(
        d_quant, d_quant_sim, d_quant_real, tables_dir, figures_dir
    )
    variance_table(df, tables_dir)
    decision_space_breakdown(d_quant, d_class, d_greedy, figures_dir, tables_dir)
    qaoa_vs_annealing(df, figures_dir, tables_dir)
    infeasibility_analysis(df, figures_dir, tables_dir)
    slack_drop_analysis(csv_path, args.per_request_csv, figures_dir, tables_dir)

    print(f"\nDone.")
    print(f"  Figures: {figures_dir}")
    print(f"  Tables:  {tables_dir}")


def infeasibility_analysis(df, figures_dir, tables_dir):
    """How often is the RAW QUBO output infeasible before the greedy+Hungarian
    cleanup fixes it, and how much work is the cleanup doing?

    Reads the raw_* columns emitted by the instrumented solver (quantum rows).
    Reports, per size: fraction of instances that were infeasible (>=1 dropped
    trip), mean trip-level violation rate, and mean dropped-trip count. This
    quantifies the classical post-processing's contribution -- the reviewer's
    "how much is the QUBO actually solving vs. the cleanup" question.
    """
    print("\n[+] Raw-QUBO infeasibility (pre-cleanup) analysis")
    need = {"raw_violation_rate", "raw_infeasible_instance", "raw_dropped_trips"}
    q = df[df["run_type"] == "Quantum"].copy()
    if q.empty or not need.issubset(q.columns):
        print("  (no raw_* infeasibility columns; run the instrumented sweep first)")
        return
    for c in ("raw_violation_rate", "raw_infeasible_instance", "raw_dropped_trips",
              "raw_selected_trips", "raw_kept_trips"):
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")
    # drop rows where instrumentation was absent (old rows) -> NaN
    q = q[q["raw_violation_rate"].notna()]
    if q.empty:
        print("  (raw_* columns present but empty; nothing to analyze)")
        return

    g = (q.groupby("num_requests", as_index=False)
           .agg(n=("raw_violation_rate", "count"),
                infeasible_instance_rate=("raw_infeasible_instance", "mean"),
                mean_violation_rate=("raw_violation_rate", "mean"),
                sem_violation_rate=("raw_violation_rate", "sem"),
                mean_dropped=("raw_dropped_trips", "mean"),
                mean_selected=("raw_selected_trips", "mean"),
                mean_kept=("raw_kept_trips", "mean"))
           .sort_values("num_requests"))
    g.to_csv(tables_dir / "raw_infeasibility_by_size.csv", index=False)
    print(g.to_string(index=False))
    print("  Interpretation: a high infeasible_instance_rate / mean_violation_rate "
          "means the RAW QUBO rarely returns a conflict-free set and the classical "
          "cleanup is doing substantial work -- report this honestly as a limitation.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.errorbar(g["num_requests"], g["mean_violation_rate"],
                 yerr=g["sem_violation_rate"], fmt="-o", capsize=3, color="#d62728")
    ax1.set_xlabel("Number of Requests")
    ax1.set_ylabel("Mean raw violation rate (dropped / selected)")
    ax1.set_title("(a) Raw QUBO infeasibility before cleanup")
    ax1.set_ylim(0, 1); ax1.grid(alpha=0.3)

    ax2.plot(g["num_requests"], g["mean_selected"], "s-", label="raw selected")
    ax2.plot(g["num_requests"], g["mean_kept"], "o-", label="kept after cleanup")
    ax2.set_xlabel("Number of Requests"); ax2.set_ylabel("Mean trip count")
    ax2.set_title("(b) Trips selected vs. kept (gap = cleanup work)")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    savefig(figures_dir / "18_raw_infeasibility.png")


def _find_per_request_csv(main_csv: Path, explicit) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    cand = main_csv.with_name(main_csv.stem + "_per_request.csv")
    return cand if cand.exists() else None


def slack_drop_analysis(main_csv: Path, per_request_csv, figures_dir, tables_dir):
    """Do DROPPED requests have systematically tighter time windows (lower slack)
    or higher detour than served ones? Reads the per-request side file.

    Slack = latest_pickup - request_time (feasible-window width). We compare the
    slack distribution of served vs. dropped requests, both overall and per size,
    with a Mann-Whitney U test (nonparametric; served/dropped are unpaired).
    """
    print("\n[+] Which requests get dropped? (slack / detour analysis)")
    pr_path = _find_per_request_csv(main_csv, per_request_csv)
    if pr_path is None:
        print("  (no per-request side file found; expected "
              f"{main_csv.stem}_per_request.csv next to the results CSV)")
        return
    pr = pd.read_csv(pr_path)
    for c in ("served", "slack", "detour_factor", "num_requests"):
        if c in pr.columns:
            pr[c] = pd.to_numeric(pr[c], errors="coerce")
    pr = pr[pr["slack"].notna() & pr["served"].notna()]
    if pr.empty:
        print("  (per-request file has no usable slack/served rows)")
        return

    served = pr[pr["served"] == 1]["slack"]
    dropped = pr[pr["served"] == 0]["slack"]
    print(f"  loaded {len(pr)} per-request rows "
          f"({len(served)} served, {len(dropped)} dropped)")
    if len(dropped) == 0:
        print("  (no dropped requests in this data -- nothing to compare)")
        return

    # Overall served-vs-dropped slack, with a nonparametric test.
    rows = [{
        "scope": "overall", "num_requests": "all",
        "n_served": len(served), "n_dropped": len(dropped),
        "mean_slack_served": round(served.mean(), 2),
        "mean_slack_dropped": round(dropped.mean(), 2),
        "median_slack_served": round(served.median(), 2),
        "median_slack_dropped": round(dropped.median(), 2),
    }]
    try:
        u, p = stats.mannwhitneyu(dropped, served, alternative="less")
        rows[0]["mannwhitney_p_dropped_lower"] = float(p)
    except Exception:
        rows[0]["mannwhitney_p_dropped_lower"] = np.nan

    # Per-size breakdown.
    for n, sub in pr.groupby("num_requests"):
        s = sub[sub["served"] == 1]["slack"]
        d = sub[sub["served"] == 0]["slack"]
        if len(d) == 0 or len(s) == 0:
            continue
        row = {
            "scope": "by_size", "num_requests": int(n),
            "n_served": len(s), "n_dropped": len(d),
            "mean_slack_served": round(s.mean(), 2),
            "mean_slack_dropped": round(d.mean(), 2),
            "median_slack_served": round(s.median(), 2),
            "median_slack_dropped": round(d.median(), 2),
        }
        try:
            _, p = stats.mannwhitneyu(d, s, alternative="less")
            row["mannwhitney_p_dropped_lower"] = float(p)
        except Exception:
            row["mannwhitney_p_dropped_lower"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "slack_by_served_dropped.csv", index=False)
    print(out.to_string(index=False))
    ov = rows[0]
    verdict = ("dropped requests have LOWER slack (tighter windows)"
               if ov["mean_slack_dropped"] < ov["mean_slack_served"]
               else "dropped requests do NOT have lower slack")
    print(f"  Verdict (overall): {verdict}; "
          f"Mann-Whitney p(dropped<served)={ov.get('mannwhitney_p_dropped_lower')}")

    # Distribution figure: served vs dropped slack.
    plt.figure(figsize=(8, 5))
    bins = np.linspace(pr["slack"].min(), pr["slack"].max(), 40)
    plt.hist(served, bins=bins, alpha=0.6, label="served", density=True, color="#4C78A8")
    plt.hist(dropped, bins=bins, alpha=0.6, label="dropped", density=True, color="#F58518")
    plt.axvline(served.mean(), color="#4C78A8", ls="--")
    plt.axvline(dropped.mean(), color="#F58518", ls="--")
    plt.xlabel("Slack = latest_pickup - request_time (window width)")
    plt.ylabel("Density")
    plt.title("Time-window slack: served vs dropped requests")
    plt.legend(); plt.grid(alpha=0.3)
    savefig(figures_dir / "19_slack_served_vs_dropped.png")

    # Detour of served requests by size (dropped have no detour by definition).
    if "detour_factor" in pr.columns and pr["detour_factor"].notna().any():
        dfc = pr[(pr["served"] == 1) & pr["detour_factor"].notna()]
        if not dfc.empty:
            gd = (dfc.groupby("num_requests")["detour_factor"]
                    .agg(["mean", "sem", "count"]).reset_index())
            gd.to_csv(tables_dir / "served_detour_by_size.csv", index=False)


if __name__ == "__main__":
    main()