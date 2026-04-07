from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def safe_series(frame: pd.DataFrame, colname: str, default: float = 0.0) -> pd.Series:
    if colname in frame.columns:
        return pd.to_numeric(frame[colname], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def safe_col(frame: pd.DataFrame, colname: str, default: float = 0.0) -> pd.Series:
    if colname in frame.columns:
        return frame[colname].fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def first_present(candidates: list[str], frame: pd.DataFrame) -> str | None:
    for c in candidates:
        if c in frame.columns:
            return c
    return None


def pick_col(frame: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in frame.columns:
            return c
    raise ValueError(f"None of these columns exist in the DataFrame: {candidates}")


def avg_by_requests(frame: pd.DataFrame, col: str, with_sem: bool = False) -> pd.DataFrame:
    t = frame[["num_requests", col]].dropna()
    if with_sem:
        grouped = t.groupby("num_requests")[col].agg(["mean", "sem", "count"]).reset_index()
        grouped.columns = ["num_requests", f"{col}_mean", f"{col}_sem", f"{col}_count"]
        return grouped
    return t.groupby("num_requests", as_index=False)[col].mean()


def fit_powerlaw(x, y, return_predictor: bool = False):
    """
    Fit y = C * x^a via log-log regression.
    Returns:
      - without predictor: a, C, r2, npts
      - with predictor:    a, C, r2, f, npts
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]

    if len(x) < 2:
        if return_predictor:
            return np.nan, np.nan, np.nan, (lambda n: np.full_like(np.asarray(n, dtype=float), np.nan)), len(x)
        return np.nan, np.nan, np.nan, len(x)

    lx = np.log10(x)
    ly = np.log10(y)

    a, b = np.polyfit(lx, ly, 1)
    ly_hat = a * lx + b

    ss_res = np.sum((ly - ly_hat) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    C = 10 ** b

    if return_predictor:
        def f(n):
            n = np.asarray(n, dtype=float)
            return C * (n ** a)
        return a, C, r2, f, len(x)

    return a, C, r2, len(x)


def dwave_qpu_time_seconds(
    num_reads: int = 2000,
    anneal_us: float = 50,
    readout_us: float = 120,
    qpu_delay_us: float = 20,
    programming_ms: float = 10,
    other_overhead_ms: float = 5,
    queue_overhead_s: float = 0.0,
) -> float:
    per_sample_us = anneal_us + readout_us + qpu_delay_us
    sampling_s = (num_reads * per_sample_us) * 1e-6
    fixed_s = (programming_ms + other_overhead_ms) * 1e-3
    return queue_overhead_s + fixed_s + sampling_s


def paired_test(
    label_a: str,
    d_a: pd.DataFrame,
    label_b: str,
    d_b: pd.DataFrame,
    metric_name: str,
    instance_cols: list[str],
) -> dict:
    merged = pd.merge(
        d_a[instance_cols + [metric_name]],
        d_b[instance_cols + [metric_name]],
        on=instance_cols,
        suffixes=("_a", "_b"),
    ).dropna()

    if len(merged) < 5:
        return {
            "samples": len(merged),
            "mean_difference": np.nan,
            "p_ttest": np.nan,
            "p_wilcoxon": np.nan,
            "cohens_d": np.nan,
            "ci_95": (np.nan, np.nan),
        }

    x = merged[f"{metric_name}_a"].to_numpy(dtype=float)
    y = merged[f"{metric_name}_b"].to_numpy(dtype=float)
    diff = y - x

    _, p_t = stats.ttest_rel(y, x)

    try:
        _, p_w = stats.wilcoxon(diff)
    except Exception:
        p_w = np.nan

    diff_std = np.std(diff, ddof=1)
    cohens_d = np.mean(diff) / diff_std if diff_std > 0 else np.nan

    ci = stats.t.interval(
        0.95,
        len(diff) - 1,
        loc=np.mean(diff),
        scale=stats.sem(diff),
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "metric": metric_name,
        "samples": len(diff),
        "mean_difference": float(np.mean(diff)),
        "p_ttest": float(p_t),
        "p_wilcoxon": float(p_w) if np.isfinite(p_w) else np.nan,
        "cohens_d": float(cohens_d) if np.isfinite(cohens_d) else np.nan,
        "ci_95": tuple(float(v) for v in ci),
    }


def pareto_mask(group: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    g = group.reset_index(drop=True)
    dominated = np.zeros(len(g), dtype=bool)

    for i in range(len(g)):
        xi, yi = g.loc[i, x_col], g.loc[i, y_col]
        for j in range(len(g)):
            if i == j:
                continue
            xj, yj = g.loc[j, x_col], g.loc[j, y_col]
            if (xj <= xi and yj >= yi) and (xj < xi or yj > yi):
                dominated[i] = True
                break

    return ~dominated


def find_crossover(f_a, f_b, n_min: float = 5, n_max: float = 2_000_000, grid: int = 6000):
    ns = np.linspace(n_min, n_max, grid)
    ta = f_a(ns)
    tb = f_b(ns)
    diff = ta - tb

    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(idx) == 0:
        return None, ns, ta, tb

    i = idx[0]
    n0, n1 = ns[i], ns[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    n_star = n0 - d0 * (n1 - n0) / (d1 - d0)
    return float(n_star), ns, ta, tb