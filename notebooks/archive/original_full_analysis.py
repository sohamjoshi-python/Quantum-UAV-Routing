import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def avg_by_requests(d, col):
    t = d[["num_requests", col]].dropna()
    return t.groupby("num_requests", as_index=False)[col].mean()

def fit_powerlaw(x, y):
    """Fit y ~ C * x^a via log10 regression. Return (a, C, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]; y = y[m]
    lx = np.log10(x); ly = np.log10(y)
    a, b = np.polyfit(lx, ly, 1)
    ly_hat = a * lx + b
    r2 = 1 - np.sum((ly - ly_hat) ** 2) / np.sum((ly - np.mean(ly)) ** 2)
    C = 10**b
    return a, C, r2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def safe_series(frame, colname, default=0.0):
    if colname in frame.columns:
        return pd.to_numeric(frame[colname], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)

def avg_by_requests(d, col):
    t = d[["num_requests", col]].dropna()
    return t.groupby("num_requests", as_index=False)[col].mean()

def fit_powerlaw(x, y):
    """Fit y ~ C * x^a via log10 regression. Return (a, C, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if len(x) < 2:
        raise ValueError("Need at least 2 positive points for power-law fit.")
    lx = np.log10(x)
    ly = np.log10(y)
    a, b = np.polyfit(lx, ly, 1)
    ly_hat = a * lx + b
    ss_res = np.sum((ly - ly_hat) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    C = 10 ** b
    return a, C, r2

def first_present(cols, frame):
    for c in cols:
        if c in frame.columns:
            return c
    return None

def dwave_qpu_time_seconds(
    num_reads=2000,
    anneal_us=50,
    readout_us=120,
    qpu_delay_us=20,
    programming_ms=10,
    other_overhead_ms=5,
    queue_overhead_s=0.0
):
    per_sample_us = anneal_us + readout_us + qpu_delay_us
    sampling_s = (num_reads * per_sample_us) * 1e-6
    fixed_s = (programming_ms + other_overhead_ms) * 1e-3
    return queue_overhead_s + fixed_s + sampling_s

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def safe_series(frame, colname, default=0.0):
    if colname in frame.columns:
        return pd.to_numeric(frame[colname], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)

def dwave_qpu_time_seconds(
    num_reads=2000,
    anneal_us=50,
    readout_us=120,
    qpu_delay_us=20,
    programming_ms=10,
    other_overhead_ms=5,
    queue_overhead_s=0.0
):
    per_sample_us = anneal_us + readout_us + qpu_delay_us
    sampling_s = (num_reads * per_sample_us) * 1e-6
    fixed_s = (programming_ms + other_overhead_ms) * 1e-3
    return queue_overhead_s + fixed_s + sampling_s

def fit_powerlaw(x, y):
    """
    Fit y ~ C * x^a via log-log regression.
    Returns a, C, r2, npts
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)

    lx = np.log10(x)
    ly = np.log10(y)
    a, b = np.polyfit(lx, ly, 1)
    ly_hat = a * lx + b
    ss_res = np.sum((ly - ly_hat) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    C = 10 ** b
    return a, C, r2, len(x)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def avg_by_requests(d, col):
    t = d[["num_requests", col]].dropna()
    return t.groupby("num_requests", as_index=False)[col].mean()

def fit_powerlaw(x, y):
    """Fit y ~ C * x^a via log10 regression. Return (a, C, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    # Check if x or y is empty after masking
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan # Return NaNs if no valid data points

    lx = np.log10(x)
    ly = np.log10(y)
    a, b = np.polyfit(lx, ly, 1)  # ly = a*lx + b
    ly_hat = a * lx + b
    r2 = 1 - np.sum((ly - ly_hat) ** 2) / np.sum((ly - np.mean(ly)) ** 2)
    C = 10 ** b
    return a, C, r2

def first_present(cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def dwave_qpu_time_seconds(
    num_reads=2000,
    anneal_us=50,
    readout_us=120,
    qpu_delay_us=20,
    programming_ms=10,
    other_overhead_ms=5,
    queue_overhead_s=0.0
):
    per_sample_us = anneal_us + readout_us + qpu_delay_us
    sampling_s = (num_reads * per_sample_us) * 1e-6
    fixed_s = (programming_ms + other_overhead_ms) * 1e-3
    return queue_overhead_s + fixed_s + sampling_s

def safe(col):
    return dq[col].fillna(0.0) if col in dq.columns else 0.0

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def avg_by_requests(d, col):
    t = d[["num_requests", col]].dropna()
    grouped = t.groupby("num_requests")[col].agg(["mean", "sem", "count"])
    grouped = grouped.reset_index()
    grouped.columns = ["num_requests", col + "_mean", col + "_sem", col + "_count"]
    return grouped

def pick_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns exist in the CSV: {candidates}")

import pandas as pd

import numpy as np

from scipy import stats

def paired_test(label_a, d_a, label_b, d_b, metric_name):
    print("\n" + "=" * 60)
    print(f"Metric: {metric_name}  |  {label_a} vs {label_b}")
    print("=" * 60)

    merged = pd.merge(
        d_a[instance_cols + [metric_name]],
        d_b[instance_cols + [metric_name]],
        on=instance_cols,
        suffixes=("_a", "_b"),
    ).dropna()

    if len(merged) < 5:
        print("Not enough paired samples.")
        return

    x = merged[f"{metric_name}_a"].values
    y = merged[f"{metric_name}_b"].values
    diff = y - x

    t_stat, p_val = stats.ttest_rel(y, x)
    try:
        w_stat, p_w = stats.wilcoxon(diff)
    except Exception:
        p_w = np.nan

    d_cohen = np.mean(diff) / np.std(diff, ddof=1)
    ci = stats.t.interval(0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff))

    print(f"Samples:                       {len(diff)}")
    print(f"Mean difference ({label_b} - {label_a}): {np.mean(diff):.4f}")
    print(f"Paired t-test p-value:         {p_val:.6g}")
    print(f"Wilcoxon p-value:              {p_w:.6g}")
    print(f"Cohen's d:                     {d_cohen:.3f}")
    print(f"95% CI of difference:          {ci}")

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def avg_by_requests(d, col):
    t = d[["num_requests", col]].dropna()
    return t.groupby("num_requests", as_index=False)[col].mean()

def fit_powerlaw(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]; y = y[m]
    lx = np.log10(x); ly = np.log10(y)
    a, b = np.polyfit(lx, ly, 1)
    ly_hat = a * lx + b
    r2 = 1 - np.sum((ly - ly_hat) ** 2) / np.sum((ly - np.mean(ly)) ** 2)
    C = 10**b
    def f(n): return C * (np.asarray(n, float) ** a)
    return a, C, r2, f

def first_present(cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def dwave_qpu_time_seconds(
    num_reads=2000, anneal_us=50, readout_us=120,
    qpu_delay_us=20, programming_ms=10, other_overhead_ms=5, queue_overhead_s=0.0
):
    per_sample_us = anneal_us + readout_us + qpu_delay_us
    sampling_s    = (num_reads * per_sample_us) * 1e-6
    fixed_s       = (programming_ms + other_overhead_ms) * 1e-3
    return queue_overhead_s + fixed_s + sampling_s

def pred_classical_total(n):
    return f_class_total(n)

def pred_quantum_total(n, qpu_kwargs=None):
    if qpu_kwargs is None: qpu_kwargs = {}
    return f_qubo(n) + f_post(n) + dwave_qpu_time_seconds(**qpu_kwargs)

def pred_greedy_total(n):
    return f_greedy_total(n)

def find_crossover(f_a, f_b, n_min=5, n_max=2_000_000_000_000, grid=6000, qpu_kwargs=None):
    ns   = np.linspace(n_min, n_max, grid)
    ta   = f_a(ns) if qpu_kwargs is None else pred_quantum_total(ns, qpu_kwargs)
    tb   = f_b(ns)
    diff = ta - tb
    idx  = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(idx) == 0:
        return None, ns, ta, tb
    i    = idx[0]
    n0, n1, d0, d1 = ns[i], ns[i+1], diff[i], diff[i+1]
    n_star = n0 - d0 * (n1 - n0) / (d1 - d0)
    return float(n_star), ns, ta, tb

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def dwave_qpu_time_seconds(
    num_reads=2000,
    anneal_us=50,
    readout_us=120,
    qpu_delay_us=20,
    programming_ms=10,
    other_overhead_ms=5,
    queue_overhead_s=0.0
):
    per_sample_us = anneal_us + readout_us + qpu_delay_us
    sampling_s = (num_reads * per_sample_us) * 1e-6
    fixed_s = (programming_ms + other_overhead_ms) * 1e-3
    return queue_overhead_s + fixed_s + sampling_s

def safe_col(frame, colname, default=0.0):
    if colname in frame.columns:
        return frame[colname].fillna(default)
    return pd.Series(default, index=frame.index)

def pareto_mask(group, x_col, y_col):
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
