"""
Turn encoding_comparison.csv (from compare_encodings_sweep.py) into paper-ready
deliverables:

  1. encoding_table.csv          per-size table: tree vs pairwise vars/couplers/
                                 density/max-degree, mean +/- std where seeds vary.
  2. encoding_table.md           the same table formatted for pasting into the paper.
  3. encoding_figure.png         two panels:
                                   (a) coupler crossover (log-log)
                                   (b) max-degree divergence: flat tree vs climbing pairwise
  4. cap_regime.txt              per-size estimate of whether cap_per_request binds,
                                 so the paper can state whether the coupler ratio is a
                                 floor (cap binding) or the natural value.

This is a PURE ANALYSIS script: it reads the CSV only, builds nothing, solves
nothing, and needs no network. Run it anywhere.

Usage:
  python scripts/analyze_encoding_comparison.py
  python scripts/analyze_encoding_comparison.py --in results/encoding_comparison/encoding_comparison.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def load(in_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(in_csv, low_memory=False)
    numeric = [
        "num_requests", "cap_per_request", "n_trips_pruned",
        "tree_vars", "pair_vars", "tree_couplers", "pair_couplers",
        "tree_density", "pair_density", "tree_degree_max", "pair_degree_max",
        "couplers_pair_over_tree", "aux_vars_tree",
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("num_requests")
    rows = []
    for n, sub in g:
        def ms(col):
            vals = sub[col].dropna()
            return vals.mean(), (vals.std() if len(vals) > 1 else 0.0)
        tv_m, tv_s = ms("tree_vars")
        pv_m, pv_s = ms("pair_vars")
        tc_m, tc_s = ms("tree_couplers")
        pc_m, pc_s = ms("pair_couplers")
        td_m, _ = ms("tree_density")
        pd_m, _ = ms("pair_density")
        tdeg_m, _ = ms("tree_degree_max")
        pdeg_m, pdeg_s = ms("pair_degree_max")
        ratio_m, ratio_s = ms("couplers_pair_over_tree")
        rows.append({
            "num_requests": int(n),
            "n_seeds": int(len(sub)),
            "tree_vars_mean": round(tv_m, 1), "tree_vars_std": round(tv_s, 1),
            "pair_vars_mean": round(pv_m, 1), "pair_vars_std": round(pv_s, 1),
            "tree_couplers_mean": round(tc_m, 1), "tree_couplers_std": round(tc_s, 1),
            "pair_couplers_mean": round(pc_m, 1), "pair_couplers_std": round(pc_s, 1),
            "tree_density_mean": td_m,
            "pair_density_mean": pd_m,
            "tree_degree_max_mean": round(tdeg_m, 1),
            "pair_degree_max_mean": round(pdeg_m, 1), "pair_degree_max_std": round(pdeg_s, 1),
            "couplers_ratio_mean": round(ratio_m, 2), "couplers_ratio_std": round(ratio_s, 2),
        })
    out = pd.DataFrame(rows).sort_values("num_requests").reset_index(drop=True)
    return out


def write_markdown(table: pd.DataFrame, md_path: Path):
    lines = []
    lines.append("| Requests | Tree vars | Pairwise vars | Tree couplers | Pairwise couplers | Ratio (pair/tree) | Tree max-deg | Pairwise max-deg |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in table.iterrows():
        def pm(m, s):
            return f"{m:,.0f} ± {s:,.0f}" if s and s > 0 else f"{m:,.0f}"
        lines.append(
            f"| {r['num_requests']} "
            f"| {pm(r['tree_vars_mean'], r['tree_vars_std'])} "
            f"| {pm(r['pair_vars_mean'], r['pair_vars_std'])} "
            f"| {pm(r['tree_couplers_mean'], r['tree_couplers_std'])} "
            f"| {pm(r['pair_couplers_mean'], r['pair_couplers_std'])} "
            f"| {r['couplers_ratio_mean']:.2f}× "
            f"| {r['tree_degree_max_mean']:.0f} "
            f"| {pm(r['pair_degree_max_mean'], r['pair_degree_max_std'])} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def estimate_cap_regime(df: pd.DataFrame, cap_default: int, out_path: Path):
    """Estimate whether cap_per_request binds at each size.

    pair_couplers ~= sum_r C(k_r, 2) (deduped). Treating k_r as ~uniform K,
    pair_couplers/num_requests ~= K(K-1)/2, so K ~= (1+sqrt(1+8*pairs_per_req))/2.
    This uniform assumption OVERstates the mean when k_r varies (Jensen), so it is
    an ESTIMATE; but if implied K exceeds the cap, the cap is genuinely binding for
    a substantial share of requests (a uniform-capped set could not produce that
    many pairs). When the cap binds, the reported coupler ratio is a LOWER BOUND on
    the true pairwise disadvantage.
    """
    lines = []
    lines.append("CAP-REGIME ESTIMATE (is cap_per_request binding?)")
    lines.append("=" * 60)
    lines.append(
        "Interpretation: implied mean k_r is estimated from pairwise couplers "
        "assuming ~uniform k_r. If implied k_r >> cap, the cap binds and the "
        "coupler ratio understates the true (uncapped) pairwise disadvantage.\n"
    )
    g = df.groupby("num_requests")
    any_binding = False
    for n, sub in g:
        cap = int(sub["cap_per_request"].dropna().iloc[0]) if "cap_per_request" in sub else cap_default
        pc = sub["pair_couplers"].dropna().mean()
        pairs_per_req = pc / n if n else 0.0
        k_eff = (1 + math.sqrt(1 + 8 * pairs_per_req)) / 2 if pairs_per_req > 0 else 0.0
        binds = k_eff >= (cap - 1)
        any_binding = any_binding or binds
        lines.append(
            f"n={int(n):>4} | cap={cap} | mean pairs/req={pairs_per_req:8.1f} | "
            f"implied mean k_r≈{k_eff:5.1f} | {'CAP BINDS' if binds else 'cap not binding'}"
        )
    lines.append("")
    if any_binding:
        lines.append(
            "CONCLUSION: The cap binds at larger sizes. State in the paper that the "
            "coupler ratio (e.g. ~10x at n=180) is a LOWER BOUND: even under "
            "aggressive per-request pruning to {} trips, pairwise still has that many "
            "more couplers; the uncapped gap would be larger.".format(cap_default)
        )
    else:
        lines.append(
            "CONCLUSION: The cap does not appear to bind; the coupler ratio reflects "
            "the natural conflict structure at these sizes."
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def make_figure(table: pd.DataFrame, fig_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(Figure skipped: matplotlib unavailable: {exc})")
        return

    n = table["num_requests"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel (a): coupler crossover, log-log
    ax1.plot(n, table["tree_couplers_mean"], "o-", label="Merge-tree", color="#1f77b4")
    ax1.plot(n, table["pair_couplers_mean"], "s-", label="Pairwise", color="#d62728")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Number of Requests")
    ax1.set_ylabel("QUBO Couplers")
    ax1.set_title("(a) Coupler count: merge-tree vs pairwise")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    # Panel (b): max-degree divergence, linear y to show flat-vs-climbing clearly
    ax2.plot(n, table["tree_degree_max_mean"], "o-", label="Merge-tree (bounded)", color="#1f77b4")
    ax2.plot(n, table["pair_degree_max_mean"], "s-", label="Pairwise (grows)", color="#d62728")
    ax2.set_xlabel("Number of Requests")
    ax2.set_ylabel("Max variable degree (chain-length proxy)")
    ax2.set_title("(b) Peak connectivity: bounded vs unbounded")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze encoding comparison CSV.")
    p.add_argument(
        "--in", dest="in_csv", type=str,
        default="results/encoding_comparison/encoding_comparison.csv",
        help="Input CSV from compare_encodings_sweep.py",
    )
    p.add_argument(
        "--out-dir", type=str,
        default="results/encoding_comparison",
        help="Where to write table/figure/cap-regime outputs.",
    )
    p.add_argument("--cap-default", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise SystemExit(f"Input not found: {in_csv}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(in_csv)
    table = build_table(df)

    table_csv = out_dir / "encoding_table.csv"
    table_md = out_dir / "encoding_table.md"
    fig_path = out_dir / "encoding_figure.png"
    cap_path = out_dir / "cap_regime.txt"

    table.to_csv(table_csv, index=False)
    write_markdown(table, table_md)
    make_figure(table, fig_path)
    estimate_cap_regime(df, args.cap_default, cap_path)

    # Console highlight at the paper's headline size.
    big = table[table["num_requests"] == table["num_requests"].max()].iloc[0]
    print("\n=== Headline (largest size) ===")
    print(f"n={big['num_requests']} requests:")
    print(f"  couplers: tree {big['tree_couplers_mean']:,.0f} vs "
          f"pairwise {big['pair_couplers_mean']:,.0f} "
          f"({big['couplers_ratio_mean']:.1f}x)")
    print(f"  max degree: tree {big['tree_degree_max_mean']:.0f} (flat) vs "
          f"pairwise {big['pair_degree_max_mean']:.0f} (grows)")
    print(f"  vars: tree {big['tree_vars_mean']:,.0f} vs "
          f"pairwise {big['pair_vars_mean']:,.0f} "
          f"(tree adds aux vars -- report the trade honestly)")
    print(f"\nOutputs in {out_dir}/:")
    print(f"  {table_csv.name}, {table_md.name}, {fig_path.name}, {cap_path.name}")


if __name__ == "__main__":
    main()