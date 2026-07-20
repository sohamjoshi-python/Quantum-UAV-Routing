"""
Analyze embedding_results.csv (from embedding_analysis.py) into paper-ready
deliverables, with HONEST handling of timeout-capped failures.

Critical distinction this script enforces:
  minorminer is a heuristic. embed_success=0 can mean either
    (a) GENUINE failure: search finished and found no embedding, OR
    (b) TIMEOUT: the search hit the time budget and gave up -- which is NOT
        evidence the QUBO is unembeddable, only that the budget was too small.
  A failure whose embed_time_s sits at (>= wall_frac of) the budget is classified
  'timeout_inconclusive' and must NOT be reported as "does not fit on hardware".
  Only failures that returned well under budget are 'genuine_fail'.

Because the input CSV does not store the timeout value, pass it with --budget
(the --timeout you used when running embedding_analysis.py). If you ran different
budgets, split the analysis or re-run uniformly.

Outputs under results/embedding/:
  - embedding_table.csv / .md   per-size, per-encoding: physical qubits, chains,
                                overhead ratios; only over embeddable sizes
  - embedding_frontier.txt      where each encoding stops embedding, with each
                                failure labelled timeout_inconclusive vs genuine
  - embedding_figure.png        max chain vs size, and physical-qubit overhead vs
                                size, both encodings (embeddable sizes only)

Usage:
  python scripts/analyze_embedding.py --budget 120
  python scripts/analyze_embedding.py --in results/embedding/embedding_results.csv --budget 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify(df: pd.DataFrame, budget: float, wall_frac: float) -> pd.DataFrame:
    df = df.copy()
    for c in ("embed_success", "embed_time_s", "physical_qubits", "max_chain",
              "mean_chain", "logical_qubits", "num_requests"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def kind(r):
        if r["embed_success"] == 1:
            return "embedded"
        # failure: timeout-capped or genuine?
        if pd.notna(r["embed_time_s"]) and r["embed_time_s"] >= wall_frac * budget:
            return "timeout_inconclusive"
        return "genuine_fail"

    df["fail_kind"] = df.apply(kind, axis=1)
    return df


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["fail_kind"] == "embedded"]
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for (n, enc), sub in ok.groupby(["num_requests", "encoding"]):
        def ms(c):
            v = sub[c].dropna()
            return (v.mean(), v.std() if len(v) > 1 else 0.0)
        lq_m, _ = ms("logical_qubits")
        pq_m, pq_s = ms("physical_qubits")
        mc_m, mc_s = ms("max_chain")
        mn_m, _ = ms("mean_chain")
        rows.append({
            "num_requests": int(n),
            "encoding": enc,
            "n_seeds": len(sub),
            "logical_qubits": round(lq_m, 1),
            "physical_qubits_mean": round(pq_m, 1),
            "physical_qubits_std": round(pq_s, 1),
            "phys_over_logical": round(pq_m / lq_m, 2) if lq_m else float("nan"),
            "max_chain_mean": round(mc_m, 1),
            "max_chain_std": round(mc_s, 1),
            "mean_chain_mean": round(mn_m, 2),
        })
    t = pd.DataFrame(rows).sort_values(["num_requests", "encoding"]).reset_index(drop=True)
    return t


def add_ratio_columns(table: pd.DataFrame) -> pd.DataFrame:
    """For sizes where BOTH encodings embedded, add pairwise/merge-tree ratios."""
    if table.empty:
        return table
    piv = table.pivot(index="num_requests", columns="encoding")
    out = []
    for n in piv.index:
        try:
            tree_pq = piv[("physical_qubits_mean", "merge_tree")][n]
            pair_pq = piv[("physical_qubits_mean", "pairwise")][n]
            tree_mc = piv[("max_chain_mean", "merge_tree")][n]
            pair_mc = piv[("max_chain_mean", "pairwise")][n]
            if pd.notna(tree_pq) and pd.notna(pair_pq):
                out.append({
                    "num_requests": int(n),
                    "phys_qubits_pair_over_tree": round(pair_pq / tree_pq, 2),
                    "max_chain_pair_over_tree": round(pair_mc / tree_mc, 2) if tree_mc else float("nan"),
                })
        except KeyError:
            continue
    return pd.DataFrame(out)


def write_markdown(table: pd.DataFrame, ratios: pd.DataFrame, md_path: Path):
    lines = ["## Embedding overhead on Pegasus (embeddable sizes only)", ""]
    lines.append("| Requests | Encoding | Logical | Physical qubits | Phys/Logical | Max chain | Mean chain |")
    lines.append("|---:|:--|---:|---:|---:|---:|---:|")
    for _, r in table.iterrows():
        pq = (f"{r['physical_qubits_mean']:,.0f} ± {r['physical_qubits_std']:,.0f}"
              if r["physical_qubits_std"] > 0 else f"{r['physical_qubits_mean']:,.0f}")
        mc = (f"{r['max_chain_mean']:.1f} ± {r['max_chain_std']:.1f}"
              if r["max_chain_std"] > 0 else f"{r['max_chain_mean']:.1f}")
        lines.append(
            f"| {r['num_requests']} | {r['encoding']} | {r['logical_qubits']:,.0f} "
            f"| {pq} | {r['phys_over_logical']:.2f}× | {mc} | {r['mean_chain_mean']:.2f} |"
        )
    if not ratios.empty:
        lines += ["", "### Pairwise / merge-tree overhead ratio (both embedded)", ""]
        lines.append("| Requests | Physical-qubit ratio | Max-chain ratio |")
        lines.append("|---:|---:|---:|")
        for _, r in ratios.iterrows():
            lines.append(
                f"| {r['num_requests']} | {r['phys_qubits_pair_over_tree']:.2f}× "
                f"| {r['max_chain_pair_over_tree']:.2f}× |"
            )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_frontier(df: pd.DataFrame, budget: float, out_path: Path):
    lines = ["EMBEDDING FRONTIER (honest timeout handling)", "=" * 55,
             f"Embedding budget assumed: {budget:.0f}s per attempt.",
             "A failure at/near this wall is TIMEOUT (inconclusive), not proof the",
             "QUBO is unembeddable.", ""]
    for enc in sorted(df["encoding"].unique()):
        sub = df[df["encoding"] == enc]
        lines.append(f"[{enc}]")
        for n in sorted(sub["num_requests"].unique()):
            s = sub[sub["num_requests"] == n]
            kinds = s["fail_kind"].value_counts().to_dict()
            n_emb = kinds.get("embedded", 0)
            n_to = kinds.get("timeout_inconclusive", 0)
            n_gf = kinds.get("genuine_fail", 0)
            tag = []
            if n_emb: tag.append(f"{n_emb} embedded")
            if n_to: tag.append(f"{n_to} TIMEOUT(inconclusive)")
            if n_gf: tag.append(f"{n_gf} genuine-fail")
            lines.append(f"  n={int(n):>4}: " + ", ".join(tag))
        lines.append("")

    # honest one-line summary of the frontier per encoding
    lines.append("SUMMARY")
    lines.append("-" * 40)
    for enc in sorted(df["encoding"].unique()):
        sub = df[df["encoding"] == enc]
        emb_sizes = sorted(sub[sub["fail_kind"] == "embedded"]["num_requests"].unique())
        to_sizes = sorted(sub[sub["fail_kind"] == "timeout_inconclusive"]["num_requests"].unique())
        gf_sizes = sorted(sub[sub["fail_kind"] == "genuine_fail"]["num_requests"].unique())
        max_emb = f"{int(max(emb_sizes))}" if emb_sizes else "none"
        line = f"{enc}: embeds through n={max_emb}"
        if to_sizes:
            line += (f"; n>={int(min(to_sizes))} INCONCLUSIVE (timeout -- "
                     f"raise --timeout/--tries to resolve)")
        if gf_sizes:
            line += f"; genuine failure at n={[int(x) for x in gf_sizes]}"
        lines.append(line)
    lines.append("")
    lines.append("NOTE: Do NOT report timeout_inconclusive sizes as 'does not fit on")
    lines.append("hardware'. Re-run those sizes with a larger budget (e.g. --timeout")
    lines.append("600 --tries 20) to determine the TRUE frontier.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def make_figure(table: pd.DataFrame, fig_path: Path):
    if table.empty:
        print("(No embeddable rows; figure skipped.)")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"(Figure skipped: {exc})")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"merge_tree": "#1f77b4", "pairwise": "#d62728"}
    labels = {"merge_tree": "Merge-tree", "pairwise": "Pairwise"}
    for enc in ("merge_tree", "pairwise"):
        sub = table[table["encoding"] == enc].sort_values("num_requests")
        if sub.empty:
            continue
        ax1.errorbar(sub["num_requests"], sub["max_chain_mean"],
                     yerr=sub["max_chain_std"], marker="o", capsize=3,
                     color=colors[enc], label=labels[enc])
        ax2.plot(sub["num_requests"], sub["physical_qubits_mean"],
                 marker="s", color=colors[enc], label=labels[enc])
    ax1.set_xlabel("Number of Requests")
    ax1.set_ylabel("Max chain length")
    ax1.set_title("(a) Max chain length vs size\n(embedding property; grows for both, slower for merge-tree)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Number of Requests")
    ax2.set_ylabel("Physical qubits")
    ax2.set_title("(b) Physical-qubit footprint on Pegasus")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze embedding results honestly.")
    p.add_argument("--in", dest="in_csv", type=str,
                   default="results/embedding/embedding_results.csv")
    p.add_argument("--out-dir", type=str, default="results/embedding")
    p.add_argument("--budget", type=float, required=True,
                   help="The --timeout (seconds) used when running embedding_analysis.py. "
                        "Failures at/near this wall are treated as inconclusive timeouts.")
    p.add_argument("--wall-frac", type=float, default=0.9,
                   help="A failure with embed_time_s >= wall_frac*budget is a timeout.")
    return p.parse_args()


def main():
    args = parse_args()
    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise SystemExit(f"Input not found: {in_csv}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv, low_memory=False)
    df = classify(df, args.budget, args.wall_frac)

    table = build_table(df)
    ratios = add_ratio_columns(table)

    table_csv = out_dir / "embedding_table.csv"
    table_md = out_dir / "embedding_table.md"
    frontier_txt = out_dir / "embedding_frontier.txt"
    fig_path = out_dir / "embedding_figure.png"

    table.to_csv(table_csv, index=False)
    write_markdown(table, ratios, table_md)
    make_figure(table, fig_path)
    print()
    write_frontier(df, args.budget, frontier_txt)

    print(f"\nOutputs in {out_dir}/: {table_csv.name}, {table_md.name}, "
          f"{frontier_txt.name}, {fig_path.name}")


if __name__ == "__main__":
    main()