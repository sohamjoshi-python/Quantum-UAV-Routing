"""
Solve-side experiment: merge-tree vs pairwise exclusivity on the SAME trip set.

Hypothesis under test
---------------------
If merge-tree exclusivity is the culprit behind ~90-98% raw infeasibility,
pairwise clique penalties (M * x_a * x_b, no aux vars) on the identical pruned
trips / objective should dramatically cut raw_violation_rate while keeping
similar post-cleanup service / VMT.

If pairwise shows the same behavior, the bottleneck is not the merge tree —
it is the mismatch between the trip-benefit QUBO objective and the routing
metrics the paper reports (cleanup-dominated pipeline).

What is held fixed
------------------
  * RTV trip dict (optional capacity-3 triples)
  * prune rule / retained trip keys (built once from merge-tree retention,
    then reused verbatim for pairwise with cap_per_request=None)
  * lambda, M, cost_alpha
  * SA num_reads / num_sweeps / seed

What differs
------------
  * exclusivity encoding only: generate_qubo (merge tree) vs generate_qubo_pairwise

Metrics compared (per encoding)
-------------------------------
  * raw_violation_rate, raw_selected / raw_kept / cleanup keep fraction
  * annealer energy (raw bitstring)
  * percent_serviced, vmt, avg_waiting_time (post cleanup + Hungarian)
  * QUBO size (vars, couplers)

Outputs
-------
  results/merge_vs_pairwise_solve/merge_vs_pairwise_solve.csv
  results/merge_vs_pairwise_solve/merge_vs_pairwise_summary.csv

Usage (from repo root):
  python scripts/compare_merge_vs_pairwise_solve.py --quick
  python scripts/compare_merge_vs_pairwise_solve.py --vehicles 10 --requests 20 --trials 3
  python scripts/compare_merge_vs_pairwise_solve.py --full --trials 5 --fresh
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import dimod
import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_experiment as R  # noqa: E402
from quantum_uav_routing.quantum import quantum_solver  # noqa: E402
from quantum_uav_routing.quantum.penalty_scaling import derive_M, infer_nu_max  # noqa: E402
from quantum_uav_routing.quantum.quantum_solver import (  # noqa: E402
    generate_qubo,
    generate_qubo_pairwise,
    qubo_stats_from_dict,
    raw_infeasibility_from_selection,
)
from quantum_uav_routing.rtv import build_rtv_graph as build_rtv  # noqa: E402
from quantum_uav_routing.rtv import trip_builder  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "merge_vs_pairwise_solve"

# Match capacity-3 wrapper defaults used in recent runs.
DEFAULT_LAMBDA = 2500.0
DEFAULT_COST_ALPHA = 10.0
DEFAULT_CAP = 100
DEFAULT_READS = 200
DEFAULT_SWEEPS = 1000

QUICK_SCENARIOS = {10: [20]}
FULL_SCENARIOS = {10: [20], 20: [40], 30: [60]}


def parse_args():
    p = argparse.ArgumentParser(
        description="Solve merge-tree vs pairwise QUBO on the same pruned trips."
    )
    p.add_argument("--city", type=str, default="32_Phoenix_City")
    p.add_argument("--vehicles", type=int, default=None,
                   help="If set with --requests, run only that one size.")
    p.add_argument("--requests", type=int, default=None)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--quick", action="store_true",
                   help="Single small scenario (10v/20r). Default if no size given.")
    p.add_argument("--full", action="store_true",
                   help="Capacity-3 paper sizes: 20/40/60 requests.")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--lambda-val", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--m-val", type=float, default=R.M_VAL)
    p.add_argument("--auto-m", action="store_true", default=True)
    p.add_argument("--no-auto-m", action="store_false", dest="auto_m")
    p.add_argument("--cost-alpha", type=float, default=DEFAULT_COST_ALPHA)
    p.add_argument("--cap-per-request", type=str, default=str(DEFAULT_CAP))
    p.add_argument("--num-reads", type=int, default=DEFAULT_READS)
    p.add_argument("--num-sweeps", type=int, default=DEFAULT_SWEEPS)
    p.add_argument("--capacity3", action="store_true", default=True)
    p.add_argument("--no-capacity3", action="store_false", dest="capacity3")
    p.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    return p.parse_args()


def _parse_cap(value: str):
    text = str(value).strip().lower()
    if text in ("none", "null", "off"):
        return None
    return int(text)


def install_capacity3_patch(nu: int = 3, delta: int = 1200):
    from quantum_uav_routing.rtv.build_trips_capacity3 import add_triples

    orig = build_rtv.build_trips

    def build_trips_with_triples(requests, vehicles, delta=1200, nu=4, max_dist_km=6.0):
        trips = orig(requests, vehicles, delta=delta, nu=nu, max_dist_km=max_dist_km)
        request_lookup = {r.id: r for r in requests}
        vehicle_by_id = {v.id: v for v in vehicles}
        trips, n_added = add_triples(
            trips,
            requests,
            vehicles,
            travel_fn=trip_builder.travel,
            request_lookup=request_lookup,
            vehicle_by_id=vehicle_by_id,
            nu=nu,
            delta=delta,
        )
        n_triples = sum(1 for k in trips if len(k) == 3)
        print(
            f"    [capacity-3] added {n_added} triple-trips "
            f"({n_triples} total 3-request trips in set)"
        )
        return trips

    build_rtv.build_trips = build_trips_with_triples


def minimal_trip_costs(module_trip_costs):
    minimal = defaultdict(lambda: float("inf"))
    for (tkey, vid), cost in module_trip_costs.items():
        if cost < minimal[tkey]:
            minimal[tkey] = cost
    return minimal


def qubo_energy(Q: dict, sample: dict) -> float:
    if not Q:
        return 0.0
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    full = {v: int(sample.get(v, 0)) for v in bqm.variables}
    return float(bqm.energy(full))


def anneal(Q: dict, num_reads: int, num_sweeps: int, seed: int) -> tuple[dict, float]:
    n = max(max(i, j) for (i, j) in Q.keys()) + 1
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(
        Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed
    )
    best = sampleset.first.sample
    sample = {i: int(best.get(i, 0)) for i in range(n)}
    energy = float(sampleset.first.energy)
    return sample, energy


def greedy_cleanup(all_vars, sample: dict):
    selected = [
        all_vars[i]
        for i, bit in sample.items()
        if bit == 1 and i < len(all_vars) and isinstance(all_vars[i], frozenset)
    ]
    kept = []
    covered = set()
    for tk in selected:
        if not (tk & covered):
            kept.append(tk)
            covered |= tk
    return selected, kept


def hungarian_and_metrics(final_trips_list, requests, vehicles):
    """Same post-cleanup assignment + routing metrics as quantum_mwis_run."""
    Sigma_opt = []
    if final_trips_list and vehicles:
        INVALID = 1e9
        C = np.full((len(final_trips_list), len(vehicles)), INVALID, dtype=float)
        for i, tkey in enumerate(final_trips_list):
            for j, v in enumerate(vehicles):
                cost = quantum_solver.travel_cached(v.id, tkey)
                if cost is not None:
                    C[i, j] = cost
        rows, cols = linear_sum_assignment(C)
        for r, c in zip(rows, cols):
            if C[r, c] < INVALID:
                Sigma_opt.append((final_trips_list[r], vehicles[c].id))

    request_lookup = quantum_solver.request_lookup
    vehicle_lookup = quantum_solver.vehicle_lookup
    travel = quantum_solver.travel

    served_requests = set()
    served_trips = []
    for trip_ids, vid in Sigma_opt:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        served_trips.append((trip_objs, vehicle_lookup[vid]))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = (
        100.0 * len(served_requests) / len(requests) if requests else 0.0
    )

    waiting, VMT = [], 0.0
    for trip, v in served_trips:
        total_time = travel(v, trip)
        VMT += float(total_time)
        _, pickups = travel(v, trip, return_timeline=True)
        for r in trip:
            waiting.append(pickups[r] - r.trr)

    avg_wait = float(sum(waiting) / len(waiting)) if waiting else 0.0
    return {
        "percent_serviced": percent_serviced,
        "vmt": float(VMT),
        "avg_waiting_time": avg_wait,
        "n_assigned_trips": len(Sigma_opt),
        "n_served_requests": len(served_requests),
    }


def trip_keys_from_all_vars(all_vars):
    return [v for v in all_vars if isinstance(v, frozenset)]


def solve_encoding(
    encoding: str,
    Q: dict,
    all_vars: list,
    requests,
    vehicles,
    num_reads: int,
    num_sweeps: int,
    seed: int,
) -> dict:
    t0 = time.perf_counter()
    sample, reported_energy = anneal(Q, num_reads, num_sweeps, seed)
    t_anneal = time.perf_counter() - t0

    energy = qubo_energy(Q, sample)
    selected, kept = greedy_cleanup(all_vars, sample)
    infeas = raw_infeasibility_from_selection(selected, kept)
    metrics = hungarian_and_metrics(kept, requests, vehicles)
    stats = qubo_stats_from_dict(Q)

    keep_frac = (
        float(infeas["raw_kept_trips"]) / float(infeas["raw_selected_trips"])
        if infeas["raw_selected_trips"]
        else float("nan")
    )

    return {
        "encoding": encoding,
        "qubo_vars": stats["qubo_vars"],
        "qubo_couplers": stats["qubo_couplers"],
        "qubo_graph_density": stats["qubo_graph_density"],
        "anneal_time_s": t_anneal,
        "annealer_energy_reported": reported_energy,
        "annealer_energy": energy,
        "raw_selected_trips": infeas["raw_selected_trips"],
        "raw_kept_trips": infeas["raw_kept_trips"],
        "raw_dropped_trips": infeas["raw_dropped_trips"],
        "raw_violation_rate": infeas["raw_violation_rate"],
        "raw_infeasible_instance": infeas["raw_infeasible_instance"],
        "cleanup_keep_fraction": keep_frac,
        **metrics,
    }


def load_completed_keys(csv_path: Path) -> set:
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
            keys.add(
                (
                    int(r["num_vehicles"]),
                    int(r["num_requests"]),
                    int(r["seed"]),
                    int(r["trial"]),
                    str(r["encoding"]),
                )
            )
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


def write_summary(full_csv: Path, summary_csv: Path):
    if not full_csv.exists():
        return
    df = pd.read_csv(full_csv, low_memory=False)
    if df.empty:
        return
    metrics = [
        "raw_violation_rate",
        "cleanup_keep_fraction",
        "annealer_energy",
        "percent_serviced",
        "vmt",
        "avg_waiting_time",
        "raw_selected_trips",
        "raw_kept_trips",
        "qubo_vars",
        "qubo_couplers",
        "anneal_time_s",
    ]
    for c in metrics:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = ["num_vehicles", "num_requests", "encoding"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n"] = len(g)
        for m in metrics:
            if m not in g.columns:
                continue
            row[f"mean_{m}"] = float(g[m].mean())
            row[f"std_{m}"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(group_cols)
    summary.to_csv(summary_csv, index=False)
    print(f"\nSummary written to {summary_csv}")

    # Side-by-side pivot for the decision metrics.
    print("\n=== Side-by-side (mean over trials) ===")
    decide = [
        "mean_raw_violation_rate",
        "mean_cleanup_keep_fraction",
        "mean_percent_serviced",
        "mean_vmt",
        "mean_annealer_energy",
        "mean_qubo_vars",
        "mean_qubo_couplers",
    ]
    for (v, r), sub in summary.groupby(["num_vehicles", "num_requests"]):
        print(f"\n  v={v} r={r}")
        wide = sub.set_index("encoding")
        for enc in ("merge_tree", "pairwise"):
            if enc not in wide.index:
                continue
            bits = [f"{enc}:"]
            for m in decide:
                if m in wide.columns and pd.notna(wide.loc[enc, m]):
                    bits.append(f"{m.replace('mean_', '')}={wide.loc[enc, m]:.4g}")
            print("   ", "  ".join(bits))

        if {"merge_tree", "pairwise"}.issubset(wide.index):
            dv = (
                float(wide.loc["pairwise", "mean_raw_violation_rate"])
                - float(wide.loc["merge_tree", "mean_raw_violation_rate"])
            )
            ds = (
                float(wide.loc["pairwise", "mean_percent_serviced"])
                - float(wide.loc["merge_tree", "mean_percent_serviced"])
            )
            print(
                f"    delta (pairwise - merge_tree): "
                f"raw_violation_rate={dv:+.4f}, percent_serviced={ds:+.2f}"
            )
            if abs(dv) < 0.05 and abs(ds) < 5.0:
                print(
                    "    => Similar behavior: bottleneck is likely NOT the merge-tree "
                    "gadget, but objective/metric mismatch (cleanup dominates)."
                )
            elif dv < -0.15:
                print(
                    "    => Pairwise sharply reduces raw infeasibility: merge-tree "
                    "encoding is implicated."
                )
            else:
                print(
                    "    => Mixed / intermediate: inspect full CSV before concluding."
                )


def scenarios_from_args(args) -> dict:
    if args.vehicles is not None and args.requests is not None:
        return {int(args.vehicles): [int(args.requests)]}
    if args.full:
        return FULL_SCENARIOS
    return QUICK_SCENARIOS


def run_one_instance(
    node_df,
    city: str,
    city_index: int,
    num_vehicles: int,
    num_requests: int,
    trial: int,
    args,
    cap_per_request,
    completed: set,
    full_csv: Path,
):
    seed = R.scenario_seed(num_vehicles, num_requests, city_index, trial)
    trial_num = trial + 1

    need_tree = (num_vehicles, num_requests, seed, trial_num, "merge_tree") not in completed
    need_pair = (num_vehicles, num_requests, seed, trial_num, "pairwise") not in completed
    if not need_tree and not need_pair:
        print(f"  skip (already done) v={num_vehicles} r={num_requests} trial={trial_num}")
        return

    print(
        f"\n--- {city} | v={num_vehicles} | r={num_requests} | "
        f"trial={trial_num} | seed={seed} ---"
    )

    _seed, requests, vehicles, baseline, stats, metadata = R.prepare_scenario(
        node_df, city, num_vehicles, num_requests, city_index, trial
    )
    if not requests:
        print("  No feasible requests; skipping.")
        return

    trips = quantum_solver.trips
    tcosts = minimal_trip_costs(quantum_solver.trip_costs)

    if args.auto_m:
        nu_max = infer_nu_max(trips, default=2)
        M_effective = derive_M(ignore_cost=args.lambda_val, nu_max=nu_max)
        print(f"  [auto_M] nu_max={nu_max} -> M={M_effective:.0f}")
    else:
        M_effective = float(args.m_val)
        print(f"  [fixed M] M={M_effective:.0f}")

    print(
        f"  lambda={args.lambda_val} cost_alpha={args.cost_alpha} "
        f"cap_per_request={cap_per_request} "
        f"reads={args.num_reads} sweeps={args.num_sweeps}"
    )

    # 1) Merge-tree QUBO (defines the retained trip set).
    Q_tree, vars_tree = generate_qubo(
        trips,
        trip_costs=tcosts,
        ignore_cost=args.lambda_val,
        M=M_effective,
        return_numpy=False,
        seed=seed,
        cap_per_request=cap_per_request,
        cost_alpha=float(args.cost_alpha),
    )
    retained = trip_keys_from_all_vars(vars_tree)
    trips_retained = {tk: trips[tk] for tk in retained}
    tcosts_retained = {tk: tcosts[tk] for tk in retained}
    print(
        f"  retained trips = {len(retained)} "
        f"(merge-tree vars={len(vars_tree)}, "
        f"aux={len(vars_tree) - len(retained)})"
    )

    # 2) Pairwise on EXACT same trip keys (no further pruning).
    Q_pair, vars_pair = generate_qubo_pairwise(
        trips_retained,
        trip_costs=tcosts_retained,
        ignore_cost=args.lambda_val,
        M=M_effective,
        return_numpy=False,
        seed=seed,
        cap_per_request=None,
        cost_alpha=float(args.cost_alpha),
    )
    if set(trip_keys_from_all_vars(vars_pair)) != set(retained):
        raise RuntimeError(
            "Pairwise trip set diverged from merge-tree retention; aborting."
        )
    print(f"  pairwise vars = {len(vars_pair)} (no aux)")

    common = {
        "city": city,
        "num_vehicles": num_vehicles,
        "num_requests": num_requests,
        "seed": seed,
        "trial": trial_num,
        "lambda_val": args.lambda_val,
        "M_val": M_effective,
        "cost_alpha": args.cost_alpha,
        "cap_per_request": (
            "none" if cap_per_request is None else int(cap_per_request)
        ),
        "num_reads": args.num_reads,
        "num_sweeps": args.num_sweeps,
        "capacity3": int(bool(args.capacity3)),
        "n_retained_trips": len(retained),
    }

    if need_tree:
        print("  annealing merge_tree ...")
        row = solve_encoding(
            "merge_tree",
            Q_tree,
            vars_tree,
            requests,
            vehicles,
            args.num_reads,
            args.num_sweeps,
            seed,
        )
        row.update(common)
        append_row(row, full_csv)
        print(
            f"    raw_violation_rate={row['raw_violation_rate']:.4f}  "
            f"keep={row['cleanup_keep_fraction']:.4f}  "
            f"serviced={row['percent_serviced']:.1f}%  "
            f"vmt={row['vmt']:.1f}  "
            f"E={row['annealer_energy']:.3g}"
        )

    if need_pair:
        print("  annealing pairwise ...")
        # Same SA seed as merge_tree so differences are encoding-driven, not
        # independent sampler noise (effort still matched via reads/sweeps).
        row = solve_encoding(
            "pairwise",
            Q_pair,
            vars_pair,
            requests,
            vehicles,
            args.num_reads,
            args.num_sweeps,
            seed,
        )
        row.update(common)
        append_row(row, full_csv)
        print(
            f"    raw_violation_rate={row['raw_violation_rate']:.4f}  "
            f"keep={row['cleanup_keep_fraction']:.4f}  "
            f"serviced={row['percent_serviced']:.1f}%  "
            f"vmt={row['vmt']:.1f}  "
            f"E={row['annealer_energy']:.3g}"
        )

    del Q_tree, Q_pair, vars_tree, vars_pair
    gc.collect()


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    cap_per_request = _parse_cap(args.cap_per_request)
    scenarios = scenarios_from_args(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / "merge_vs_pairwise_solve.csv"
    summary_csv = out_dir / "merge_vs_pairwise_summary.csv"

    if args.fresh:
        for path in (full_csv, summary_csv):
            if path.exists():
                path.unlink()

    if args.capacity3:
        install_capacity3_patch()
        # Re-bind save_results' captured build_trips after patch.
        # load_or_build_network -> configure_runtime captures current build_rtv.build_trips.

    print("Merge-tree vs pairwise SOLVE comparison")
    print(f"  capacity3={args.capacity3}  auto_M={args.auto_m}")
    print(f"  lambda={args.lambda_val}  cost_alpha={args.cost_alpha}")
    print(f"  cap_per_request={cap_per_request}  reads={args.num_reads}  sweeps={args.num_sweeps}")
    print(f"  scenarios={scenarios}  trials={args.trials}")
    print(f"  out={out_dir}")

    node_df = R.load_or_build_network(args.city)
    completed = load_completed_keys(full_csv)
    if completed:
        print(f"Resuming: {len(completed)} encoding-rows already present.")

    for city_index, city in enumerate([args.city]):
        for num_vehicles, request_list in scenarios.items():
            for num_requests in request_list:
                for trial in range(args.trials):
                    run_one_instance(
                        node_df,
                        city,
                        city_index,
                        num_vehicles,
                        num_requests,
                        trial,
                        args,
                        cap_per_request,
                        completed,
                        full_csv,
                    )

    write_summary(full_csv, summary_csv)
    print(
        "\nDone. If pairwise << merge_tree on raw_violation_rate with similar "
        "service, the merge-tree formulation is implicated. If both look alike, "
        "the objective/metric mismatch is the better explanation."
    )


if __name__ == "__main__":
    main()
