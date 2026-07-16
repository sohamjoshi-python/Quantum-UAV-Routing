"""
MATCHED-SERVICE QUALITY COMPARISON: pairwise QUBO vs ILP.

Goal: answer the real question honestly -- "once the QUBO is (a) using the
annealer-solvable PAIRWISE exclusivity encoding (0% raw infeasibility) and (b)
given an objective where route cost actually competes with service (cost_alpha),
how close does its ROUTE QUALITY get to the exact ILP optimum AT THE SAME SERVICE
LEVEL?"

Why this is the fair test:
  - The old comparison was rigged two ways: the merge-tree produced ~98% infeasible
    raw output (SA can't solve its aux network), and the objective
    (lambda*|t| - cost, lambda=2500) rewards SERVICE ~4-15x more than it penalizes
    a long route -- so the QUBO was a service-maximizer, ILP a cost-minimizer.
    Comparing their VMT directly was apples-to-oranges.
  - ILP minimizes:  sum(route_cost) + ignore_cost * (unserved).  The QUBO maximizes
    the algebraically-equivalent  ignore_cost*served - cost.  Same tradeoff; the
    QUBO just needs cost_alpha to scale its cost term so the tradeoff is explorable.

What this script does:
  For one instance, sweep cost_alpha for the PAIRWISE QUBO. Each alpha yields a
  (percent_serviced, VMT) point. ILP gives ONE (serviced, VMT) reference. Plot the
  QUBO's service-vs-VMT curve against the ILP point. The honest quality gap is the
  VERTICAL distance (VMT) between the QUBO curve and ILP AT THE SAME service level
  -- NOT the raw VMT difference at different service levels.

REQUIRES: generate_qubo_pairwise must accept a cost_alpha multiplier on the cost
term, i.e. w_t = ignore_cost*|t| - cost_alpha*trip_cost. If your local version does
not yet have cost_alpha, add it (one line at the w_t computation). This script
passes cost_alpha through.

Usage (from repo root):
  python scripts/matched_quality_comparison.py --seed 402042 --nv 20 --nr 40 \
      --alphas 1 5 10 20 50 --reads 200 --sweeps 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=402042)
    p.add_argument("--nv", type=int, default=20, help="num vehicles")
    p.add_argument("--nr", type=int, default=40, help="num requests")
    p.add_argument("--capacity3", action="store_true", default=True)
    p.add_argument("--cap-per-request", type=int, default=100)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[1, 5, 10, 20, 50],
                   help="cost_alpha values to sweep for the pairwise QUBO")
    p.add_argument("--lambda-val", type=float, default=2500.0)
    p.add_argument("--reads", type=int, default=200)
    p.add_argument("--sweeps", type=int, default=2000)
    p.add_argument("--out", type=str, default="results/matched_quality")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- import the project's machinery ---
    # These imports mirror run_experiment.py's setup. The exact scenario-building
    # entrypoint is save_results.final_trips + inject_scenario_globals.
    import run_experiment as R
    from quantum_uav_routing.quantum import quantum_solver as QS
    from quantum_uav_routing.classical import ilp_solver

    # capacity-3 trip generation if requested
    if args.capacity3:
        from quantum_uav_routing.rtv import build_rtv_graph as build_rtv
        from quantum_uav_routing.rtv import trip_builder
        from quantum_uav_routing.rtv.build_trips_capacity3 import add_triples
        _orig = build_rtv.build_trips
        def _with_triples(requests, vehicles, delta=1200, nu=4, max_dist_km=6.0):
            trips = _orig(requests, vehicles, delta=delta, nu=nu, max_dist_km=max_dist_km)
            rl = {r.id: r for r in requests}; vb = {v.id: v for v in vehicles}
            trips, _ = add_triples(trips, requests, vehicles, trip_builder.travel,
                                   rl, vb, nu=3, delta=delta)
            return trips
        build_rtv.build_trips = _with_triples

    print(f"Loading network...")
    node_df = R.load_or_build_network("32_Phoenix_City")

    # Build ONE scenario
    scenario = R.save_results.final_trips(
        num_requests=args.nr, num_vehicles=args.nv, seed=args.seed)
    requests, vehicles, baseline = R.inject_scenario_globals(node_df, scenario)
    trips = QS.trips
    trip_costs = QS.trip_costs

    print(f"\nInstance: v={args.nv} r={args.nr} seed={args.seed} "
          f"capacity3={args.capacity3}")
    print(f"  trips={len(trips)}  cap_per_request={args.cap_per_request}")

    # ---------- ILP reference (exact) ----------
    print("\n=== ILP (exact reference) ===")
    import io, contextlib
    metadata = {"city": "32_Phoenix_City", "v": args.nv, "r": args.nr}
    stats = R.save_results.summarize_requests(requests)
    ilp_csv = str(out_dir / "_ilp_tmp.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        ilp_solver.classical_ilp_run(metadata, baseline, ilp_csv, stats,
                                      seed=args.seed, trial=1)
    import pandas as pd
    ilp_row = pd.read_csv(ilp_csv)
    ilp_row = ilp_row[ilp_row.run_type == "Classical"].iloc[-1]
    ilp_serv = float(ilp_row["percent_serviced"])
    ilp_vmt = float(ilp_row["vmt"])
    print(f"  ILP: serviced={ilp_serv:.1f}%  VMT={ilp_vmt:.0f}")

    # ---------- Pairwise QUBO sweep over cost_alpha ----------
    print("\n=== Pairwise QUBO: cost_alpha sweep ===")
    print("  (each alpha trades service for route quality)")
    rows = []
    for alpha in args.alphas:
        serv, vmt = _solve_pairwise(QS, trips, trip_costs, requests, vehicles,
                                    ignore_cost=args.lambda_val, cost_alpha=alpha,
                                    cap=args.cap_per_request,
                                    reads=args.reads, sweeps=args.sweeps,
                                    seed=args.seed)
        rows.append({"cost_alpha": alpha, "percent_serviced": serv, "vmt": vmt})
        print(f"  alpha={alpha:>5}: serviced={serv:5.1f}%  VMT={vmt:8.0f}")

    df = pd.DataFrame(rows)
    df["ilp_serviced"] = ilp_serv
    df["ilp_vmt"] = ilp_vmt
    csv_path = out_dir / f"matched_seed{args.seed}_v{args.nv}r{args.nr}.csv"
    df.to_csv(csv_path, index=False)

    # ---------- Honest quality-gap readout ----------
    print("\n=== HONEST QUALITY GAP (VMT at matched service) ===")
    # find the alpha whose service is closest to ILP's
    df["serv_gap"] = (df["percent_serviced"] - ilp_serv).abs()
    best = df.loc[df["serv_gap"].idxmin()]
    print(f"  ILP:            serviced={ilp_serv:.1f}%  VMT={ilp_vmt:.0f}")
    print(f"  QUBO (closest): serviced={best['percent_serviced']:.1f}%  "
          f"VMT={best['vmt']:.0f}  (alpha={best['cost_alpha']})")
    if best["percent_serviced"] > 0:
        ratio = best["vmt"] / ilp_vmt if ilp_vmt else float("nan")
        print(f"  --> at ~matched service, QUBO VMT is {ratio:.2f}x the ILP optimum")
        print(f"      (this is the REAL quality gap -- not the old apples-to-oranges number)")
    print(f"\n  Full curve saved: {csv_path}")
    print(f"  Interpretation: plot percent_serviced (x) vs VMT (y) for the QUBO")
    print(f"  points and mark the ILP point. If the QUBO curve passes NEAR the ILP")
    print(f"  point, quality is competitive. If it sits well above at matched")
    print(f"  service, that's the true (and honest) gap to report.")


def _solve_pairwise(QS, trips, trip_costs, requests, vehicles,
                    ignore_cost, cost_alpha, cap, reads, sweeps, seed):
    """Solve one pairwise-QUBO instance and return (percent_serviced, vmt).
    Mirrors quantum_mwis_run's decode+cleanup+Hungarian but forces the pairwise
    encoding and a cost_alpha-scaled objective."""
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    # minimal per-trip cost
    min_cost = defaultdict(lambda: float("inf"))
    for (tk, vid), c in trip_costs.items():
        if c < min_cost[tk]:
            min_cost[tk] = c

    # Build pairwise QUBO. NOTE: requires generate_qubo_pairwise to accept
    # cost_alpha; if your version lacks it, add cost_alpha*cost in w_t.
    try:
        Q, all_vars = QS.generate_qubo_pairwise(
            trips, trip_costs=min_cost, ignore_cost=ignore_cost,
            M=37500.0, return_numpy=False, seed=seed,
            cap_per_request=cap, cost_alpha=cost_alpha)
    except TypeError:
        # fallback: no cost_alpha kwarg -> pre-scale the costs so cost term grows
        scaled = {tk: cost_alpha * c for tk, c in min_cost.items()}
        Q, all_vars = QS.generate_qubo_pairwise(
            trips, trip_costs=scaled, ignore_cost=ignore_cost,
            M=37500.0, return_numpy=False, seed=seed,
            cap_per_request=cap)

    bitstring = QS.solve_qubo_qiskit_real(Q, reps=2, real=False)
    raw = [all_vars[i] for i, v in bitstring.items() if v == 1]
    sel = [tk for tk in raw if isinstance(tk, frozenset)]
    final, covered = [], set()
    for tk in sel:
        if not (tk & covered):
            final.append(tk); covered |= tk

    # Hungarian vehicle assignment
    Sigma = []
    if final and vehicles:
        INVALID = 1e9
        C = np.full((len(final), len(vehicles)), INVALID)
        for i, tk in enumerate(final):
            for j, v in enumerate(vehicles):
                c = QS.travel_cached(v.id, tk)
                if c is not None:
                    C[i, j] = c
        rr, cc = linear_sum_assignment(C)
        for r_, c_ in zip(rr, cc):
            if C[r_, c_] < INVALID:
                Sigma.append((final[r_], vehicles[c_].id))

    served = set()
    vmt = 0.0
    for trip_ids, vid in Sigma:
        trip_objs = frozenset(QS.request_lookup[r] for r in trip_ids)
        v = QS.vehicle_lookup[vid]
        served |= {r.id for r in trip_objs}
        t = QS.travel(v, trip_objs)
        vmt += float(t)
    serv = 100.0 * len(served) / len(requests) if requests else 0.0
    return serv, vmt


if __name__ == "__main__":
    main()