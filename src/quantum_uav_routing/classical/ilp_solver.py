import pulp
import time
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..io.save_results import save_metrics_to_csv

# Wall-clock limit for CBC on each ILP instance. When the limit is reached, we
# report the best incumbent found so far and the optimality gap at cutoff.
DEFAULT_ILP_TIME_LIMIT_S = 2 * 3600.0  # 2 hours

def ilp_model_stats(model):
    vars_ = model.variables()
    num_vars = len(vars_)
    num_constraints = len(model.constraints)

    # Type counts (PuLP categories: "Binary", "Integer", "Continuous")
    num_bin = sum(1 for v in vars_ if v.cat == "Binary")
    num_int = sum(1 for v in vars_ if v.cat == "Integer")
    num_cont = sum(1 for v in vars_ if v.cat == "Continuous")

    # Nonzeros in constraint matrix (approx): count variable appearances in constraints
    # This uses PuLP's internal constraint representation.
    nnz = 0
    for cname, c in model.constraints.items():
        # c is an LpConstraint, c.items() gives (LpVariable, coeff)
        nnz += len(c.items())

    return {
        "ilp_num_vars": num_vars,
        "ilp_num_constraints": num_constraints,
        "ilp_num_binary_vars": num_bin,
        "ilp_num_integer_vars": num_int,
        "ilp_num_continuous_vars": num_cont,
        "ilp_num_nonzero_coeffs": nnz,
    }


def _parse_cbc_log(log_path: str | None) -> dict:
    """Extract incumbent / bound / time-limit flags from a CBC log file."""
    out = {
        "ilp_time_limit_hit": 0,
        "ilp_best_bound": float("nan"),
        "ilp_log_incumbent_objective": float("nan"),
    }
    if not log_path or not os.path.isfile(log_path):
        return out

    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out

    if re.search(
        r"(stopped on time limit|maximum time|time limit exceeded|exiting on maximum)",
        text,
        re.I,
    ):
        out["ilp_time_limit_hit"] = 1

    float_pat = r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    incumbent_patterns = [
        rf"Integer solution of {float_pat} found",
        rf"best objective {float_pat}",
        rf"Objective value:\s+{float_pat}",
    ]
    bound_patterns = [
        rf"best possible (?:solution )?(?:is )?{float_pat}",
        rf"Best possible\s+{float_pat}",
        rf"Lower bound:\s+{float_pat}",
    ]

    for pat in incumbent_patterns:
        matches = re.findall(pat, text, re.I)
        if matches:
            out["ilp_log_incumbent_objective"] = float(matches[-1])

    for pat in bound_patterns:
        matches = re.findall(pat, text, re.I)
        if matches:
            out["ilp_best_bound"] = float(matches[-1])

    return out


def _compute_optimality_gap(incumbent: float | None, bound: float | None) -> float:
    if incumbent is None or bound is None:
        return float("nan")
    if math.isnan(incumbent) or math.isnan(bound):
        return float("nan")
    denom = max(abs(incumbent), 1e-9)
    return max(0.0, (incumbent - bound) / denom)


def _extract_assignment(epsilon, chi):
    Sigma_opt = [
        (t_id, v_id)
        for (t_id, v_id), var in epsilon.items()
        if var.value() is not None and var.value() > 0.5
    ]
    ignored_requests = [
        r_id
        for r_id, var in chi.items()
        if var.value() is not None and var.value() > 0.5
    ]
    return Sigma_opt, ignored_requests


def _assignment_is_complete(Sigma_opt, ignored_requests, requests, trips):
    """True if every request is served exactly once or explicitly ignored."""
    served = set()
    for t_id, _v_id in Sigma_opt:
        served |= set(trips[t_id])
    ignored = set(ignored_requests)
    all_requests = set(requests)
    return (
        served.isdisjoint(ignored)
        and served | ignored == all_requests
    )


def _ilp_solve_diagnostics(
    model,
    log_info: dict,
    solve_time_s: float,
    time_limit_s: float,
) -> dict:
    status_code = model.status
    sol_status_code = getattr(model, "sol_status", None)
    status_text = pulp.LpStatus.get(status_code, str(status_code))
    sol_status_text = (
        pulp.LpSolution.get(sol_status_code, str(sol_status_code))
        if sol_status_code is not None
        else None
    )

    incumbent = pulp.value(model.objective)
    if incumbent is None and not math.isnan(log_info["ilp_log_incumbent_objective"]):
        incumbent = log_info["ilp_log_incumbent_objective"]

    best_bound = getattr(model, "bestBound", None)
    if best_bound is None or (
        isinstance(best_bound, float) and math.isnan(best_bound)
    ):
        best_bound = log_info["ilp_best_bound"]
    if isinstance(best_bound, float) and math.isnan(best_bound):
        best_bound = None

    optimal = status_code == pulp.LpStatusOptimal
    has_incumbent = sol_status_code in (
        pulp.LpSolutionOptimal,
        pulp.LpSolutionIntegerFeasible,
    ) or incumbent is not None

    time_limit_hit = bool(log_info["ilp_time_limit_hit"])
    if (
        not optimal
        and time_limit_s is not None
        and time_limit_s > 0
        and solve_time_s >= 0.99 * time_limit_s
    ):
        time_limit_hit = True

    gap = 0.0 if optimal else _compute_optimality_gap(incumbent, best_bound)

    return {
        "ilp_status": status_text,
        "ilp_sol_status": sol_status_text,
        "ilp_optimal": int(optimal),
        "ilp_has_incumbent": int(has_incumbent),
        "ilp_time_limit_s": float(time_limit_s) if time_limit_s is not None else None,
        "ilp_time_limit_hit": int(time_limit_hit),
        "ilp_incumbent_objective": (
            float(incumbent) if incumbent is not None else float("nan")
        ),
        "ilp_best_bound": (
            float(best_bound) if best_bound is not None else float("nan")
        ),
        "ilp_optimality_gap": float(gap) if not math.isnan(gap) else float("nan"),
        "ilp_used_warm_start_fallback": 0,
    }

def optimal_assignment(
    requests,
    vehicles,
    trips,
    trip_costs,
    trip_to_vehicle,
    baseline_vehicle_costs,
    ignore_cost=10000,
    time_limit_s=DEFAULT_ILP_TIME_LIMIT_S,
):
    """
    PNAS-style ILP for trip-vehicle assignment with a GREEDY WARM START,
    not a fixed greedy solution.

    Returns
    -------
    Sigma_opt : list of (t_id, v_id)
    ignored_requests : list of request IDs
    stats : dict
    """
    # -------------------------------------------------
    # 1. Create ILP model
    # -------------------------------------------------
    model = pulp.LpProblem("OptimalAssignment", pulp.LpMinimize)

    # -------------------------------------------------
    # 2. Decision variables
    # ε_(i,j) and χ_k
    # -------------------------------------------------
    epsilon = {}
    for t_id in trips:
        for v_id in trip_to_vehicle[t_id]:
            if (t_id, v_id) in trip_costs:
                epsilon[(t_id, v_id)] = pulp.LpVariable(
                    f"eps_{t_id}_{v_id}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpBinary
                )

    chi = {
        r_id: pulp.LpVariable(
            f"chi_{r_id}",
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary
        )
        for r_id in requests
    }

    # -------------------------------------------------
    # 3. Objective
    # incremental delay + ignore penalty
    # -------------------------------------------------
    model += (
        pulp.lpSum(
            (trip_costs[(t_id, v_id)] - baseline_vehicle_costs[v_id])
            * epsilon[(t_id, v_id)]
            for (t_id, v_id) in epsilon
        )
        +
        pulp.lpSum(ignore_cost * chi[r_id] for r_id in requests)
    )

    # -------------------------------------------------
    # 4. Constraints
    # -------------------------------------------------

    # (8) Each vehicle assigned at most one trip
    for v_id in vehicles:
        model += (
            pulp.lpSum(
                epsilon[(t_id, v_id)]
                for (t_id, vv) in epsilon
                if vv == v_id
            ) <= 1
        )

    # (9) Each request served exactly once or ignored
    for r_id in requests:
        model += (
            pulp.lpSum(
                epsilon[(t_id, v_id)]
                for (t_id, v_id) in epsilon
                if r_id in trips[t_id]
            ) + chi[r_id] == 1
        )

    # -------------------------------------------------
    # 5. Greedy warm start (Algorithm 2 idea)
    # IMPORTANT: warm start only, DO NOT fix variables
    # -------------------------------------------------

    # Initialize all variables to zero
    for var in epsilon.values():
        var.setInitialValue(0)

    for var in chi.values():
        var.setInitialValue(0)

    sorted_eps = sorted(
        epsilon.keys(),
        key=lambda x: (
            -len(trips[x[0]]),   # trip size descending
            trip_costs[x]        # cost ascending
        )
    )

    assigned_requests = set()
    assigned_vehicles = set()
    warm_Sigma_opt = []
    warm_ignored = []

    for (t_id, v_id) in sorted_eps:
        if v_id in assigned_vehicles:
            continue
        if any(r in assigned_requests for r in trips[t_id]):
            continue

        # Warm start only
        epsilon[(t_id, v_id)].setInitialValue(1)
        warm_Sigma_opt.append((t_id, v_id))

        assigned_vehicles.add(v_id)
        assigned_requests |= trips[t_id]

    for r_id in requests:
        if r_id not in assigned_requests:
            chi[r_id].setInitialValue(1)
            warm_ignored.append(r_id)

    # -------------------------------------------------
    # 6. Solve (with wall-clock limit; report incumbent at cutoff)
    # -------------------------------------------------
    log_path = None
    log_info = {
        "ilp_time_limit_hit": 0,
        "ilp_best_bound": float("nan"),
        "ilp_log_incumbent_objective": float("nan"),
    }
    solve_time_s = float("nan")
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cbc.log", delete=False
        ) as log_file:
            log_path = log_file.name

        # CBC on Windows requires keepFiles=True when warmStart=True.
        solver = pulp.PULP_CBC_CMD(
            msg=False,
            warmStart=True,
            keepFiles=os.name == "nt",
            timeLimit=time_limit_s,
            logPath=log_path,
        )
        t_solve_start = time.perf_counter()
        model.solve(solver)
        solve_time_s = time.perf_counter() - t_solve_start
        log_info = _parse_cbc_log(log_path)
    finally:
        if log_path and os.path.isfile(log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass

    # -------------------------------------------------
    # 7. Extract solution (incumbent if time-limited; else warm start)
    # -------------------------------------------------
    Sigma_opt, ignored_requests = _extract_assignment(epsilon, chi)
    if not _assignment_is_complete(Sigma_opt, ignored_requests, requests, trips):
        Sigma_opt, ignored_requests = warm_Sigma_opt, warm_ignored
        used_fallback = True
    else:
        used_fallback = False

    stats = ilp_model_stats(model)
    solve_stats = _ilp_solve_diagnostics(
        model,
        log_info,
        solve_time_s=solve_time_s,
        time_limit_s=time_limit_s,
    )
    if solve_stats["ilp_optimal"]:
        solve_stats["ilp_optimality_gap"] = 0.0
    elif not math.isnan(solve_stats["ilp_incumbent_objective"]):
        solve_stats["ilp_optimality_gap"] = _compute_optimality_gap(
            solve_stats["ilp_incumbent_objective"],
            solve_stats["ilp_best_bound"],
        )
    solve_stats["ilp_solve_time_limit_s"] = float(time_limit_s)
    solve_stats["ilp_solve_wall_time_s"] = float(solve_time_s)
    solve_stats["ilp_used_warm_start_fallback"] = int(used_fallback)
    stats.update(solve_stats)

    return Sigma_opt, ignored_requests, stats


def classical_ilp_run(
    metadata,
    baseline_vehicle_costs,
    csv_filename="results.csv",
    stats={},
    seed=123,
    trial=1,
    time_limit_s=DEFAULT_ILP_TIME_LIMIT_S,
):
    """
    Classical ILP run with clean timing semantics:

    - total_run_time: end-to-end time inside THIS function (solver pipeline only)
    - solve_time: time spent inside optimal_assignment() (ILP solve/model time)
    - rtv_graph_build_time: scenario precompute time from final_trips (t1 - t0), if available
    - scenario_total_time: scenario pipeline time (if provided via stats), optional
    """

    # Per-solver wall time (does NOT include scenario generation unless you call this inside it)
    run_start = time.perf_counter()

    # ----------------------------
    # ILP SOLVE
    # ----------------------------
    t2 = time.perf_counter()

    Sigma_opt, ignored, ilp_stats = optimal_assignment(
        requests=[r.id for r in requests],
        vehicles=[v.id for v in vehicles],
        trips=trips,
        trip_costs=trip_costs,
        trip_to_vehicle=trip_to_vehicle,
        baseline_vehicle_costs=baseline_vehicle_costs,
        ignore_cost=10000,
        time_limit_s=time_limit_s,
    )

    t3 = time.perf_counter()

    # ----------------------------
    # POST-PROCESSING
    # ----------------------------
    served_requests = set()
    served_trips = []

    for trip_ids, vid in Sigma_opt:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        v_obj = vehicle_lookup[vid]
        served_trips.append((trip_objs, v_obj))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = 100.0 * len(served_requests) / len(requests) if requests else 0.0

    # ----------------------------
    # METRICS (RAW)
    # ----------------------------
    detours = []
    waiting = []
    VMT = 0.0

    for trip, v in served_trips:
        total_time = travel(v, trip)
        shortest = sum(r.t_star for r in trip)
        detours.append(total_time / shortest if shortest > 0 else 1.0)
        VMT += float(total_time)

        _, pickups = travel(v, trip, return_timeline=True)
        for r in trip:
            waiting.append(pickups[r] - r.trr)

    avg_detour = float(sum(detours) / len(detours)) if detours else 0.0
    max_detour = float(max(detours)) if detours else 0.0
    avg_wait = float(sum(waiting) / len(waiting)) if waiting else 0.0
    max_wait = float(max(waiting)) if waiting else 0.0

    run_end = time.perf_counter()

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print("Number of nodes in the graph:", node_df["node_id"].nunique())
    print("Number of vehicles:", len(vehicles))
    print("Number of requests:", len(requests))
    print(f"% Serviced Requests: {percent_serviced:.2f}%")

    print("Optimal Assignment:")
    for trip, v in served_trips:
        print(" ", trip_name(trip), "→", v)

    print("\nIgnored Requests:", ignored)

    print("\n--- Performance Metrics ---")
    print(f"Average Waiting Time: {avg_wait:.2f} s")
    print(f"Maximum Waiting Time: {max_wait:.2f} s")
    print(f"Average Detour Factor: {avg_detour:.3f}")
    print(f"Maximum Detour Factor: {max_detour:.3f}")
    print(f"Vehicle Miles Traveled (VMT): {VMT:.2f}")

    # IMPORTANT: do not report "t3 - t0" as solver runtime; that's scenario-cumulative.
    print(f"Total Run Time (solver only): {run_end - run_start:.9f} s")
    print(f"ILP Solve Time:              {t3 - t2:.9f} s")
    print(
        f"ILP status: {ilp_stats.get('ilp_status')} | "
        f"sol_status: {ilp_stats.get('ilp_sol_status')} | "
        f"optimal: {ilp_stats.get('ilp_optimal')} | "
        f"time_limit_hit: {ilp_stats.get('ilp_time_limit_hit')}"
    )
    if ilp_stats.get("ilp_incumbent_objective") is not None and not math.isnan(
        ilp_stats.get("ilp_incumbent_objective", float("nan"))
    ):
        print(
            f"ILP incumbent objective: {ilp_stats['ilp_incumbent_objective']:.6f} | "
            f"best bound: {ilp_stats.get('ilp_best_bound')} | "
            f"gap: {ilp_stats.get('ilp_optimality_gap')}"
        )
    if ilp_stats.get("ilp_used_warm_start_fallback"):
        print("ILP used greedy warm-start fallback (no usable CBC incumbent).")

    if "t0" in globals() and "t1" in globals():
        print(f"RTV Graph Build Time:        {t1 - t0:.2f} s")

    # ----------------------------
    # SAVE METRICS
    # ----------------------------
    metrics_to_save = {
        "city": metadata.get("city", ""),
        "run_type": "Classical",
        "nodes": node_df["node_id"].nunique(),
        "num_vehicles": len(vehicles),
        "num_requests": len(requests),

        "percent_serviced": percent_serviced,
        "avg_waiting_time": avg_wait,
        "max_waiting_time": max_wait,
        "avg_detour_factor": avg_detour,
        "max_detour_factor": max_detour,
        "vmt": VMT,

        # Clean timing semantics:
        "total_run_time": run_end - run_start,   # solver pipeline wall time
        "solve_time": t3 - t2,                   # ILP solver/model call time

        # Scenario build time (if available from final_trips):
        "rtv_graph_build_time": (t1 - t0) if ("t0" in globals() and "t1" in globals()) else None,

        "seed": seed,
        "trial": trial,
    }

    # Add scenario-level time if you computed it in the outer loop and stuffed into stats
    # (recommended): stats["scenario_total_time"] = scenario_end - scenario_start

    metrics_to_save.update(stats)
    metrics_to_save.update(ilp_stats)

    save_metrics_to_csv(csv_filename, metrics_to_save)
    return Sigma_opt, ignored