from __future__ import annotations

import time
from collections import defaultdict

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from scipy.optimize import linear_sum_assignment, minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (
    EstimatorOptions,
    QiskitRuntimeService,
    SamplerOptions,
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
)
from qiskit_optimization import QuadraticProgram

from ..io.save_results import save_metrics_to_csv


def generate_qubo(
    trips,
    trip_costs,
    ignore_cost=10000.0,
    M=20000.0,
    return_numpy=False,
    seed=123,
    cap_per_request=30,
    cap_total_trips=None,
    score_mode="benefit",
):
    rng = np.random.default_rng(seed)

    req_to_trips = defaultdict(list)
    for tkey in trips.keys():
        for r in tkey:
            req_to_trips[r].append(tkey)

    trip_score = {}
    for tkey in trips.keys():
        c = float(trip_costs[tkey])
        benefit = float(ignore_cost) * len(tkey) - c
        trip_score[tkey] = benefit if score_mode == "benefit" else -c

    kept = set()
    if cap_per_request is not None and cap_per_request > 0:
        for r, tlist in req_to_trips.items():
            t_sorted = sorted(tlist, key=lambda tk: trip_score.get(tk, -np.inf), reverse=True)
            kept.update(t_sorted[:cap_per_request])
    else:
        kept = set(trips.keys())

    if cap_total_trips is not None and len(kept) > cap_total_trips:
        kept = set(
            sorted(list(kept), key=lambda tk: trip_score.get(tk, -np.inf), reverse=True)[:cap_total_trips]
        )

    trip_vars = list(kept)
    idx = {t: i for i, t in enumerate(trip_vars)}

    req_to_trip_idxs = defaultdict(list)
    for t in trip_vars:
        ti = idx[t]
        for r in t:
            req_to_trip_idxs[r].append(ti)

    aux_needed = 0
    for _, tlist in req_to_trip_idxs.items():
        k = len(tlist)
        if k > 1:
            aux_needed += (k - 1)

    T = len(trip_vars)
    N = T + aux_needed
    Qdict = defaultdict(float)
    all_vars = list(trip_vars) + [None] * aux_needed
    next_aux = T

    def add_var(label):
        nonlocal next_aux
        all_vars[next_aux] = label
        idx[label] = next_aux
        next_aux += 1
        return idx[label]

    def addQ(i, j, v):
        Qdict[(i, j)] += float(v)
        if i != j:
            Qdict[(j, i)] += float(v)

    def add_square_constraint(u, a, b, weight):
        addQ(u, u, weight)
        addQ(a, a, weight)
        addQ(b, b, weight)
        addQ(u, a, -2 * weight)
        addQ(u, b, -2 * weight)
        addQ(a, b, 2 * weight)

    for t in trip_vars:
        i = idx[t]
        w_t = float(ignore_cost) * len(t) - float(trip_costs[t])
        addQ(i, i, -w_t)

    for r, tlist in req_to_trip_idxs.items():
        if len(tlist) <= 1:
            continue

        current = list(tlist)
        rng.shuffle(current)

        level = 0
        while len(current) > 1:
            nxt = []
            for k in range(0, len(current), 2):
                if k + 1 == len(current):
                    nxt.append(current[k])
                    continue
                a = current[k]
                b = current[k + 1]
                u = add_var(("aux", r, level, k // 2))
                add_square_constraint(u, a, b, M)
                nxt.append(u)
            current = nxt
            level += 1

    if return_numpy:
        Q = np.zeros((N, N), dtype=float)
        for (i, j), v in Qdict.items():
            Q[i, j] = v
        return Q, all_vars

    return dict(Qdict), all_vars


def solve_qubo_qiskit_real(Q, reps=2, max_iter=10, real=False):
    n = Q.shape[0] if isinstance(Q, np.ndarray) else max(max(k) for k in Q.keys()) + 1

    if not real:
        if isinstance(Q, np.ndarray):
            qubo = {}
            for i in range(n):
                for j in range(i, n):
                    if Q[i, j] != 0:
                        qubo[(i, j)] = Q[i, j]
        else:
            qubo = Q

        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(qubo, num_reads=1000)
        best_sample = sampleset.first.sample
        return {i: int(best_sample.get(i, 0)) for i in range(n)}

    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(str(i))

    linear, quadratic = {}, {}
    if isinstance(Q, np.ndarray):
        for i in range(n):
            if Q[i, i] != 0:
                linear[str(i)] = Q[i, i]
            for j in range(i + 1, n):
                if Q[i, j] != 0:
                    quadratic[(str(i), str(j))] = Q[i, j]
    else:
        for (i, j), val in Q.items():
            if i == j:
                linear[str(i)] = val
            else:
                quadratic[(str(i), str(j))] = val

    qp.minimize(linear=linear, quadratic=quadratic)
    hamiltonian, _ = qp.to_ising()

    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)

    ansatz = QAOAAnsatz(hamiltonian, reps=reps)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(ansatz)
    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)

    estimator = Estimator(mode=backend, options=EstimatorOptions(resilience_level=1))

    def cost_func(params, est_obj):
        pub = (isa_circuit, [isa_hamiltonian], [params])
        job = est_obj.run([pub])
        result = job.result()[0]
        return result.data.evs[0]

    init_params = np.random.rand(ansatz.num_parameters) * 2 * np.pi
    actual_max_iter = max(max_iter, 2 * ansatz.num_parameters)

    res = minimize(
        cost_func,
        init_params,
        args=(estimator,),
        method="COBYLA",
        options={"maxiter": actual_max_iter},
    )

    sampler = Sampler(mode=backend, options=SamplerOptions())
    ansatz_measured = ansatz.copy()
    ansatz_measured.measure_all()
    isa_measured = pm.run(ansatz_measured)

    final_job = sampler.run([(isa_measured, [res.x])])
    final_result = final_job.result()[0]

    counts = final_result.data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get).replace(" ", "").replace("_", "")
    reversed_bits = best_bitstring[::-1]

    solution_bits = reversed_bits[:n].ljust(n, "0")
    return {i: int(bit) for i, bit in enumerate(solution_bits)}


def qubo_stats_from_dict(Qd):
    if not Qd:
        return {
            "qubo_vars": 0,
            "qubo_couplers": 0,
            "qubo_graph_density": 0.0,
            "node_density_avg": 0.0,
            "node_density_max": 0.0,
            "node_density_min": 0.0,
            "degree_avg": 0.0,
            "degree_max": 0.0,
            "degree_min": 0.0,
            "qubo_max_abs": 0.0,
            "qubo_min_nonzero_abs": 0.0,
            "qubo_dynamic_range": float("inf"),
            "qubo_logical_qubits": 0,
        }

    max_idx = 0
    max_abs = 0.0
    min_nz = float("inf")
    degrees = defaultdict(int)
    couplers = 0

    for (i, j), v in Qd.items():
        if i > max_idx:
            max_idx = i
        if j > max_idx:
            max_idx = j

        av = abs(float(v))
        if av > 0:
            if av > max_abs:
                max_abs = av
            if av < min_nz:
                min_nz = av

        if i != j and v != 0:
            if i < j:
                couplers += 1
            degrees[i] += 1
            degrees[j] += 1

    n = max_idx + 1
    denom_edges = n * (n - 1) / 2
    graph_density = (couplers / denom_edges) if denom_edges > 0 else 0.0

    deg_arr = np.zeros(n, dtype=float)
    for node, d in degrees.items():
        if 0 <= node < n:
            deg_arr[node] = float(d)

    node_density = (deg_arr / (n - 1)) if n > 1 else np.zeros(n)

    if min_nz == float("inf"):
        min_nz = 0.0
        dyn = float("inf")
    else:
        dyn = (max_abs / min_nz) if min_nz > 0 else float("inf")

    return {
        "qubo_vars": int(n),
        "qubo_couplers": int(couplers),
        "qubo_graph_density": float(graph_density),
        "node_density_avg": float(node_density.mean()) if n else 0.0,
        "node_density_max": float(node_density.max()) if n else 0.0,
        "node_density_min": float(node_density.min()) if n else 0.0,
        "degree_avg": float(deg_arr.mean()) if n else 0.0,
        "degree_max": float(deg_arr.max()) if n else 0.0,
        "degree_min": float(deg_arr.min()) if n else 0.0,
        "qubo_max_abs": float(max_abs),
        "qubo_min_nonzero_abs": float(min_nz),
        "qubo_dynamic_range": float(dyn),
        "qubo_logical_qubits": int(n),
    }


def condition_number(Q_dense, max_n=200):
    n = int(Q_dense.shape[0])
    if n == 0 or n > max_n:
        return np.nan
    s = np.linalg.svd(Q_dense, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] != 0 else float("inf")


def quantum_mwis_run(
    metadata,
    baseline_vehicle_costs,
    requests,
    vehicles,
    request_lookup,
    vehicle_lookup,
    trips,
    trip_to_vehicle,
    trip_costs,
    node_df,
    rtv_graph_build_time,
    travel_fn,
    csv_filename="results.csv",
    request_stats=None,
    real=False,
    seed=123,
    trial=1,
):
    if request_stats is None:
        request_stats = {}

    run_start = time.perf_counter()

    t_min_cost_prep_start = time.perf_counter()
    minimal_qubo_trip_costs = defaultdict(lambda: float("inf"))
    for (tkey, vid), cost in trip_costs.items():
        if cost < minimal_qubo_trip_costs[tkey]:
            minimal_qubo_trip_costs[tkey] = cost
    t_min_cost_prep_end = time.perf_counter()

    t_qubo_gen_start = time.perf_counter()
    Q_matrix, all_vars = generate_qubo(
        trips,
        trip_costs=minimal_qubo_trip_costs,
        return_numpy=False,
        seed=seed,
    )
    t_qubo_gen_end = time.perf_counter()

    t_quantum_solve_start = time.perf_counter()
    quantum_bitstring = solve_qubo_qiskit_real(Q_matrix, reps=2, real=real)
    t_quantum_solve_end = time.perf_counter()

    t_decode_start = time.perf_counter()
    raw_selected_trips = [all_vars[i] for i, v in quantum_bitstring.items() if v == 1]
    selected_trip_keys = [tk for tk in raw_selected_trips if isinstance(tk, frozenset)]

    final_trips_list = []
    covered = set()
    for tk in selected_trip_keys:
        if not (tk & covered):
            final_trips_list.append(tk)
            covered |= tk
    t_decode_end = time.perf_counter()

    t_vehicle_assign_start = time.perf_counter()
    Sigma_opt = []
    if final_trips_list and vehicles:
        INVALID = 1e9
        C = np.full((len(final_trips_list), len(vehicles)), INVALID, dtype=float)
        for i, tkey in enumerate(final_trips_list):
            for j, v in enumerate(vehicles):
                cost = trip_costs.get((tkey, v.id), None)
                if cost is not None:
                    C[i, j] = cost
        rows, cols = linear_sum_assignment(C)
        for r, c in zip(rows, cols):
            if C[r, c] < INVALID:
                Sigma_opt.append((final_trips_list[r], vehicles[c].id))
    t_vehicle_assign_end = time.perf_counter()

    t_metrics_calc_start = time.perf_counter()
    served_requests = set()
    served_trips = []
    for trip_ids, vid in Sigma_opt:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        served_trips.append((trip_objs, vehicle_lookup[vid]))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = 100.0 * len(served_requests) / len(requests) if requests else 0.0

    detours, waiting, vmt = [], [], 0.0
    for trip, v in served_trips:
        total_time = travel_fn(v, trip)
        shortest = sum(r.t_star for r in trip)
        detours.append(total_time / shortest if shortest > 0 else 1.0)
        vmt += float(total_time)
        _, pickups = travel_fn(v, trip, return_timeline=True)
        for r in trip:
            waiting.append(pickups[r] - r.trr)

    avg_detour = float(np.mean(detours)) if detours else 0.0
    max_detour = float(np.max(detours)) if detours else 0.0
    avg_wait = float(np.mean(waiting)) if waiting else 0.0
    max_wait = float(np.max(waiting)) if waiting else 0.0
    t_metrics_calc_end = time.perf_counter()

    t_struct_stats_start = time.perf_counter()
    base_stats = qubo_stats_from_dict(Q_matrix)
    qstats = {f"base_{k}": v for k, v in base_stats.items()}

    energygap = np.nan
    conditionnumber = np.nan
    t_struct_stats_end = time.perf_counter()

    run_end = time.perf_counter()

    metrics_to_save = {
        "city": metadata.get("city", ""),
        "run_type": "Quantum",
        "nodes": node_df["node_id"].nunique(),
        "num_vehicles": len(vehicles),
        "num_requests": len(requests),
        "percent_serviced": percent_serviced,
        "avg_waiting_time": avg_wait,
        "max_waiting_time": max_wait,
        "avg_detour_factor": avg_detour,
        "max_detour_factor": max_detour,
        "vmt": vmt,
        "total_run_time": run_end - run_start,
        "solve_time": t_quantum_solve_end - t_quantum_solve_start,
        "rtv_graph_build_time": rtv_graph_build_time,
        "qubo_build_time": (t_min_cost_prep_end - t_min_cost_prep_start) + (t_qubo_gen_end - t_qubo_gen_start),
        "time_min_cost_prep": t_min_cost_prep_end - t_min_cost_prep_start,
        "time_qubo_gen": t_qubo_gen_end - t_qubo_gen_start,
        "time_quantum_solve": t_quantum_solve_end - t_quantum_solve_start,
        "time_decode": t_decode_end - t_decode_start,
        "time_vehicle_assignment": t_vehicle_assign_end - t_vehicle_assign_start,
        "time_metrics_calc": t_metrics_calc_end - t_metrics_calc_start,
        "time_struct_stats": t_struct_stats_end - t_struct_stats_start,
        "time_total_quantum_block": run_end - run_start,
        "energy_gap": float(energygap),
        "condition_number": float(conditionnumber),
        "real_quantum_hardware": 1 if real else 0,
        "seed": seed,
        "trial": trial,
    }

    for k, v in request_stats.items():
        if k not in metrics_to_save:
            metrics_to_save[k] = v

    metrics_to_save.update(qstats)

    save_metrics_to_csv(csv_filename, metrics_to_save)
    print("\n--- Quantum MWIS Results Saved ---")
    print("Percent Serviced:", percent_serviced)

    return Sigma_opt