
from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from ..rtv.trip_builder import (
    travel,
    request_lookup,
    vehicle_lookup,
    requests,
    vehicles,
    trips,
    trip_costs,
    trip_to_vehicle,
    node_df,
)
from ..io.save_results import save_metrics_to_csv, CSV_COLUMNS


def configure_runtime(**kwargs):
    from ..rtv import trip_builder
    trip_builder.configure_runtime(**kwargs)

def generate_qubo(
    trips,
    trip_costs,
    ignore_cost=10000.0,
    M=20000.0,
    return_numpy=True,
    seed=123,
    # --- NEW: hard bounds that make QUBO gen < n^2 (for constant caps) ---
    cap_per_request=30,     # keep at most this many trip-vars touching each request
    cap_total_trips=None,   # optional global cap on number of trip variables
    score_mode="benefit",   # "benefit" or "cost"
):
    """
    BOUNDED QUBO GENERATION (drop-in replacement).

    Key idea:
      - Bound # trip variables by pruning candidate trips BEFORE building Q.
      - Keep at most `cap_per_request` trips incident to each request (by score).
      - Optional `cap_total_trips` for a global ceiling.
    This makes variable count and couplers O(n * cap_per_request), i.e. < n^2 for constant cap.

    Model:
      - Trip vars x_t (one per retained trip)
      - For each request r: enforce sum_{t contains r} x_t <= 1 using a binary-tree of aux vars
        with penalties M*(u - a - b)^2 (O(k_r) terms, not clique)

    Objective:
      maximize benefit w_t = ignore_cost*|t| - trip_costs[t]
      QUBO minimization uses diagonal -w_t
    """

    rng = np.random.default_rng(seed)

    # -------------------------
    # 0) Build request -> trips
    # -------------------------
    # We do this once, then bound.
    req_to_trips = defaultdict(list)  # r -> list of trip keys
    for tkey in trips.keys():
        for r in tkey:
            req_to_trips[r].append(tkey)

    # -----------------------------------------
    # 1) Score trips and apply per-request bound
    # -----------------------------------------
    # Score definition (you can change, but keep it deterministic):
    # - "benefit": higher is better
    # - "cost": lower cost is better
    trip_score = {}
    for tkey in trips.keys():
        c = float(trip_costs[tkey])
        benefit = float(ignore_cost) * len(tkey) - c
        trip_score[tkey] = benefit if score_mode == "benefit" else -c

    kept = set()

    # For each request, keep top cap_per_request incident trips by score
    if cap_per_request is not None and cap_per_request > 0:
        for r, tlist in req_to_trips.items():
            # sort by score descending
            t_sorted = sorted(tlist, key=lambda tk: trip_score.get(tk, -np.inf), reverse=True)
            kept.update(t_sorted[:cap_per_request])
    else:
        kept = set(trips.keys())

    # Optional global cap
    if cap_total_trips is not None and len(kept) > cap_total_trips:
        kept = set(sorted(list(kept), key=lambda tk: trip_score.get(tk, -np.inf), reverse=True)[:cap_total_trips])

    trip_vars = list(kept)
    T = len(trip_vars)

    # Rebuild request adjacency but only for retained trips
    req_to_trip_idxs = defaultdict(list)
    idx = {t: i for i, t in enumerate(trip_vars)}
    for t in trip_vars:
        ti = idx[t]
        for r in t:
            req_to_trip_idxs[r].append(ti)

    # --------------------------------------------
    # 2) Compute aux var budget (tree => k_r - 1)
    # --------------------------------------------
    aux_needed = 0
    for r, tlist in req_to_trip_idxs.items():
        k = len(tlist)
        if k > 1:
            aux_needed += (k - 1)

    N = T + aux_needed

    # -------------------------
    # 3) Allocate Q (sparse-first)
    # -------------------------
    # Using a dict is MUCH faster/leaner when N grows, and avoids O(N^2) init cost.
    Qdict = defaultdict(float)

    all_vars = list(trip_vars) + [None] * aux_needed
    next_aux = T

    def add_var(label):
        nonlocal next_aux
        if next_aux >= N:
            raise RuntimeError("Aux overflow: aux_needed miscomputed.")
        all_vars[next_aux] = label
        idx[label] = next_aux
        next_aux += 1
        return idx[label]

    def addQ(i, j, v):
        # store symmetric in dict (your downstream code sometimes assumes full dict)
        Qdict[(i, j)] += float(v)
        if i != j:
            Qdict[(j, i)] += float(v)

    def add_square_constraint(u, a, b, weight):
        """
        Add weight*(u - a - b)^2 to Q.
        Expansion: weight*(u + a + b - 2ua - 2ub + 2ab)
        """
        addQ(u, u, weight)
        addQ(a, a, weight)
        addQ(b, b, weight)

        addQ(u, a, -2 * weight)
        addQ(u, b, -2 * weight)
        addQ(a, b,  2 * weight)

    # -------------------------
    # 4) Objective on diagonal
    # -------------------------
    for t in trip_vars:
        i = idx[t]
        w_t = float(ignore_cost) * len(t) - float(trip_costs[t])
        addQ(i, i, -w_t)

    # --------------------------------------------------------
    # 5) For each request: sum(x in S_r) <= 1 via merge tree
    # --------------------------------------------------------
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

    if next_aux != N:
        raise RuntimeError(f"Aux mismatch: predicted {aux_needed}, used {next_aux - T}")

    # -------------------------
    # 6) Return format
    # -------------------------
    if not return_numpy:
        return dict(Qdict), all_vars

    # Dense convert only at the end (still O(N^2), so keep N small via caps)
    Q = np.zeros((N, N), dtype=float)
    for (i, j), v in Qdict.items():
        Q[i, j] = v
    return Q, all_vars

import numpy as np

from scipy.optimize import minimize

import dimod

from dwave.samplers import SimulatedAnnealingSampler

from qiskit.circuit.library import QAOAAnsatz

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    EstimatorV2 as Estimator,
    SamplerV2 as Sampler,
    EstimatorOptions,
    SamplerOptions
)

from qiskit_optimization import QuadraticProgram

def solve_qubo_qiskit_real(Q, reps=2, max_iter=10, real=False):
    """
    Hybrid Solver Interface:
    - real=False: Uses D-Wave Simulated Annealing (Fast, Low Memory)
    - real=True:  Uses IBM Quantum Hardware (QAOA V2 Primitives)
    """
    # --- 1. Determine QUBO size ---
    n = Q.shape[0] if isinstance(Q, np.ndarray) else max(max(k) for k in Q.keys()) + 1

    # ---------------------------------------------------------
    # MODE A: SIMULATED ANNEALING (LOCAL / TESTING)
    # ---------------------------------------------------------
    if not real:
        print(f"Mode: LOCAL SIMULATION (Simulated Annealing) | Size: {n} variables")

        # Convert Q to dict format for dimod
        if isinstance(Q, np.ndarray):
            qubo = {}
            for i in range(n):
                for j in range(i, n):
                    if Q[i, j] != 0: qubo[(i, j)] = Q[i, j]
        else:
            qubo = Q

        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(qubo, num_reads=1000)

        best_sample = sampleset.first.sample
        # Ensure mapping consistency for all indices
        return {i: int(best_sample.get(i, 0)) for i in range(n)}

    # ---------------------------------------------------------
    # MODE B: IBM QUANTUM HARDWARE (V2 RUNTIME)
    # ---------------------------------------------------------
    print(f"Mode: REAL HARDWARE (IBM Quantum) | Size: {n} variables")

    # 1. Build QuadraticProgram
    qp = QuadraticProgram()
    for i in range(n): qp.binary_var(str(i))

    linear, quadratic = {}, {}
    if isinstance(Q, np.ndarray):
        for i in range(n):
            if Q[i, i] != 0: linear[str(i)] = Q[i, i]
            for j in range(i + 1, n):
                if Q[i, j] != 0: quadratic[(str(i), str(j))] = Q[i, j]
    else:
        for (i, j), val in Q.items():
            if i == j: linear[str(i)] = val
            else: quadratic[(str(i), str(j))] = val

    qp.minimize(linear=linear, quadratic=quadratic)
    hamiltonian, offset = qp.to_ising()

    # 2. Setup Backend
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    print(f"Using IBM backend: {backend.name}")

    # 3. Create and Transpile Circuit
    ansatz = QAOAAnsatz(hamiltonian, reps=reps)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(ansatz)
    isa_hamiltonian = hamiltonian.apply_layout(isa_circuit.layout)

    # 4. Optimization Loop
    estimator = Estimator(mode=backend, options=EstimatorOptions(resilience_level=1))

    def cost_func(params, est_obj):
        pub = (isa_circuit, [isa_hamiltonian], [params])
        job = est_obj.run([pub])
        result = job.result()[0]
        return result.data.evs[0]

    init_params = np.random.rand(ansatz.num_parameters) * 2 * np.pi
    actual_max_iter = max(max_iter, 2 * ansatz.num_parameters)

    res = minimize(
        cost_func, init_params, args=(estimator,),
        method="COBYLA", options={'maxiter': actual_max_iter}
    )

    # 5. Final Sampling
    sampler = Sampler(mode=backend, options=SamplerOptions())
    ansatz_measured = ansatz.copy()
    ansatz_measured.measure_all()
    isa_measured = pm.run(ansatz_measured)

    final_job = sampler.run([(isa_measured, [res.x])])
    final_result = final_job.result()[0]

    # 6. Extract Bitstring
    counts = final_result.data.meas.get_counts()
    best_bitstring = max(counts, key=counts.get).replace(" ", "").replace("_", "")
    reversed_bits = best_bitstring[::-1]

    solution_bits = reversed_bits[:n].ljust(n, '0')
    return {i: int(bit) for i, bit in enumerate(solution_bits)}

import numpy as np

import dimod

from dwave.samplers import SimulatedAnnealingSampler

def get_subspace_bounds(Q, k, l, a, b):
    """
    Computes lower and upper bounds for the subspace y*_ab
    where x_k=a and x_l=b.
    """
    n = Q.shape[0]
    x_fixed = np.zeros(n)
    x_fixed[k], x_fixed[l] = a, b

    # Upper Bound: Specific configuration (current state)
    y_hat = x_fixed.T @ Q @ x_fixed

    # Lower Bound: Sum of all potential negative contributions in this subspace
    Q_neg = np.minimum(Q, 0)
    x_ones = np.ones(n)
    x_ones[k], x_ones[l] = a, b
    y_tilde = x_ones.T @ Q_neg @ x_ones

    return y_tilde, y_hat

from collections import defaultdict

import heapq

import numpy as np

def compress_qubo_fast_topk(
    Q,
    target_max=1.0,
    K=20,
    keep_diagonal=True,
    min_abs_keep=0.0,
    rescale_after=True,
    return_stats=False,
):
    def to_upper_dict(Q_in):
        if isinstance(Q_in, dict):
            out = {}
            for (i, j), c in Q_in.items():
                if c == 0:
                    continue
                i = int(i); j = int(j); c = float(c)
                if i <= j:
                    out[(i, j)] = out.get((i, j), 0.0) + c
                else:
                    out[(j, i)] = out.get((j, i), 0.0) + c
            return out

        if isinstance(Q_in, np.ndarray):
            # FAST: grab nonzeros from upper triangle (no Python N^2 scan)
            Qtri = np.triu(Q_in)
            ii, jj = np.nonzero(Qtri)
            out = {(int(i), int(j)): float(Qtri[i, j]) for i, j in zip(ii, jj)}
            return out

        raise TypeError("Q must be dict or numpy ndarray")

    def scale_to_target(Qd, tgt):
        if not Qd:
            return Qd, 1.0
        m = max(abs(c) for c in Qd.values())
        if m == 0:
            return Qd, 1.0
        s = float(tgt) / m
        if s == 1.0:
            return Qd, 1.0
        return {k: v * s for k, v in Qd.items()}, s

    Qd0 = to_upper_dict(Q)
    Qd, s1 = scale_to_target(Qd0, target_max)

    out = {}
    heaps = defaultdict(list)

    for (i, j), c in Qd.items():
        if i == j:
            if keep_diagonal and abs(c) >= min_abs_keep:
                out[(i, j)] = c
            continue

        a = abs(c)
        if a < min_abs_keep:
            continue

        hi = heaps[i]
        if len(hi) < K:
            heapq.heappush(hi, (a, i, j))
        elif a > hi[0][0]:
            heapq.heapreplace(hi, (a, i, j))

        hj = heaps[j]
        if len(hj) < K:
            heapq.heappush(hj, (a, i, j))
        elif a > hj[0][0]:
            heapq.heapreplace(hj, (a, i, j))

    keep_edges = set()
    for h in heaps.values():
        for _, i, j in h:
            keep_edges.add((i, j))

    for (i, j), c in Qd.items():
        if i != j and (i, j) in keep_edges:
            out[(i, j)] = c

    if rescale_after:
        out, s2 = scale_to_target(out, target_max)
    else:
        s2 = 1.0

    if not return_stats:
        return out

    n_vars = (max(max(i, j) for (i, j) in out.keys()) + 1) if out else 0
    n_couplers = sum(1 for (i, j) in out.keys() if i != j)
    n_diag = sum(1 for (i, j) in out.keys() if i == j)

    return out, {
        "K": K,
        "scale_factor_1": s1,
        "scale_factor_2": s2,
        "n_vars": n_vars,
        "n_diag": n_diag,
        "n_couplers": n_couplers,
    }

import random

import numpy as np

def generate_requests_and_vehicles(
    num_requests,
    num_vehicles,
    node2d_to_3d,
    shortest_path_cached,
    node_df, # New argument
    T_max = 3*3600,
    seed=42,
    min_slack=1200,
    max_slack=4800,
):
    random.seed(seed)
    np.random.seed(seed)

    nodes_2d = list(node2d_to_3d.keys())
    # Create a map for 2D node IDs to their coordinates
    node_coords_map = node_df.set_index('node_id')[['x_coord', 'y_coord']].to_dict('index')


    # Vehicles
    vehicles = [
        Vehicle(
            random.choice(nodes_2d),
            random.randint(0, T_max // 2),
            [],
            vid + 1
        )
        for vid in range(num_vehicles)
    ]

    # Requests
    requests = []
    for rid in range(1, num_requests + 1):
        origin, destination = random.sample(nodes_2d, 2)

        # Get coordinates for the origin
        origin_coords = node_coords_map.get(origin)
        if origin_coords is None:
            # This case implies an origin node was chosen that isn't in node_df, which shouldn't happen
            # if nodes_2d is derived from node_df. If it does, skipping this request.
            continue

        # Get coordinates for the destination
        destination_coords = node_coords_map.get(destination)
        if destination_coords is None:
            # Skip request if destination coordinates are not found
            continue

        release_time = random.randint(0, T_max - max_slack)

        travel_time = shortest_path_cached(
            node2d_to_3d[origin],
            node2d_to_3d[destination]
        )

        if not np.isfinite(travel_time):
            continue

        slack = random.randint(min_slack, max_slack)

        requests.append(
            Request(
                origin,
                destination,
                release_time,
                release_time + slack,   # latest pickup
                travel_time,            # earliest arrival
                rid,
                origin_coords['x_coord'], # Pass x_coord
                origin_coords['y_coord'],  # Pass y_coord
                destination_coords['x_coord'], # Pass dest_x
                destination_coords['y_coord']  # Pass dest_y
            )
        )

    return requests, vehicles

def argmin_overlap(Q1, Q2, num_reads=500):
    sampler = SimulatedAnnealingSampler()
    s1 = sampler.sample_qubo(Q1, num_reads=num_reads)
    s2 = sampler.sample_qubo(Q2, num_reads=num_reads)

    E1_min = s1.first.energy
    E2_min = s2.first.energy

    sols1 = {tuple(r.sample.values()) for r in s1.data() if r.energy == E1_min}
    sols2 = {tuple(r.sample.values()) for r in s2.data() if r.energy == E2_min}

    return len(sols1 & sols2) / max(1, len(sols1))

def energy_gap(Q, num_reads=200):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads)

    energies = sorted({rec.energy for rec in sampleset.data()})
    if len(energies) < 2:
        return 0.0  # degenerate or solver failure
    return energies[1] - energies[0]

import numpy as np

from collections import defaultdict

def qubo_stats_from_dict(Qd):
    """
    FAST structural stats directly from a QUBO dict {(i,j): val} without densifying.
    Assumes QUBO indices are 0..n-1 (or at least max index defines n).
    """
    if not Qd:
        return {
            "qubo_vars": 0, "qubo_couplers": 0, "qubo_graph_density": 0.0,
            "node_density_avg": 0.0, "node_density_max": 0.0, "node_density_min": 0.0,
            "degree_avg": 0.0, "degree_max": 0.0, "degree_min": 0.0,
            "qubo_max_abs": 0.0, "qubo_min_nonzero_abs": 0.0, "qubo_dynamic_range": float("inf"),
            "qubo_logical_qubits": 0,
        }

    # n = max index + 1
    max_idx = 0
    max_abs = 0.0
    min_nz = float("inf")

    degrees = defaultdict(int)
    couplers = 0

    for (i, j), v in Qd.items():
        if i > max_idx: max_idx = i
        if j > max_idx: max_idx = j

        av = abs(float(v))
        if av > 0:
            if av > max_abs: max_abs = av
            if av < min_nz: min_nz = av

        if i != j and v != 0:
            # count each undirected edge once
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
    """
    Only compute condition number for small matrices.
    Anything bigger is a waste of time for your use case.
    """
    n = int(Q_dense.shape[0])
    if n == 0 or n > max_n:
        return np.nan
    s = np.linalg.svd(Q_dense, compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] != 0 else float("inf")

def energy_gap_fast(Q, num_reads=25):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads)
    energies = sorted({rec.energy for rec in sampleset.data()})
    return float(energies[1] - energies[0]) if len(energies) >= 2 else 0.0

def argmin_overlap_FAST_DISABLE(*args, **kwargs):
    return np.nan

import time

def _qubo_dict_to_numpy(Qd, n=None):
    import numpy as np
    if n is None:
        n = max(max(i, j) for (i, j) in Qd.keys()) + 1 if Qd else 0
    Qm = np.zeros((n, n), dtype=float)
    for (i, j), c in Qd.items():
        Qm[i, j] += c
        if i != j:
            Qm[j, i] += c
    return Qm

from scipy.optimize import linear_sum_assignment

import time

import numpy as np

from collections import defaultdict

def quantum_mwis_run(
    metadata,
    baseline_vehicle_costs,   # kept for signature parity; not used here
    csv_filename="results.csv",
    real=False,
    request_stats={},
    seed=123,
    trial=1
):
    """
    Quantum MWIS run with clean timing semantics.

    - total_run_time: end-to-end wall time inside THIS function (solver pipeline only)
    - solve_time: time spent in solve_qubo_qiskit_real (sampler/QAOA/SA)
    - rtv_graph_build_time: scenario build time from final_trips (t1 - t0), if available
    - qubo_build_time: QUBO build time (min-cost prep + QUBO generation)
    """

    run_start = time.perf_counter()

    # ----------------------------
    # 1) Prepare minimal costs
    # ----------------------------
    t_min_cost_prep_start = time.perf_counter()
    minimal_qubo_trip_costs = defaultdict(lambda: float("inf"))
    for (tkey, vid), cost in trip_costs.items():
        if cost < minimal_qubo_trip_costs[tkey]:
            minimal_qubo_trip_costs[tkey] = cost
    t_min_cost_prep_end = time.perf_counter()

    # ----------------------------
    # 2) Generate QUBO
    # ----------------------------
    t_qubo_gen_start = time.perf_counter()
    Q_matrix, all_vars = generate_qubo(
        trips,
        trip_costs=minimal_qubo_trip_costs,
        return_numpy=False,
        seed=seed
    )
    t_qubo_gen_end = time.perf_counter()

    # ----------------------------
    # 3) Solve QUBO
    # ----------------------------
    t_quantum_solve_start = time.perf_counter()
    quantum_bitstring = solve_qubo_qiskit_real(Q_matrix, reps=2, real=real)
    t_quantum_solve_end = time.perf_counter()

    # ----------------------------
    # 4) Decode + conflict-free selection
    # ----------------------------
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

    # ----------------------------
    # 5) Vehicle assignment (Hungarian)
    # ----------------------------
    t_vehicle_assign_start = time.perf_counter()
    Sigma_opt = []
    if final_trips_list and vehicles:
        INVALID = 1e9
        C = np.full((len(final_trips_list), len(vehicles)), INVALID, dtype=float)
        for i, tkey in enumerate(final_trips_list):
            for j, v in enumerate(vehicles):
                cost = travel_cached(v.id, tkey)
                if cost is not None:
                    C[i, j] = cost
        rows, cols = linear_sum_assignment(C)
        for r, c in zip(rows, cols):
            if C[r, c] < INVALID:
                Sigma_opt.append((final_trips_list[r], vehicles[c].id))
    t_vehicle_assign_end = time.perf_counter()

    # ----------------------------
    # 6) Metrics calc
    # ----------------------------
    t_metrics_calc_start = time.perf_counter()

    served_requests = set()
    served_trips = []
    for trip_ids, vid in Sigma_opt:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        served_trips.append((trip_objs, vehicle_lookup[vid]))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = 100.0 * len(served_requests) / len(requests) if requests else 0.0

    detours, waiting, VMT = [], [], 0.0
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

    t_metrics_calc_end = time.perf_counter()

    # ----------------------------
    # 7) Structural stats + diagnostics
    # ----------------------------
    t_struct_stats_start = time.perf_counter()

    base_stats = qubo_stats_from_dict(Q_matrix) if isinstance(Q_matrix, dict) else qubo_stats_from_dict({(i,j): Q_matrix[i,j] for i in range(Q_matrix.shape[0]) for j in range(Q_matrix.shape[1]) if Q_matrix[i,j] != 0})
    qstats = {f"base_{k}": v for k, v in base_stats.items()}

    energygap = np.nan
    conditionnumber = np.nan
    if isinstance(Q_matrix, np.ndarray) and Q_matrix.shape[0] <= 200:
        conditionnumber = condition_number(Q_matrix, max_n=200)

    t_struct_stats_end = time.perf_counter()

    run_end = time.perf_counter()

    # ----------------------------
    # Save metrics
    # ----------------------------
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
        "vmt": VMT,

        # Clean timing semantics:
        "total_run_time": run_end - run_start,
        "solve_time": t_quantum_solve_end - t_quantum_solve_start,
        "rtv_graph_build_time": (t1 - t0) if ("t0" in globals() and "t1" in globals()) else None,

        # Detailed breakdown (optional but consistent with QC):
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

    # Prevent request_stats from overwriting core keys
    for k, v in request_stats.items():
        if k not in metrics_to_save:
            metrics_to_save[k] = v

    metrics_to_save.update(qstats)

    save_metrics_to_csv(csv_filename, metrics_to_save)
    print("\n--- Quantum MWIS Results Saved ---")
    print("Percent Serviced:", percent_serviced)

    return Sigma_opt

import pandas as pd

import os

import numpy as np

CSV_COLUMNS = [

    # =============================
    # Core metadata
    # =============================
    "city",
    "run_type",
    "nodes",
    "num_vehicles",
    "num_requests",
    "seed",
    "trial",

    # =============================
    # Scenario-level timing (measured ONCE around final_trips in the outer loop)
    # =============================
    #"scenario_total_time",          # NEW: includes request generation + RTV build + trip enumeration
    "rtv_graph_build_time",         # scenario component (from final_trips)

    # =============================
    # Performance outcomes
    # =============================
    "percent_serviced",
    "avg_waiting_time",
    "max_waiting_time",
    "avg_detour_factor",
    "max_detour_factor",
    "vmt",

    # =============================
    # Solver-level timing (measured INSIDE each solver function)
    # =============================
    "total_run_time",               # solver pipeline wall time (per function call)
    "solve_time",                   # optimizer call only (ILP solve or quantum solve)
    "qubo_build_time",              # all pre-solve QUBO work (prep + gen [+ compress])

    # =============================
    # Quantum pipeline breakdown
    # (These will be blank/None for Classical + Greedy)
    # =============================
    "time_min_cost_prep",
    "time_qubo_gen",
    "time_compress",
    "time_quantum_solve",
    "time_decode",
    "time_vehicle_assignment",
    "time_metrics_calc",
    "time_struct_stats",
    "time_total_quantum_block",     # should match total_run_time for Quantum/QC

    # =============================
    # ILP structural size (Classical only; else 0/None)
    # =============================
    "ilp_num_vars",
    "ilp_num_constraints",
    "ilp_num_integer_vars",
    "ilp_num_nonzero_coeffs",

    # =============================
    # Quantum diagnostics (Quantum/QC only; else None)
    # =============================
    "energy_gap",
    "argmin_preservation",
    "condition_number",
    "real_quantum_hardware",

    # =============================
    # Compression parameter (QC only; else None)
    # =============================
    "K",

    # =============================
    # Slack diagnostics (scenario-level; copied into each row via stats)
    # =============================
    "infeasible_windows",
    "arrival_violations",
    "mean_slack",
    "min_slack",
    "max_slack",
    "p25_slack",
    "median_slack",
    "p75_slack",

    # =============================
    # Baseline QUBO stats (Quantum + QC; else None)
    # =============================
    "base_qubo_vars",
    "base_qubo_couplers",
    "base_qubo_graph_density",
    "base_node_density_avg",
    "base_node_density_max",
    "base_node_density_min",
    "base_degree_avg",
    "base_degree_max",
    "base_degree_min",
    "base_qubo_max_abs",
    "base_qubo_min_nonzero_abs",
    "base_qubo_dynamic_range",

    # =============================
    # Compressed QUBO stats (QC only; else None)
    # =============================
    "comp_qubo_vars",
    "comp_qubo_couplers",
    "comp_qubo_graph_density",
    "comp_node_density_avg",
    "comp_node_density_max",
    "comp_node_density_min",
    "comp_degree_avg",
    "comp_degree_max",
    "comp_degree_min",
    "comp_qubo_max_abs",
    "comp_qubo_min_nonzero_abs",
    "comp_qubo_dynamic_range",

    # =============================
    # Qubit counts (Quantum + QC; else None)
    # =============================
    "qubo_logical_qubits",

    # =============================
    # Greedy instrumentation (Greedy only; else None/0)
    # =============================
    "time_greedy_build_candidates",
    "greedy_edges_total",
    "time_greedy_sort",
    "time_greedy_select",
    "time_greedy_total",
    "greedy_sort_work",
    "greedy_assignments",
]
