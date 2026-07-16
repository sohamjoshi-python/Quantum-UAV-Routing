from __future__ import annotations

import time
from collections import defaultdict
import os
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
from .penalty_scaling import derive_M, infer_nu_max
from dotenv import load_dotenv
load_dotenv()


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
    cost_alpha=1.0,         # w_t = ignore_cost*|t| - cost_alpha * trip_cost
    return_parts=False,     # if True, also return (Q_obj, Q_excl) dicts/matrices
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
      maximize benefit w_t = ignore_cost*|t| - cost_alpha * trip_costs[t]
      QUBO minimization uses diagonal -w_t

    If return_parts=True, returns
      (Q, all_vars, Q_obj, Q_excl, excl_gates, request_roots, req_to_trip_idxs, request_gates)
    where Q_obj holds only objective (benefit) terms, Q_excl holds only
    exclusivity-penalty terms, excl_gates is a bottom-up list of (u, a, b),
    request_roots maps request id -> root variable index, req_to_trip_idxs
    maps request id -> list of trip-variable indices, and request_gates maps
    request id -> that request's (u,a,b) gates in bottom-up order.
    Q == Q_obj + Q_excl entrywise.
    """

    rng = np.random.default_rng(seed)
    cost_alpha = float(cost_alpha)

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
        benefit = float(ignore_cost) * len(tkey) - cost_alpha * c
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
    Q_obj = defaultdict(float)
    Q_excl = defaultdict(float)
    excl_gates = []  # (u, a, b) merge-tree gates, bottom-up order
    request_gates = defaultdict(list)  # request -> [(u, a, b), ...] bottom-up

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

    def addQ(i, j, v, store=None):
        # store symmetric in dict (your downstream code sometimes assumes full dict)
        target = Qdict if store is None else store
        target[(i, j)] += float(v)
        if i != j:
            target[(j, i)] += float(v)
        if store is not None:
            # also accumulate into the full Q
            Qdict[(i, j)] += float(v)
            if i != j:
                Qdict[(j, i)] += float(v)

    def add_square_constraint(u, a, b, weight, request_id=None):
        """
        Add weight*(u - a - b)^2 to Q.
        Expansion: weight*(u + a + b - 2ua - 2ub + 2ab)
        """
        addQ(u, u, weight, store=Q_excl)
        addQ(a, a, weight, store=Q_excl)
        addQ(b, b, weight, store=Q_excl)

        addQ(u, a, -2 * weight, store=Q_excl)
        addQ(u, b, -2 * weight, store=Q_excl)
        addQ(a, b,  2 * weight, store=Q_excl)
        excl_gates.append((u, a, b))
        if request_id is not None:
            request_gates[request_id].append((u, a, b))

    # -------------------------
    # 4) Objective on diagonal
    # -------------------------
    for t in trip_vars:
        i = idx[t]
        w_t = float(ignore_cost) * len(t) - cost_alpha * float(trip_costs[t])
        addQ(i, i, -w_t, store=Q_obj)

    # --------------------------------------------------------
    # 5) For each request: sum(x in S_r) <= 1 via merge tree
    # --------------------------------------------------------
    request_roots = {}  # request id -> QUBO variable index of tree root
    for r, tlist in req_to_trip_idxs.items():
        if len(tlist) == 0:
            continue
        if len(tlist) == 1:
            request_roots[r] = tlist[0]
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
                add_square_constraint(u, a, b, M, request_id=r)
                nxt.append(u)

            current = nxt
            level += 1

        request_roots[r] = current[0]

    if next_aux != N:
        raise RuntimeError(f"Aux mismatch: predicted {aux_needed}, used {next_aux - T}")

    # -------------------------
    # 6) Return format
    # -------------------------
    Q_full = dict(Qdict)
    Q_obj_out = dict(Q_obj)
    Q_excl_out = dict(Q_excl)

    if return_parts:
        if return_numpy:
            def _to_dense(qd):
                Q = np.zeros((N, N), dtype=float)
                for (i, j), v in qd.items():
                    Q[i, j] = v
                return Q
            return (
                _to_dense(Q_full),
                all_vars,
                _to_dense(Q_obj_out),
                _to_dense(Q_excl_out),
                list(excl_gates),
                dict(request_roots),
                {r: list(idxs) for r, idxs in req_to_trip_idxs.items()},
                {r: list(gs) for r, gs in request_gates.items()},
            )
        return (
            Q_full,
            all_vars,
            Q_obj_out,
            Q_excl_out,
            list(excl_gates),
            dict(request_roots),
            {r: list(idxs) for r, idxs in req_to_trip_idxs.items()},
            {r: list(gs) for r, gs in request_gates.items()},
        )

    if not return_numpy:
        return Q_full, all_vars

    # Dense convert only at the end (still O(N^2), so keep N small via caps)
    Q = np.zeros((N, N), dtype=float)
    for (i, j), v in Q_full.items():
        Q[i, j] = v
    return Q, all_vars

import numpy as np
from scipy.optimize import minimize
import dimod
from dwave.samplers import SimulatedAnnealingSampler

# Qiskit Imports
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

def solve_qubo_qiskit_real(Q, reps=2, max_iter=10, real=False, num_reads=1000, num_sweeps=1000):
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
        sampleset = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps)

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

_ibm_token = os.getenv("QISKIT_IBM_TOKEN")
if _ibm_token:
    QiskitRuntimeService.save_account(
        token=_ibm_token,
        channel="ibm_quantum_platform",
        overwrite=True,
    )

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

def _save_per_request_rows(rows, csv_path):
    """Append per-request outcome rows to a side CSV (header written once).
    Pure data capture; independent of the main results file."""
    import csv as _csv
    import os as _os
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    write_header = not _os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def quantum_mwis_run(
    metadata,
    baseline_vehicle_costs,   # kept for signature parity; not used here
    csv_filename="results.csv",
    real=False,
    request_stats={},
    seed=123,
    trial=1,
    ignore_costs = 5000,
    M_val = 25000,
    auto_M=False,
    cap_per_request=30,
    num_reads=1000,
    num_sweeps=1000,
    cost_alpha=1.0,
):
    """
    Quantum MWIS run with clean timing semantics.

    - total_run_time: end-to-end wall time inside THIS function (solver pipeline only)
    - solve_time: time spent in solve_qubo_qiskit_real (sampler/QAOA/SA)
    - rtv_graph_build_time: scenario build time from final_trips (t1 - t0), if available
    - qubo_build_time: QUBO build time (min-cost prep + QUBO generation)

    When auto_M=True, the exclusivity penalty is derived per scenario as
    M = 5.0 * ignore_costs * nu_max (nu_max inferred from the trip set).
    With auto_M=False (default), behavior is identical to a fixed M_val.

    cap_per_request: max trip vars incident to each request when building the QUBO.
    Pass None to disable pruning (keep all incident trips).

    num_reads / num_sweeps: Simulated Annealing effort when real=False.

    cost_alpha: scales trip cost in the QUBO objective
      w_t = ignore_costs * |t| - cost_alpha * trip_cost
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
    # Per-scenario penalty: when auto_M is on, scale M with the trip capacity
    # actually present in this scenario (nu_max) so the exclusivity penalty keeps
    # dominating the benefit ceiling lambda*nu_max as capacity grows. Anchored to
    # the validated capacity-2 point (k=5.0), so capacity-2 is unchanged and
    # capacity-3 gets the larger M its bigger trips require. See penalty_scaling.py.
    if auto_M:
        nu_max = infer_nu_max(trips, default=2)
        M_effective = derive_M(ignore_cost=ignore_costs, nu_max=nu_max)
        print(f"  [auto_M] nu_max={nu_max} -> M={M_effective:.0f} "
              f"(was fixed {M_val})")
    else:
        M_effective = M_val

    t_qubo_gen_start = time.perf_counter()
    Q_matrix, all_vars = generate_qubo(
        trips,
        trip_costs=minimal_qubo_trip_costs,
        ignore_cost=ignore_costs,
        M=M_effective,
        return_numpy=False,
        seed=seed,
        cap_per_request=cap_per_request,
        cost_alpha=cost_alpha,
    )
    t_qubo_gen_end = time.perf_counter()

    # ----------------------------
    # 3) Solve QUBO
    # ----------------------------
    t_quantum_solve_start = time.perf_counter()
    quantum_bitstring = solve_qubo_qiskit_real(
        Q_matrix, reps=2, real=real, num_reads=num_reads, num_sweeps=num_sweeps
    )
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

    # --- INSTRUMENTATION (data capture only; does not affect the pipeline) ---
    # Measure how infeasible the RAW QUBO output was BEFORE the greedy cleanup.
    # Trips dropped between selected_trip_keys and final_trips_list are request-
    # exclusivity violations the QUBO penalty M failed to prevent. Quantifies how
    # much work the classical cleanup is doing.
    infeasibility_stats = raw_infeasibility_from_selection(
        selected_trip_keys, final_trips_list
    )

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

    # --- INSTRUMENTATION (data capture only; does not affect the pipeline) ---
    # Record per-request served/dropped outcome and each request's time-window
    # slack, so downstream analysis can test whether DROPPED requests systematically
    # had tighter windows / higher detour. Written to a side CSV keyed by
    # (seed, trial, size, request_id) so the main results schema is untouched.
    # Slack = latest_pickup - request_time (width of the feasible pickup window).
    try:
        served_detour_by_rid = {}
        for trip, v in served_trips:
            for r in trip:
                total_time = travel(v, trip)
                shortest = sum(rr.t_star for rr in trip)
                served_detour_by_rid[r.id] = (
                    total_time / shortest if shortest > 0 else 1.0
                )
        per_request_rows = []
        for r in requests:
            served_flag = 1 if r.id in served_requests else 0
            slack = None
            try:
                slack = float(r.tplr) - float(r.trr)
            except (TypeError, ValueError):
                slack = None
            per_request_rows.append({
                "seed": seed,
                "trial": trial,
                "num_vehicles": len(vehicles),
                "num_requests": len(requests),
                "real_quantum_hardware": 1 if real else 0,
                "request_id": r.id,
                "served": served_flag,
                "slack": slack,
                "detour_factor": served_detour_by_rid.get(r.id, None),
            })
        _per_request_csv = str(csv_filename).replace(".csv", "") + "_per_request.csv"
        _save_per_request_rows(per_request_rows, _per_request_csv)
    except Exception as _e:  # never let instrumentation break a run
        print(f"(per-request capture skipped: {_e})")

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
        "lambda_val": ignore_costs,
        "M_val": M_effective if auto_M else M_val,
        "num_reads": num_reads if not real else None,
        "num_sweeps": num_sweeps if not real else None,
        "cost_alpha": cost_alpha,

        "energy_gap": float(energygap),
        "condition_number": float(conditionnumber),
        "real_quantum_hardware": 1 if real else 0,
        "seed": seed,
        "trial": trial,

        # Raw pre-cleanup infeasibility (instrumentation; how much the classical
        # cleanup is doing). See raw_infeasibility_from_selection.
        "raw_selected_trips": infeasibility_stats["raw_selected_trips"],
        "raw_kept_trips": infeasibility_stats["raw_kept_trips"],
        "raw_dropped_trips": infeasibility_stats["raw_dropped_trips"],
        "raw_violation_rate": infeasibility_stats["raw_violation_rate"],
        "raw_infeasible_instance": infeasibility_stats["raw_infeasible_instance"],
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


from collections import defaultdict
import numpy as np


def generate_qubo_pairwise(
    trips,
    trip_costs,
    ignore_cost=10000.0,
    M=20000.0,
    return_numpy=True,
    seed=123,
    cap_per_request=30,
    cap_total_trips=None,
    score_mode="benefit",
    cost_alpha=1.0,
):
    """
    NAIVE PAIRWISE-PENALTY QUBO (baseline for the merge-tree comparison).

    Request exclusivity sum_{t in S_r} x_t <= 1 is enforced by penalizing every
    co-selected pair: for each request r and each pair (a, b) of trips both
    containing r, add M * x_a * x_b. This is the standard clique penalty and
    introduces NO auxiliary variables, but O(k_r^2) couplers per request.

    Everything else (trip scoring, per-request cap, global cap, objective) is
    identical to generate_qubo(), so this is a clean structural baseline.

    Returns (same contract as generate_qubo):
      - return_numpy=False: (dict Qdict, all_vars)
      - return_numpy=True:  (np.ndarray Q, all_vars)
    """
    rng = np.random.default_rng(seed)
    cost_alpha = float(cost_alpha)

    # --- 0) request -> trips (unbounded) ---
    req_to_trips = defaultdict(list)
    for tkey in trips.keys():
        for r in tkey:
            req_to_trips[r].append(tkey)

    # --- 1) score + per-request cap (IDENTICAL to generate_qubo) ---
    trip_score = {}
    for tkey in trips.keys():
        c = float(trip_costs[tkey])
        benefit = float(ignore_cost) * len(tkey) - cost_alpha * c
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
            sorted(list(kept), key=lambda tk: trip_score.get(tk, -np.inf), reverse=True)[
                :cap_total_trips
            ]
        )

    trip_vars = list(kept)
    T = len(trip_vars)

    # --- rebuild request adjacency over retained trips ---
    req_to_trip_idxs = defaultdict(list)
    idx = {t: i for i, t in enumerate(trip_vars)}
    for t in trip_vars:
        ti = idx[t]
        for r in t:
            req_to_trip_idxs[r].append(ti)

    # NO auxiliary variables in the pairwise encoding.
    N = T
    all_vars = list(trip_vars)

    Qdict = defaultdict(float)

    def addQ(i, j, v):
        Qdict[(i, j)] += float(v)
        if i != j:
            Qdict[(j, i)] += float(v)

    # --- objective on diagonal (IDENTICAL to generate_qubo) ---
    for t in trip_vars:
        i = idx[t]
        w_t = float(ignore_cost) * len(t) - cost_alpha * float(trip_costs[t])
        addQ(i, i, -w_t)

    # --- request exclusivity via pairwise clique penalties ---
    # For each request, penalize every unordered pair of incident trips.
    # De-duplicate pairs across requests so two trips sharing >1 request are not
    # double-penalized (keeps the comparison to the merge tree fair).
    penalized_pairs = set()
    for r, tlist in req_to_trip_idxs.items():
        k = len(tlist)
        if k <= 1:
            continue
        for a_pos in range(k):
            ia = tlist[a_pos]
            for b_pos in range(a_pos + 1, k):
                ib = tlist[b_pos]
                pair = (ia, ib) if ia < ib else (ib, ia)
                if pair in penalized_pairs:
                    continue
                penalized_pairs.add(pair)
                addQ(pair[0], pair[1], M)

    if not return_numpy:
        return dict(Qdict), all_vars

    Q = np.zeros((N, N), dtype=float)
    for (i, j), v in Qdict.items():
        Q[i, j] = v
    return Q, all_vars


def compare_encodings(
    trips,
    trip_costs,
    ignore_cost=5000.0,
    M=25000.0,
    seed=123,
    cap_per_request=30,
    cap_total_trips=None,
):
    """
    Build BOTH encodings on the same pruned trip set and return a side-by-side
    structural comparison (the table Reviewer 3 asked for). Uses the existing
    qubo_stats_from_dict() already defined in quantum_solver.py.

    Returns a dict with 'merge_tree', 'pairwise', and 'ratios' sub-dicts.
    """
    q_tree, _ = generate_qubo(
        trips, trip_costs, ignore_cost=ignore_cost, M=M, return_numpy=False,
        seed=seed, cap_per_request=cap_per_request, cap_total_trips=cap_total_trips,
    )
    q_pair, _ = generate_qubo_pairwise(
        trips, trip_costs, ignore_cost=ignore_cost, M=M, return_numpy=False,
        seed=seed, cap_per_request=cap_per_request, cap_total_trips=cap_total_trips,
    )
    s_tree = qubo_stats_from_dict(q_tree)
    s_pair = qubo_stats_from_dict(q_pair)

    def ratio(a, b):
        return (a / b) if b else float("inf")

    return {
        "merge_tree": {
            "qubo_vars": s_tree["qubo_vars"],
            "qubo_couplers": s_tree["qubo_couplers"],
            "qubo_graph_density": s_tree["qubo_graph_density"],
            "degree_max": s_tree["degree_max"],
        },
        "pairwise": {
            "qubo_vars": s_pair["qubo_vars"],
            "qubo_couplers": s_pair["qubo_couplers"],
            "qubo_graph_density": s_pair["qubo_graph_density"],
            "degree_max": s_pair["degree_max"],
        },
        "ratios": {
            # pairwise / merge-tree: >1 means merge tree is more compact on that axis
            "couplers_pairwise_over_tree": ratio(
                s_pair["qubo_couplers"], s_tree["qubo_couplers"]
            ),
            "vars_tree_over_pairwise": ratio(
                s_tree["qubo_vars"], s_pair["qubo_vars"]
            ),
        },
    }


def raw_infeasibility_from_selection(selected_trip_keys, final_trips_list):
    """
    Measure how infeasible the RAW QUBO output was, BEFORE the greedy cleanup.

    Inputs are the two lists already produced inside quantum_mwis_run:
      - selected_trip_keys : every trip the solver set to 1 (raw QUBO output)
      - final_trips_list   : trips surviving the greedy conflict-cleanup

    A trip is dropped iff it shared a request with an already-accepted trip, i.e.
    a request-exclusivity violation the QUBO penalty M failed to prevent.

    Returns a dict:
      raw_selected_trips     : count of raw selected trips
      raw_kept_trips         : count after cleanup
      raw_dropped_trips      : how many were conflicting
      raw_violation_rate     : dropped / selected  (trip-level fraction; 0 if none selected)
      raw_infeasible_instance: 1 if ANY trip was dropped, else 0 (instance-level flag)
    """
    n_sel = len(selected_trip_keys)
    n_kept = len(final_trips_list)
    n_drop = n_sel - n_kept
    return {
        "raw_selected_trips": int(n_sel),
        "raw_kept_trips": int(n_kept),
        "raw_dropped_trips": int(n_drop),
        "raw_violation_rate": (float(n_drop) / n_sel) if n_sel else 0.0,
        "raw_infeasible_instance": 1 if n_drop > 0 else 0,
    }