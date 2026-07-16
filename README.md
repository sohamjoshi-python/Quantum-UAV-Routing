# Quantum UAV Routing

Code and experiments accompanying the paper:

> **Scaling QUBO-Based Hybrid Quantum Optimization for Emergency UAV Routing**
> Soham Joshi.

This repository studies a **two-stage hybrid quantum–classical pipeline** for
routing a fleet of Unmanned Aerial Vehicles (UAVs) delivering emergency medical
supplies on a three-dimensional model of the Phoenix, Arizona road network. The
fleet-assignment problem is posed as a Request–Trip–Vehicle (RTV) assignment and
solved three ways for comparison:

- an **exact Integer Linear Program (ILP)** (CBC via PuLP) — the optimal reference,
- a **greedy heuristic** (Alonso-Mora-style) — a fast baseline,
- a **hybrid QUBO pipeline**: request exclusivity is encoded as a **binary
  merge-tree QUBO** and solved with **simulated annealing** (a classical surrogate
  for quantum annealing), followed by **classical greedy conflict resolution** and
  **Hungarian vehicle assignment**.

The paper's contribution is *representational and structural*: the merge-tree QUBO
grows sub-quadratically in decision variables (fitted exponent ≈ 1.63 vs ≈ 2.74 for
the ILP), is sparse and constant-degree, and embeds onto the D-Wave Pegasus
topology with lower overhead than a dense pairwise encoding. All quantum numbers
come from simulated annealing plus a modeled D-Wave timing proxy — **not** from
quantum hardware.

## Pipeline overview

1. Download the GMNS+ Phoenix network and lift it into 3D using discrete altitude
   levels (0, 50, 100, 200, 400 ft).
2. Compute shortest-path travel times via bidirectional Dijkstra.
3. Generate randomized requests and vehicles (fixed seeds) for each
   vehicle–request configuration.
4. Build the RTV graph of feasible trips (capacity ν = 2, max wait 1800 s,
   max detour 1200 s).
5. Solve each instance with the greedy, ILP, and QUBO pipelines.
6. Save per-instance metrics (percent serviced, VMT, detour, waiting time,
   runtime, structural diagnostics).
7. Aggregate into the paper's figures and tables.

## Repository structure

```text
Quantum-UAV-Routing/
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/                         # (git-ignored) auto-downloaded + generated network
├── results/                      # aggregated CSVs, tables, and figures for the paper
│   ├── analysis_figures/
│   ├── analysis_tables/
│   ├── quantum_tuning/
│   ├── encoding_comparison/
│   ├── embedding/
│   └── matched_quality/
├── scripts/                      # runnable experiment + analysis entry points
│   ├── run_experiment.py
│   ├── compare_results.py
│   ├── optimal_tuning_params.py
│   ├── rank_all_scenarios.py
│   ├── compare_encodings_sweep.py
│   ├── analyze_encoding_comparison.py
│   ├── compare_merge_vs_pairwise_solve.py
│   ├── embedding_analysis.py
│   ├── analyze_embedding.py
│   ├── matched_quality_comparison.py
│   └── run_infeasibility_le60.py
└── src/quantum_uav_routing/
    ├── core/            # Request / Vehicle / trip entities
    ├── network/         # GMNS download, 3D lift, shortest paths
    ├── rtv/             # RTV graph + feasible trip construction
    ├── classical/       # greedy + ILP solvers
    ├── quantum/         # merge-tree QUBO, penalty scaling, encoding comparison
    ├── io/              # metric computation + CSV writers
    └── analysis/        # shared analysis helpers
```

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/sohamjoshi-python/Quantum-UAV-Routing.git
cd Quantum-UAV-Routing

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Data

You do **not** need to download anything manually. On its first run,
`scripts/run_experiment.py` will:

1. clone/download the public
   [GMNS_Plus_Dataset](https://github.com/HanZhengIntelliTransport/GMNS_Plus_Dataset)
   into `data/GMNS_Plus_Dataset/`, and
2. generate the 3D-lifted network files (`node_3d.csv`, `link_3d.csv`,
   `node2d_to_3d_mapping.csv`) under `data/raw/`.

The entire `data/` folder is git-ignored because it is fully reproducible from
the source dataset. The GMNS+ datasets were developed by Professor Xuesong Zhou
and the NSF POSE research team at Arizona State University.

### Optional: quantum hardware

The reported results use simulated annealing only. To run the small-instance
QAOA gate-model compatibility check on IBM Quantum, create a `.env` file (also
git-ignored) with your token:

```env
QISKIT_IBM_TOKEN=your_token_here
```

When the token is absent, all quantum runs fall back to simulated annealing and
the hardware/QAOA steps are skipped automatically.

## Reproducing the paper

Run every command from the repository root. The main experiment is the heaviest
step (exact ILP solves dominate); the analysis scripts are cheap and simply read
the CSVs the experiment produces.

### 1. Main experiment — service, route quality, runtime

Runs greedy, ILP, and the QUBO pipeline across all 21 vehicle–request
configurations (5–90 vehicles, 5–180 requests) and 5 trials per configuration,
appending to `results/experiment_results.csv`. It resumes automatically if
interrupted; use `--fresh` to start over.

```bash
python scripts/run_experiment.py --fresh --trials 5 --skip-real-quantum
```

Then aggregate into the paper's figures/tables (written to
`results/analysis_figures/` and `results/analysis_tables/`):

```bash
python scripts/compare_results.py --csv results/experiment_results.csv
```

This produces the decision-space scaling (Fig. 1, Table I), percent serviced
(Fig. 2, Table II), VMT / route quality (Fig. 3, Table III), and runtime /
modeled-annealing scaling (Fig. 7, Table VI) results.

### 2. Hyperparameter selection (λ, M)

Sweeps the QUBO penalty weights with repeated annealing draws and ranks
configurations by mean percent serviced with cross-size consistency (Section
IV-D). Outputs go to `results/quantum_tuning/`.

```bash
python scripts/optimal_tuning_params.py
python scripts/rank_all_scenarios.py --csv results/quantum_tuning/accuracy_by_run.csv
```

### 3. Raw feasibility (merge-tree under annealing)

Instruments the raw pre-cleanup QUBO infeasibility on the smaller sizes
(≤ 60 requests) into a separate CSV (Table II "Raw viol." / Fig. 4):

```bash
python scripts/run_infeasibility_le60.py --fresh --trials 10 --skip-real-quantum
```

### 4. Encoding comparison — merge-tree vs pairwise (clique)

A structural sweep (no solving) that measures variables, couplers, density, and
max degree for both encodings on identical trip sets, then formats the paper
table and figure (Table IV, Fig. 5). Outputs to `results/encoding_comparison/`.

```bash
python scripts/compare_encodings_sweep.py
python scripts/analyze_encoding_comparison.py
```

To also compare the two encodings on the *solve* side (raw violation rate and
service on the same trip set):

```bash
python scripts/compare_merge_vs_pairwise_solve.py
```

### 5. Hardware embeddability (D-Wave Pegasus)

Minor-embeds both encodings onto the Pegasus topology with `minorminer`,
recording physical-qubit count and chain length; the analysis script honestly
separates genuine failures from timeouts (Table V, Fig. 6). Outputs to
`results/embedding/`.

```bash
python scripts/embedding_analysis.py --timeout 120
python scripts/analyze_embedding.py --budget 120
```

### 6. Matched-service route quality

Sweeps the pairwise QUBO's objective weight to compare route quality (VMT)
against the ILP at matched service levels (Table III context). Outputs to
`results/matched_quality/`.

```bash
python scripts/matched_quality_comparison.py
```

## Single-instance runs

`run_experiment.py` can also be used to explore one configuration at a time via
its flags. Key options:

- `--trials` — number of seeded trials per configuration
- `--skip-real-quantum` — never call IBM hardware (annealing only)
- `--skip-classical` — quantum-only (useful for annealing sweeps)
- `--lambda-val` / `--m-val` — QUBO service reward λ and exclusivity penalty M
- `--cap-per-request` — max trip variables kept per request (κ; `none` disables)
- `--num-reads` / `--num-sweeps` — simulated-annealing sampler settings
- `--ilp-time-limit` — wall-clock cap per ILP solve (default 2 h)

## Reproducibility notes

- All request/vehicle generation uses deterministic per-scenario seeds derived
  from a fixed base seed, so instances are shared across all three solvers.
- Simulated annealing is stochastic; small run-to-run variation in the QUBO
  metrics is expected. Structural quantities (variable/coupler counts, degrees,
  embeddings) are deterministic given a fixed trip set.
- The `data/` folder and any secrets (`.env`) are git-ignored and rebuilt/reused
  locally.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@inproceedings{joshi_quantum_uav_routing,
  title     = {Scaling QUBO-Based Hybrid Quantum Optimization for Emergency UAV Routing},
  author    = {Joshi, Soham},
  booktitle = {IEEE Quantum Week},
  year      = {2026}
}
```

## Acknowledgments

The author thanks Professor Xuesong Zhou at Arizona State University for
mentorship and guidance, the NSF POSE research team at ASU for the GMNS+
datasets, and IBM Quantum for access to quantum computing resources.

## License

MIT License.
