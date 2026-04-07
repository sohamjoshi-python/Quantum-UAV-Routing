# Quantum UAV Routing

This repository contains code for studying quantum and classical approaches to Unmanned Aerial Vehicle (UAV) routing for emergency medical delivery. The project compares classical Integer Linear Programming (ILP) and greedy assignment methods against a quantum optimization formulation based on QUBO / MWIS.

## Overview

Efficient UAV routing is important for time-sensitive delivery tasks such as transporting emergency medical supplies. This project studies whether a quantum optimization formulation can reduce decision complexity while maintaining competitive routing performance relative to classical baselines.

The repository supports three main solver pipelines:

- **Classical ILP**
- **Classical Greedy**
- **Quantum MWIS / QUBO**

The full workflow is:

1. Load and preprocess a GMNS transportation network
2. Build a 3D UAV routing graph
3. Generate requests and vehicles
4. Construct feasible trips and RTV-style assignment structures
5. Run greedy, ILP, or quantum optimization
6. Save experiment metrics
7. Analyze scaling and performance results

## Repository Structure

```text
Quantum-UAV-Routing/
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── phoenix/
│   ├── interim/
│   └── processed/
├── results/
│   ├── raw_runs/
│   ├── aggregated/
│   ├── figures/
│   └── logs/
├── scripts/
│   └── run_experiment.py
├── src/
│   └── quantum_uav_routing/
│       ├── __init__.py
│       ├── models.py
│       ├── network.py
│       ├── scenario.py
│       ├── analysis_utils.py
│       ├── io/
│       │   └── save_results.py
│       ├── classical/
│       │   ├── greedy_solver.py
│       │   └── ilp_solver.py
│       ├── quantum/
│       │   └── quantum_solver.py
│       └── rtv/
│           ├── trip_builder.py
│           └── travel.py
├── notebooks/
└── tests/
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Quantum-UAV-Routing
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

If you are using a virtual environment, activate it first.

## Data

Place the GMNS input files in:

```text
data/raw/phoenix/
├── node.csv
├── link.csv
└── zone.csv
```

If `zone.csv` is missing, the code will generate it automatically.

You can also replace `phoenix` with another network folder, as long as it contains compatible GMNS files.

## Quickstart

### Run one classical ILP experiment

```bash
python scripts/run_experiment.py \
  --network-dir data/raw/phoenix \
  --num-requests 20 \
  --num-vehicles 10 \
  --seed 42 \
  --solver classical \
  --results-csv results/results.csv
```

### Run one greedy baseline experiment

```bash
python scripts/run_experiment.py \
  --network-dir data/raw/phoenix \
  --num-requests 20 \
  --num-vehicles 10 \
  --seed 42 \
  --solver greedy \
  --results-csv results/results.csv
```

### Run one quantum baseline experiment

```bash
python scripts/run_experiment.py \
  --network-dir data/raw/phoenix \
  --num-requests 20 \
  --num-vehicles 10 \
  --seed 42 \
  --solver quantum \
  --results-csv results/results.csv
```

### Run with real quantum hardware

```bash
python scripts/run_experiment.py \
  --network-dir data/raw/phoenix \
  --num-requests 20 \
  --num-vehicles 10 \
  --seed 42 \
  --solver quantum \
  --results-csv results/results.csv \
  --real-quantum
```

## Command-Line Arguments

`run_experiment.py` supports the following arguments:

- `--network-dir`: path to the GMNS dataset directory
- `--num-requests`: number of requests to generate
- `--num-vehicles`: number of vehicles to generate
- `--seed`: random seed for reproducibility
- `--solver`: one of `greedy`, `classical`, or `quantum`
- `--results-csv`: output CSV path for experiment metrics
- `--real-quantum`: optional flag to use real quantum hardware instead of local simulated annealing

## Solver Modes

### Classical Greedy

Uses a greedy trip-assignment approach inspired by Alonso-Mora-style assignment logic. This mode is fast and useful as a baseline approximation.

### Classical ILP

Solves the assignment problem using an Integer Linear Programming formulation. This is the main classical exact baseline for small to medium instances.

### Quantum MWIS / QUBO

Builds a QUBO formulation over feasible trips, solves it using simulated annealing or QAOA-based hardware execution, then decodes the selected trips into a conflict-free routing solution.

## Outputs

Experiment results are saved to the CSV file passed through `--results-csv`.

Typical recorded metrics include:

- city
- run type
- number of requests
- number of vehicles
- percent serviced
- waiting times
- detour factors
- VMT
- solver runtime
- RTV graph build time
- QUBO build time
- ILP size statistics
- quantum structural diagnostics

Recommended output locations:

```text
results/
├── raw_runs/
├── aggregated/
├── figures/
└── logs/
```

## Reproducibility

This project uses fixed seeds for request and vehicle generation. To reproduce a run exactly, keep the following fixed:

- dataset folder
- number of requests
- number of vehicles
- seed
- solver mode
- hardware mode for quantum runs

The `--seed` argument is especially important when comparing solver outputs fairly across methods.

## Notes

- The repo assumes that `build_trips(...)` and `travel(...)` live in stable module locations under `src/quantum_uav_routing/rtv/`.
- If your current implementation still depends on hidden notebook globals, fix that before treating the repo as final.
- Real hardware quantum runs may require valid provider credentials and extra setup.

## License

MIT License

## Contact

For questions, bug reports, or collaboration, open an issue in the repository or contact the repository owner.