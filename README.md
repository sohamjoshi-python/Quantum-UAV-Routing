# Quantum UAV Routing

This repository contains code for studying quantum and classical approaches to Unmanned Aerial Vehicle (UAV) routing for emergency medical delivery. The project compares classical ILP and greedy assignment methods against a quantum optimization formulation based on QUBO / MWIS.

## Repository Structure
- `data/` raw and processed datasets
- `src/quantum_uav_routing/` core pipeline code
- `scripts/` reproducible experiment entry points
- `results/` generated CSVs, logs, and figures
- `notebooks/` exploratory analysis and debugging
- `tests/` unit tests for major components

## Pipeline
1. Load and preprocess GMNS network
2. Build 3D UAV graph
3. Generate requests and vehicles
4. Build RV/RR relations and feasible trips
5. Construct RTV graph
6. Run greedy / ILP / quantum solver
7. Save metrics and analyze scaling/performance

## Installation
```bash
pip install -r requirements.txt
