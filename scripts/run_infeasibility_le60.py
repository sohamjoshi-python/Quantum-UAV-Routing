"""
Targeted rerun for the raw-QUBO-infeasibility + per-request-slack instrumentation,
capped at <= 60 requests so it stays cheap (skips the 80/120/140/160/180 sizes and
their expensive ILP solves).

Why a wrapper instead of editing run_experiment.py: run_experiment reads its module
-level SCENARIOS_BY_VEHICLES inside main(). We override that attribute here BEFORE
calling main(), keeping your committed file untouched. We also route output to a
SEPARATE csv so your existing 10-trial experiment_results.csv is not disturbed.

Usage (from repo root):
  python scripts/run_infeasibility_le60.py --fresh --trials 10 --skip-real-quantum

All flags are forwarded to run_experiment. We inject a default --results-csv of
results/experiment_results_infeas.csv unless you pass your own.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_experiment as R  # noqa: E402

# Keep only vehicle groups whose request sizes are all <= 60, and within each
# group drop any request size > 60. This yields:
#   5:[5,10], 10:[10,20], 20:[20,30,40], 30:[30,60]
MAX_REQUESTS = 60
_capped = {}
for veh, req_list in R.SCENARIOS_BY_VEHICLES.items():
    kept = [r for r in req_list if r <= MAX_REQUESTS]
    if kept:
        _capped[veh] = kept
R.SCENARIOS_BY_VEHICLES = _capped

if __name__ == "__main__":
    # Default to a separate results CSV so the existing full sweep is untouched.
    if not any(a.startswith("--results-csv") for a in sys.argv[1:]):
        sys.argv += ["--results-csv", "results/experiment_results_infeas.csv"]
    print(f"[wrapper] scenarios (<= {MAX_REQUESTS} requests): {R.SCENARIOS_BY_VEHICLES}")
    R.main()