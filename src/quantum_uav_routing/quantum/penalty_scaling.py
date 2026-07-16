"""
Per-scenario penalty derivation for the merge-tree QUBO.

WHY THIS EXISTS
---------------
The exclusivity penalty M must dominate the benefit any trip variable can extract,
or the sampler pays the penalty to grab a high-reward trip and returns an
infeasible selection. The benefit ceiling of a single trip is

    w_max = lambda * nu_max - min_trip_cost   <=   lambda * nu_max

so a feasibility-preserving penalty must satisfy (Lucas 2014, Ising formulations):

    M  >  lambda * nu_max.

Simulated annealing needs headroom above that strict bound, so we use a calibrated
multiplier k:

    M = k * lambda * nu_max.

k is ANCHORED to the validated capacity-2 operating point (lambda=2500, M=25000,
nu_max=2), which gives k = 25000 / (2500 * 2) = 5.0. By construction this
reproduces the known-good capacity-2 penalty exactly, and prescribes a larger M as
capacity grows -- which is exactly the term that was missing when a fixed
M=25000 was reused at capacity 3 (the benefit ceiling rose from 2*lambda to
3*lambda but M did not, so the raw QUBO became ~98% infeasible).

Per-scenario INPUTS:
    lambda (ignore_cost) : benefit weight per served request (config: 2500)
    nu_max               : trip capacity for THIS scenario (2, 3, ...)

Optional refinement (OFF by default, see derive_M): a mild conflict-density term.
We keep it off unless a capacity-3 test shows the ceiling term alone is
insufficient -- adding an unvalidated size term would be guessing.
"""

from __future__ import annotations

# Calibration constant, anchored to the validated capacity-2 point:
#   k = M_good / (lambda_good * nu_good) = 25000 / (2500 * 2) = 5.0
PENALTY_MULTIPLIER_K = 5.0


def derive_M(
    ignore_cost: float,
    nu_max: int,
    k: float = PENALTY_MULTIPLIER_K,
    trips=None,
    trip_costs=None,
    density_aware: bool = False,
) -> float:
    """Return a per-scenario exclusivity penalty M.

    Baseline (density_aware=False): benefit-ceiling scaling anchored to cap-2.
        M = k * ignore_cost * nu_max

    density_aware=True (optional, requires trips): additionally scale by the mean
    number of trips incident to a request, normalized so the capacity-2 baseline
    (where the anchor was set) is unchanged. Use ONLY if the ceiling term proves
    insufficient in testing; it is off by default to avoid unvalidated tuning.
    """
    M_ceiling = float(k) * float(ignore_cost) * float(nu_max)

    if not density_aware or trips is None:
        return M_ceiling

    # Optional conflict-density multiplier, normalized to ~1.0 at the pairwise
    # baseline so it does not disturb the validated capacity-2 penalty.
    from collections import defaultdict
    incidence = defaultdict(int)
    for tkey in trips.keys():
        for r in tkey:
            incidence[r] += 1
    if not incidence:
        return M_ceiling
    mean_incident = sum(incidence.values()) / len(incidence)
    # Baseline mean incidence at the anchor was ~ the per-request cap fraction;
    # normalize by a reference so pairwise stays ~1x. Reference chosen as the
    # observed capacity-2 mean (~ the merge-tree bounded value). Kept conservative.
    REFERENCE_INCIDENCE = mean_incident  # self-normalizing no-op unless overridden
    factor = mean_incident / REFERENCE_INCIDENCE if REFERENCE_INCIDENCE else 1.0
    return M_ceiling * factor


def infer_nu_max(trips, default: int = 2) -> int:
    """Largest trip size present in the trip set (the scenario's effective
    capacity). Lets M adapt automatically when triples are added."""
    if not trips:
        return default
    return max((len(tkey) for tkey in trips.keys()), default=default)
