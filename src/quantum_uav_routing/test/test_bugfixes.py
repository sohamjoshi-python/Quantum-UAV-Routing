"""Regression tests for the three pre-camera-ready bugs.

1. QUBO couplers must be stored once. dimod adds (i, j) and (j, i).
2. Request and vehicle endpoints sit on the ground layer, not _z400.
3. Conflict cleanup keeps trips in decreasing reward order.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import dimod
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_uav_routing.network.load_network import (  # noqa: E402
    ground_level_node_map,
)
from quantum_uav_routing.network.shortest_path import (  # noqa: E402
    build_3d_graph,
    path_length_by_altitude,
)
from quantum_uav_routing.quantum.quantum_solver import (  # noqa: E402
    generate_qubo,
    order_trips_by_decreasing_reward,
    qubo_stats_from_dict,
    upper_triangle_qubo,
)


def _gate_energies(Q, trip_index):
    """Minimum energy of one merge gate over the auxiliary bit, for each (a, b)."""
    qubo = upper_triangle_qubo(Q)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    energies = {}
    for a, b in itertools.product((0, 1), (0, 1)):
        energies[(a, b)] = min(
            bqm.energy({trip_index[0]: a, trip_index[1]: b, trip_index[2]: u})
            for u in (0, 1)
        )
    return energies


def test_merge_gate_energies_are_not_doubled():
    """One exclusivity gate has minima 0, 0, 0, M. A doubled coupler rewards conflicts."""
    trip_a = frozenset({1})
    trip_b = frozenset({1, 2})
    trips = {trip_a: None, trip_b: None}
    costs = {trip_a: 0.0, trip_b: 0.0}
    M = 5.0
    Q, all_vars = generate_qubo(
        trips,
        costs,
        ignore_cost=0.0,
        M=M,
        return_numpy=False,
        cap_per_request=None,
        seed=0,
    )

    assert all(i <= j for (i, j) in Q)
    trip_positions = [i for i, label in enumerate(all_vars) if isinstance(label, frozenset)]
    aux_positions = [i for i, label in enumerate(all_vars) if not isinstance(label, frozenset)]
    assert len(trip_positions) == 2
    assert len(aux_positions) == 1

    energies = _gate_energies(Q, [trip_positions[0], trip_positions[1], aux_positions[0]])
    assert energies[(0, 0)] == pytest.approx(0.0)
    assert energies[(0, 1)] == pytest.approx(0.0)
    assert energies[(1, 0)] == pytest.approx(0.0)
    assert energies[(1, 1)] == pytest.approx(M)

    # A single gate is a triangle: true degree 2, not the doubled degree 4.
    stats = qubo_stats_from_dict(Q)
    assert stats["degree_max"] == pytest.approx(2.0)
    assert stats["qubo_couplers"] == 3

    mirrored = {}
    for (i, j), value in Q.items():
        mirrored[(i, j)] = value
        if i != j:
            mirrored[(j, i)] = value
    mirrored_energies = _gate_energies(
        mirrored, [trip_positions[0], trip_positions[1], aux_positions[0]]
    )
    assert mirrored_energies[(1, 1)] == pytest.approx(M)
    mirrored_stats = qubo_stats_from_dict(mirrored)
    assert mirrored_stats["degree_max"] == pytest.approx(2.0)
    assert mirrored_stats["qubo_couplers"] == 3


def test_three_trip_request_optimum_selects_at_most_one():
    """With a real penalty, no minimum-energy state selects two trips of one request."""
    trips = {
        frozenset({1}): None,
        frozenset({1, 2}): None,
        frozenset({1, 2, 3}): None,
    }
    costs = {tk: 0.0 for tk in trips}
    M = 7.0
    Q, all_vars = generate_qubo(
        trips,
        costs,
        ignore_cost=0.0,
        M=M,
        return_numpy=False,
        cap_per_request=None,
        seed=1,
    )
    qubo = upper_triangle_qubo(Q)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    n = max(all_vars.__len__(), max(bqm.variables) + 1)
    trip_idxs = [i for i, label in enumerate(all_vars) if isinstance(label, frozenset)]
    assert len(trip_idxs) == 3

    best = None
    best_states = []
    for bits in itertools.product((0, 1), repeat=n):
        sample = dict(zip(range(n), bits))
        energy = bqm.energy(sample)
        if best is None or energy < best - 1e-8:
            best = energy
            best_states = [sample]
        elif abs(energy - best) <= 1e-8:
            best_states.append(sample)

    assert best == pytest.approx(0.0)
    assert best_states
    for sample in best_states:
        assert sum(sample[i] for i in trip_idxs) <= 1


def test_cleanup_keeps_higher_reward_trip():
    low = frozenset({1})
    high = frozenset({1, 2})
    costs = {low: 0.0, high: 0.0}
    # Incoming order would keep `low` and drop `high`. Reward order does the opposite.
    ordered = order_trips_by_decreasing_reward([low, high], costs, ignore_cost=10.0)
    assert ordered == [high, low]

    covered = set()
    kept = []
    for tk in ordered:
        if not (tk & covered):
            kept.append(tk)
            covered |= tk
    assert kept == [high]


def test_ground_mapping_is_not_the_top_layer():
    """pandas to_dict on a duplicated original id keeps only the last altitude."""
    frame = pd.DataFrame(
        {
            "original_node_id": [1, 1, 1, 1, 1],
            "node_id": ["1_z0", "1_z50", "1_z100", "1_z200", "1_z400"],
        }
    )
    collapsed = frame.set_index("original_node_id")["node_id"].to_dict()
    assert collapsed[1] == "1_z400"

    mapping = ground_level_node_map([1, 2])
    assert mapping[1] == "1_z0"
    assert mapping[2] == "2_z0"


def _corridor(length):
    """Two nodes, five altitudes, one horizontal link per layer, vertical links."""
    altitudes = [0, 50, 100, 200, 400]
    nodes = []
    links = []
    for node in (1, 2):
        for z in altitudes:
            nodes.append(
                {
                    "node_id": f"{node}_z{z}",
                    "x_coord": 0.0 if node == 1 else 1.0,
                    "y_coord": 0.0,
                    "z_coord": z,
                }
            )
    for z in altitudes:
        links.append(
            {
                "from_node_id": f"1_z{z}",
                "to_node_id": f"2_z{z}",
                "length": length,
                "link_type": "horizontal",
                "altitude": z,
                "dir_flag": 1,
                "geometry": None,
            }
        )
    for node in (1, 2):
        for z1, z2 in zip(altitudes[:-1], altitudes[1:]):
            links.append(
                {
                    "from_node_id": f"{node}_z{z1}",
                    "to_node_id": f"{node}_z{z2}",
                    "length": float(z2 - z1),
                    "link_type": "vertical",
                    "altitude": f"{z1}->{z2}",
                    "dir_flag": 1,
                    "geometry": None,
                }
            )
    return build_3d_graph(pd.DataFrame(nodes), pd.DataFrame(links))


def test_ground_endpoints_climb_and_descend_on_a_long_hop():
    short = path_length_by_altitude(_corridor(10.0), "1_z0", "2_z0")
    long = path_length_by_altitude(_corridor(5000.0), "1_z0", "2_z0")

    assert short["path"][0] == "1_z0" and short["path"][-1] == "2_z0"
    assert short["length_by_altitude"][0] == pytest.approx(10.0)
    assert short["vertical_length"] == pytest.approx(0.0)

    assert long["path"][0] == "1_z0" and long["path"][-1] == "2_z0"
    assert long["vertical_length"] > 0.0
    assert 400 in {int(node.rsplit("_z", 1)[1]) for node in long["path"]}
    assert long["length_by_altitude"][400] > long["length_by_altitude"][0]
    assert long["share_by_altitude"][400] > 0.5
