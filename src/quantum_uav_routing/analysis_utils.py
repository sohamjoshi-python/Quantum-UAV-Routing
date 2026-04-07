from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .models import Request, Vehicle


@dataclass
class ScenarioArtifacts:
    requests: list[Request]
    vehicles: list[Vehicle]
    request_lookup: dict[str, Request]
    vehicle_lookup: dict[str, Vehicle]
    trips: dict[frozenset[str], frozenset[str]]
    trip_to_vehicle: dict[frozenset[str], list[str]]
    trip_costs: dict[tuple[frozenset[str], str], float]
    baseline_vehicle_costs: dict[str, float]
    rtv_graph_build_time: float


def generate_requests_and_vehicles(
    num_requests: int,
    num_vehicles: int,
    node2d_to_3d: dict,
    shortest_path_fn: Callable[[str, str], float],
    node_df,
    t_max: int = 3 * 3600,
    seed: int = 42,
    min_slack: int = 1200,
    max_slack: int = 4800,
) -> tuple[list[Request], list[Vehicle]]:
    random.seed(seed)
    np.random.seed(seed)

    nodes_2d = list(node2d_to_3d.keys())
    node_coords = node_df.set_index("node_id")[["x_coord", "y_coord"]].to_dict("index")

    vehicles = [
        Vehicle(
            qv=random.choice(nodes_2d),
            tv=random.randint(0, t_max // 2),
            Pv=[],
            id=str(i + 1),
        )
        for i in range(num_vehicles)
    ]

    requests: list[Request] = []
    next_id = 1

    while len(requests) < num_requests:
        origin, destination = random.sample(nodes_2d, 2)

        origin_coords = node_coords.get(origin)
        dest_coords = node_coords.get(destination)
        if origin_coords is None or dest_coords is None:
            continue

        release_time = random.randint(0, t_max - max_slack)
        travel_time = shortest_path_fn(node2d_to_3d[origin], node2d_to_3d[destination])

        if not np.isfinite(travel_time):
            continue

        slack = random.randint(min_slack, max_slack)

        requests.append(
            Request(
                origin=origin,
                destination=destination,
                trr=release_time,
                tplr=release_time + slack,
                t_star=float(travel_time),
                id=str(next_id),
                x=float(origin_coords["x_coord"]),
                y=float(origin_coords["y_coord"]),
                dest_x=float(dest_coords["x_coord"]),
                dest_y=float(dest_coords["y_coord"]),
            )
        )
        next_id += 1

    return requests, vehicles


def compute_baseline_vehicle_costs(vehicles: list[Vehicle]) -> dict[str, float]:
    return {v.id: 0.0 for v in vehicles}


def build_scenario_artifacts(
    num_requests: int,
    num_vehicles: int,
    node2d_to_3d: dict,
    shortest_path_fn: Callable[[str, str], float],
    node_df,
    build_trips_fn: Callable,
    t_max: int = 3 * 3600,
    seed: int = 123,
    nu: int = 4,
) -> ScenarioArtifacts:
    requests, vehicles = generate_requests_and_vehicles(
        num_requests=num_requests,
        num_vehicles=num_vehicles,
        node2d_to_3d=node2d_to_3d,
        shortest_path_fn=shortest_path_fn,
        node_df=node_df,
        t_max=t_max,
        seed=seed,
    )

    request_lookup = {r.id: r for r in requests}
    vehicle_lookup = {v.id: v for v in vehicles}

    t0 = time.perf_counter()
    all_trips_with_costs = build_trips_fn(requests=requests, vehicles=vehicles, nu=nu)
    t1 = time.perf_counter()

    trips: dict[frozenset[str], frozenset[str]] = {}
    trip_to_vehicle: dict[frozenset[str], list[str]] = {}
    trip_costs: dict[tuple[frozenset[str], str], float] = {}

    for t_key, v_costs in all_trips_with_costs.items():
        norm_key = frozenset(str(x) for x in t_key)
        trips[norm_key] = norm_key
        trip_to_vehicle[norm_key] = [str(v_id) for v_id in v_costs.keys()]
        for v_id, cost in v_costs.items():
            trip_costs[(norm_key, str(v_id))] = float(cost)

    baseline_vehicle_costs = compute_baseline_vehicle_costs(vehicles)

    return ScenarioArtifacts(
        requests=requests,
        vehicles=vehicles,
        request_lookup=request_lookup,
        vehicle_lookup=vehicle_lookup,
        trips=trips,
        trip_to_vehicle=trip_to_vehicle,
        trip_costs=trip_costs,
        baseline_vehicle_costs=baseline_vehicle_costs,
        rtv_graph_build_time=t1 - t0,
    )


def summarize_requests(requests: list[Request]) -> dict[str, float]:
    trr = np.array([r.trr for r in requests], dtype=float)
    tplr = np.array([r.tplr for r in requests], dtype=float)
    t_star = np.array([r.t_star for r in requests], dtype=float)

    slack = tplr - trr

    return {
        "num_requests": len(requests),
        "infeasible_windows": int(np.sum(slack < 0)),
        "arrival_violations": int(np.sum(t_star < trr)),
        "mean_slack": float(np.mean(slack)),
        "min_slack": float(np.min(slack)),
        "p25_slack": float(np.percentile(slack, 25)),
        "median_slack": float(np.median(slack)),
        "p75_slack": float(np.percentile(slack, 75)),
        "max_slack": float(np.max(slack)),
    }