"""
load_requests.py
================
Data structures and random scenario generation for UAV ride-pooling requests
and vehicles.

Classes
-------
* ``Request``  – a single passenger trip demand (origin, destination, time
  windows, coordinates).
* ``Vehicle``  – a UAV with a current position, time, and onboard passengers.

Functions
---------
* ``generate_requests_and_vehicles`` – randomly sample a scenario from a 2-D
  node set given a road/airspace network and a shortest-path oracle.
* ``summarize_requests`` – compute feasibility and time-slack diagnostics for
  a scenario (for logging / CSV export).

Design notes
------------
* ``Request.id`` is always a ``str`` so that it is hashable in both
  ``frozenset`` and ``dict`` keys without ambiguity.
* ``Request.t_star`` stores the *direct* origin-to-destination travel time
  (i.e. the lower bound on trip duration), NOT an absolute arrival timestamp.
  This matches the notebook convention used in ``build_trips``.
* Coordinates (``x``, ``y``, ``dest_x``, ``dest_y``) are stored in WGS-84
  decimal degrees and are used for spatial-proximity pruning in the RTV graph.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "Request",
    "Vehicle",
    "generate_requests_and_vehicles",
    "summarize_requests",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default scenario parameters
# ---------------------------------------------------------------------------

#: Simulation horizon (seconds).  Requests are released within [0, T_max).
DEFAULT_T_MAX: int = 3 * 3600  # 3 hours

#: Minimum time slack (s) between release time and latest pickup.
DEFAULT_MIN_SLACK: int = 1200  # 20 min

#: Maximum time slack (s) between release time and latest pickup.
DEFAULT_MAX_SLACK: int = 4800  # 80 min


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class Request:
    """A single passenger trip demand.

    Parameters
    ----------
    origin:
        2-D node ID of the pickup location.
    destination:
        2-D node ID of the drop-off location.
    request_time:
        Time (s) at which the request is released.
    latest_pickup:
        Hard deadline for pickup (s).  Vehicles must arrive by this time.
    earliest_arrival:
        Direct travel time (s) from origin to destination; used as the
        detour-constraint baseline (matches notebook convention for ``t_star``).
    id:
        Unique request identifier.  Stored internally as ``str``.
    x_coord, y_coord:
        WGS-84 longitude / latitude of the *origin* node (optional; required
        for spatial-proximity pruning in the RTV builder).
    dest_x, dest_y:
        WGS-84 longitude / latitude of the *destination* node (optional).
    """

    __slots__ = (
        "origin", "destination",
        "trr", "tplr", "t_star",
        "id",
        "x", "y", "dest_x", "dest_y",
        # populated by the RTV pipeline after construction
        "origin_3d", "dest_3d",
    )

    def __init__(
        self,
        origin: Any,
        destination: Any,
        request_time: float,
        latest_pickup: float,
        earliest_arrival: float,
        id: Any,
        x_coord: Optional[float] = None,
        y_coord: Optional[float] = None,
        dest_x: Optional[float] = None,
        dest_y: Optional[float] = None,
    ) -> None:
        self.origin = origin
        self.destination = destination
        self.trr = float(request_time)       # release time
        self.tplr = float(latest_pickup)     # latest pickup
        self.t_star = float(earliest_arrival)  # direct OD travel time
        self.id = str(id)
        self.x = x_coord
        self.y = y_coord
        self.dest_x = dest_x
        self.dest_y = dest_y
        # 3-D node IDs are set externally by the RTV pipeline
        self.origin_3d: Optional[Any] = None
        self.dest_3d: Optional[Any] = None

    # ------------------------------------------------------------------
    # Rich comparison – required for heapq and frozenset operations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"R{self.id}"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Request):
            return NotImplemented
        return self.id < other.id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Request):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class Vehicle:
    """A UAV available for trip assignment.

    Parameters
    ----------
    start_node:
        2-D node ID of the vehicle's current position.
    start_time:
        Earliest time (s) at which the vehicle is available.
    onboard:
        List of :class:`Request` objects already being served (default empty).
    id:
        Unique vehicle identifier.  Stored internally as ``str``.
    """

    __slots__ = ("qv", "tv", "Pv", "id")

    def __init__(
        self,
        start_node: Any,
        start_time: float,
        onboard: Optional[list] = None,
        id: Any = "vehicle",
    ) -> None:
        self.qv = start_node
        self.tv = float(start_time)
        self.Pv: list = onboard if onboard is not None else []
        self.id = str(id)

    def __repr__(self) -> str:
        return f"V{self.id}"


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


def generate_requests_and_vehicles(
    num_requests: int,
    num_vehicles: int,
    node2d_to_3d: dict,
    shortest_path_cached: Callable[[Any, Any], float],
    node_df,
    T_max: int = DEFAULT_T_MAX,
    seed: int = 42,
    min_slack: int = DEFAULT_MIN_SLACK,
    max_slack: int = DEFAULT_MAX_SLACK,
) -> tuple[list[Request], list[Vehicle]]:
    """Randomly generate a set of :class:`Request` and :class:`Vehicle` objects.

    Requests are sampled uniformly from the 2-D node set.  Trips with
    non-finite shortest-path costs are silently skipped to guarantee that
    every returned request is routable.

    Parameters
    ----------
    num_requests:
        Number of trip requests to attempt to generate.  The actual count
        may be lower if some OD pairs are unreachable or coordinates are
        missing from *node_df*.
    num_vehicles:
        Number of vehicles to generate.
    node2d_to_3d:
        Mapping from 2-D node IDs to their ground-level 3-D counterparts
        (output of ``convert_to_3d``).
    shortest_path_cached:
        Callable ``(src_3d_node, dst_3d_node) -> float`` that returns the
        shortest-path travel time in seconds.  Must accept the 3-D node IDs
        produced by *node2d_to_3d*.
    node_df:
        2-D node DataFrame; must contain columns ``node_id``, ``x_coord``,
        ``y_coord``.
    T_max:
        Simulation horizon in seconds.  Vehicle start times are drawn from
        ``[0, T_max // 2)``; request release times from ``[0, T_max - max_slack)``.
    seed:
        Random seed for reproducibility.
    min_slack:
        Minimum allowed pickup-window width (s).
    max_slack:
        Maximum allowed pickup-window width (s).

    Returns
    -------
    requests : list[Request]
        Routable requests with valid time windows.
    vehicles : list[Vehicle]
        Vehicles placed at random nodes.
    """
    random.seed(seed)
    np.random.seed(seed)

    nodes_2d = list(node2d_to_3d.keys())

    # Build a fast coordinate lookup from node_df
    node_coords_map: dict = (
        node_df.set_index("node_id")[["x_coord", "y_coord"]].to_dict("index")
    )

    # --- Vehicles ---
    vehicles: list[Vehicle] = [
        Vehicle(
            start_node=random.choice(nodes_2d),
            start_time=random.randint(0, T_max // 2),
            onboard=[],
            id=vid + 1,
        )
        for vid in range(num_vehicles)
    ]

    # --- Requests ---
    requests: list[Request] = []
    for rid in range(1, num_requests + 1):
        origin, destination = random.sample(nodes_2d, 2)

        origin_coords = node_coords_map.get(origin)
        destination_coords = node_coords_map.get(destination)
        if origin_coords is None or destination_coords is None:
            logger.debug("Skipping request %d: missing coordinates.", rid)
            continue

        release_time = random.randint(0, T_max - max_slack)

        travel_time = shortest_path_cached(
            node2d_to_3d[origin],
            node2d_to_3d[destination],
        )
        if not np.isfinite(travel_time):
            logger.debug("Skipping request %d: unreachable OD pair.", rid)
            continue

        slack = random.randint(min_slack, max_slack)

        requests.append(
            Request(
                origin=origin,
                destination=destination,
                request_time=release_time,
                latest_pickup=release_time + slack,
                earliest_arrival=travel_time,  # direct OD travel time
                id=rid,
                x_coord=origin_coords["x_coord"],
                y_coord=origin_coords["y_coord"],
                dest_x=destination_coords["x_coord"],
                dest_y=destination_coords["y_coord"],
            )
        )

    logger.info(
        "Generated %d requests (%d attempted) and %d vehicles.",
        len(requests), num_requests, len(vehicles),
    )
    return requests, vehicles


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def summarize_requests(requests: list[Request]) -> dict:
    """Return feasibility and time-slack diagnostics for a scenario.

    The returned dictionary is designed to be merged directly into a results
    CSV row (see ``save_results.py``).

    Parameters
    ----------
    requests:
        List of :class:`Request` objects for the current scenario.

    Returns
    -------
    dict
        Keys: ``num_requests``, ``infeasible_windows``, ``arrival_violations``,
        ``mean_slack``, ``min_slack``, ``p25_slack``, ``median_slack``,
        ``p75_slack``, ``max_slack``.
    """
    trr = np.array([r.trr for r in requests])
    tplr = np.array([r.tplr for r in requests])
    t_star = np.array([r.t_star for r in requests])

    slack = tplr - trr

    return {
        "num_requests": len(requests),
        # pickup window is impossible before the vehicle could even arrive
        "infeasible_windows": int(np.sum(slack < 0)),
        # request release is after the direct travel time would have elapsed
        "arrival_violations": int(np.sum(t_star < trr)),
        # time-slack distribution
        "mean_slack": float(np.mean(slack)),
        "min_slack": float(np.min(slack)),
        "p25_slack": float(np.percentile(slack, 25)),
        "median_slack": float(np.median(slack)),
        "p75_slack": float(np.percentile(slack, 75)),
        "max_slack": float(np.max(slack)),
    }
