
from __future__ import annotations

import itertools
import heapq
import multiprocessing as mp
from functools import lru_cache, partial
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from ..network import shortest_path_cached
from itertools import permutations


# Legacy notebook globals expected by these functions.
node2d_to_3d = {}
vehicle_lookup = {}
request_lookup = {}
requests = []
vehicles = []
trips = {}
trip_costs = {}
trip_to_vehicle = {}
node_df = None
link_gdf = None

def violates_constraints(route, vehicle, requests, nu, delta):
  """
  Returns TRUE if ANY constraint is violated.
  Returns FALSE if the route is fully feasible.
  """
  t = vehicle.tv  # vehicle.tv represents current time
  q = node2d_to_3d[vehicle.qv]  # vehicle.qv represents current location, converted to 3D

  onboard = set(vehicle.Pv) # vehicle.Pv is the list of passengers already on board (requests)
  picked_up = set(vehicle.Pv) # passengers that have already been picked up
  seen_pickup = set(vehicle.Pv) # set precedence

  for action, r in route:
    next_loc_2d = r.origin if action == "pickup" else r.destination
    next_loc_3d = node2d_to_3d[next_loc_2d]

    travel_time_segment = shortest_path_cached(q, next_loc_3d)
    if travel_time_segment is None:
        return True # If no path, it's a violation

    t += travel_time_segment
    q = next_loc_3d

    if action == "dropoff" and r not in seen_pickup:
      return True

    if action == "pickup":
      if t > r.tplr:
        return True

      onboard.add(r)
      seen_pickup.add(r)

    elif action == "dropoff":
      if t > r.t_star + delta:
        return True

      onboard.remove(r)

    if len(onboard) > nu:
      return True

  return False


def all_valid_permutations(vehicle, trip_requests):
    """
    Returns all valid pickup-dropoff sequences
    respecting pickup-before-dropoff constraints.
    """
    stops = []

    for p in vehicle.Pv:
      stops.append(("dropoff", p))

    for r in trip_requests:
      stops.append(("pickup", r))
      stops.append(("dropoff", r))

    for perm in permutations(stops):
      ok = True
      picked = set(vehicle.Pv)

      for action, r in perm:
        if action == "pickup":
          picked.add(r)
        else:
          if r not in picked:
            ok = False
            break

      if ok:
        yield perm



def compute_total_delay(route, vehicle):
  """
  Compute the sum of delays for all passengers after executing this route.
  """
  t = vehicle.tv
  q = node2d_to_3d[vehicle.qv] # Convert initial vehicle location to 3D

  dropoff_times = {}

  for action, r in route:

    next_loc_2d = r.origin if action == "pickup" else r.destination
    next_loc_3d = node2d_to_3d[next_loc_2d] # Convert next location to 3D

    travel_time_segment = shortest_path_cached(q, next_loc_3d)
    # Handle cases where no path exists for calculating delay (though violates_constraints should catch this)
    if travel_time_segment is None:
        return float('inf') # Return a very high cost if path is not found

    t += travel_time_segment
    q = next_loc_3d

    if action == "dropoff":
      dropoff_times[r] = t

  total_delay = 0
  for r, tdr in dropoff_times.items():
    total_delay += tdr - r.t_star

  return total_delay

from functools import lru_cache
import itertools
import heapq
import itertools

def travel(vehicle, trip_requests, nu=4, delta=1200, return_timeline=False):
    """
    Alonso–Mora (2017) exact feasibility + cost
    Fully time-consistent and cost-correct for large-scale simulations.
    """

    if len(trip_requests) == 0:
        return 0 if not return_timeline else (0, {})

    MAX_WAIT = 1800  # seconds (hard operational bound)

    requests = list(trip_requests)

    # Direct (solo) travel times
    direct_time = {
        r: shortest_path_cached(
            node2d_to_3d[r.origin],
            node2d_to_3d[r.destination]
        )
        for r in requests
    }

    # Priority queue entries:
    # (route_time, tie_breaker, sim_time, node, picked, dropped, onboard, pickup_times)
    counter = itertools.count()

    start = (
        0.0,                          # route_time (COST we minimize)
        next(counter),
        vehicle.tv,                   # simulation time
        node2d_to_3d[vehicle.qv],     # location
        frozenset(),                  # picked
        frozenset(),                  # dropped
        frozenset(),                  # onboard
        {}                             # pickup_times
    )

    pq = [start]

    # Dominance pruning:
    # key = (node, onboard, picked)
    best_seen = {}

    best_cost = float("inf")
    best_pickups = None

    while pq:
        route_time, _, sim_time, node, picked, dropped, onboard, pickup_times = heapq.heappop(pq)

        dom_key = (node, onboard, picked)
        if dom_key in best_seen and best_seen[dom_key] <= route_time:
            continue
        best_seen[dom_key] = route_time

        # Finished trip
        if len(dropped) == len(requests):
            if route_time < best_cost:
                best_cost = route_time
                best_pickups = pickup_times
            continue

        # -----------------------
        # PICKUPS
        # -----------------------
        if len(onboard) < nu:
            for r in requests:
                if r in picked:
                    continue

                t = shortest_path_cached(node, node2d_to_3d[r.origin])
                if t is None:
                    continue

                arrival_time = sim_time + t
                pickup_time = max(arrival_time, r.trr)

                # Latest pickup constraint
                if pickup_time > r.tplr:
                    continue

                # Waiting time constraint
                if pickup_time - r.trr > MAX_WAIT:
                    continue

                new_pickups = dict(pickup_times)
                new_pickups[r] = pickup_time

                heapq.heappush(pq, (
                    route_time + t,                   # cost increases by travel only
                    next(counter),
                    pickup_time,
                    node2d_to_3d[r.origin],
                    picked | {r},
                    dropped,
                    onboard | {r},
                    new_pickups
                ))

        # -----------------------
        # DROPOFFS
        # -----------------------
        for r in onboard:
            t = shortest_path_cached(node, node2d_to_3d[r.destination])
            if t is None:
                continue

            drop_time = sim_time + t

            # Correct detour check:
            pickup_time = pickup_times[r]
            if (drop_time - pickup_time) - direct_time[r] > delta:
                continue

            heapq.heappush(pq, (
                route_time + t,
                next(counter),
                drop_time,
                node2d_to_3d[r.destination],
                picked,
                dropped | {r},
                onboard - {r},
                pickup_times
            ))

    if best_cost == float("inf"):
        return None

    if return_timeline:
        return best_cost, best_pickups

    return best_cost



@lru_cache(maxsize=None)
def travel_cached(vehicle_id, trip_key):
    """
    Cached wrapper: (vehicle_id, frozenset(request_ids)) → cost or None
    """
    v = vehicle_lookup[vehicle_id]
    trip_requests = {request_lookup[rid] for rid in trip_key}
    result = travel(v, trip_requests)
    if isinstance(result, str): # Infeasible
        return None
    return result