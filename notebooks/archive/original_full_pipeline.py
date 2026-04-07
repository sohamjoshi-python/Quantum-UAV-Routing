import os

T_max = 3 * 3600

import pandas as pd

import geopandas as gpd

from shapely import wkt, geometry, Polygon

import h3.api.basic_int as h3

import numpy as np

import folium

from shapely import wkt

from shapely.geometry import Point, Polygon, LineString

def create_graph(network_name):

    # === CONFIGURATION ===
    H3_RESOLUTION = 7
    BUFFER_KM = 1
    network_path = f"/content/GMNS_Plus_Dataset/{network_name}"
    node_file = f"{network_path}/node.csv"
    zone_file = f"{network_path}/zone.csv"

    # === 1. Load nodes ===
    node_df = pd.read_csv(node_file)

    # Check if we need to create geometry from x_coord and y_coord
    if 'geometry' not in node_df.columns:
        if 'x_coord' in node_df.columns and 'y_coord' in node_df.columns:
            print("\nCreating geometry from x_coord and y_coord...")
            node_df["geometry"] = node_df.apply(lambda row: geometry.Point(row['x_coord'], row['y_coord']), axis=1)
        else:
            print("\nError: Cannot find geometry or coordinate columns!")
            print("Available columns:", node_df.columns.tolist())
    else:
        node_df["geometry"] = node_df["geometry"].apply(wkt.loads)

    node_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs="EPSG:4326")

    # === 2. Convex hull with buffer ===
    convex_hull = node_gdf.geometry.union_all().convex_hull
    buffer_degree = BUFFER_KM / 111  # approx conversion
    buffered_area = convex_hull.buffer(buffer_degree)
    minx, miny, maxx, maxy = buffered_area.bounds

    # === 3. Grid scan and fill with H3 ===
    step = 0.003  # ~300m
    lats = np.arange(miny, maxy, step)
    lons = np.arange(minx, maxx, step)

    h3_cells = set()
    for lat in lats:
        for lon in lons:
            if buffered_area.contains(geometry.Point(lon, lat)):
                h3_id = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
                h3_cells.add(h3_id)

    # === 4. Generate centroid CSV for zones ===
    zone_data = []

    for h in h3_cells:
        lat, lon = h3.cell_to_latlng(h)
        # Get full H3 hexagon boundary as (lat, lon) list
        boundary_latlon = h3.cell_to_boundary(h)

        # Convert to (lon, lat) for shapely
        boundary_lonlat = [(lng, lat) for lat, lng in boundary_latlon]

        # Create polygon
        hex_polygon = Polygon(boundary_lonlat)

        zone_data.append({
            "zone_id": h,
            "x_coord": lon,
            "y_coord": lat,
            "geometry": geometry.Point(lon, lat).wkt,
            "H3_geometry": hex_polygon.wkt  # Add H3 hexagon as WKT string
        })


    zone_df = pd.DataFrame(zone_data)
    zone_df.to_csv(zone_file, index=False)

def parse_graph(network_name):
    # === CONFIGURATION ===
    network_path = f"/content/GMNS_Plus_Dataset/{network_name}"

    # === LOAD FILES ===
    # Specify encoding as 'latin1' and low_memory=False to handle potential encoding issues and mixed types
    node_df = pd.read_csv(f"{network_path}/node.csv", encoding='latin1', low_memory=False)
    link_df = pd.read_csv(f"{network_path}/link.csv", encoding='latin1', low_memory=False)
    zone_df = pd.read_csv(f"{network_path}/zone.csv", encoding='latin1', low_memory=False)

    print(f"Nodes loaded: {len(node_df)}")
    print(f"Zones loaded: {len(zone_df)}")
    print(f"Links loaded: {len(link_df)}")

    # === PARSE NODE GEOMETRY FROM X/Y ===
    node_df["geometry"] = node_df.apply(lambda row: Point(row["x_coord"], row["y_coord"]), axis=1)
    node_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs="EPSG:4326")

    # === PARSE ZONE GEOMETRY FROM H3 WKT POLYGONS ===
    if "H3_geometry" in zone_df.columns:
        # Parse WKT into shapely Polygons if not already parsed
        if isinstance(zone_df["H3_geometry"].iloc[0], str):
            zone_df["H3_geometry"] = zone_df["H3_geometry"].apply(wkt.loads)

        # Set it as active geometry column
        zone_df["geometry"] = zone_df["H3_geometry"]
        zone_gdf = gpd.GeoDataFrame(zone_df, geometry="geometry", crs="EPSG:4326")
    else:
        # Fallback to generic 'geometry' column
        zone_df["geometry"] = zone_df["geometry"].apply(wkt.loads)
        zone_gdf = gpd.GeoDataFrame(zone_df, geometry="geometry", crs="EPSG:4326")

    # === PARSE LINK GEOMETRIES FROM WKT ===
    if "geometry" in link_df.columns and isinstance(link_df["geometry"].iloc[0], str):
        link_df["geometry"] = link_df["geometry"].apply(wkt.loads)
        link_gdf = gpd.GeoDataFrame(link_df, geometry="geometry", crs="EPSG:4326")
    else:
        link_gdf = None

    # === INITIALIZE FOLIUM MAP ===
    center_lat = node_df["y_coord"].mean()
    center_lon = node_df["x_coord"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")

    # === DRAW ZONE POLYGONS ===
    for _, row in zone_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, Polygon):
            folium.Polygon(
                locations=[(lat, lon) for lon, lat in geom.exterior.coords],
                color="purple",
                weight=2,
                fill=True,
                fill_opacity=0.3,
                tooltip=f"Zone ID: {row['zone_id']}"
            ).add_to(m)

    # === COLOR FUNCTION FOR LINKS ===
    def get_color_by_travel_time(time):
        if time < 1:
            return "green"
        elif time < 2:
            return "orange"
        else:
            return "red"

    # === DRAW LINKS ===
    if link_gdf is not None:
        for _, row in link_gdf.iterrows():
            if isinstance(row.geometry, LineString):
                folium.PolyLine(
                    locations=[(lat, lon) for lon, lat in row.geometry.coords],
                    color=get_color_by_travel_time(row.get("vdf_fftt", 5)),
                    weight=3,
                    opacity=0.8,
                    popup=f"vdf_fftt: {row.get('vdf_fftt', 'N/A')}"
                ).add_to(m)

    # === DRAW NODE MARKERS ===
    for _, row in node_df.iterrows():
        folium.CircleMarker(
            location=[row["y_coord"], row["x_coord"]],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Node ID: {row['node_id']}"
        ).add_to(m)

    # === DISPLAY INTERACTIVE MAP ===
    return m, node_df, link_df, link_gdf

def convert_to_3d(node_df, link_df):
  # Convert to 3d

  altitudes = [0, 50, 100, 200, 400]

  node_3d_rows = []

  for _, row in node_df.iterrows():
      for z in altitudes:
          node_3d_rows.append({
              "node_id": f"{row.node_id}_z{z}",
              "original_node_id": row.node_id,   # keep for reference
              "zone_id": row.zone_id,             # still valid
              "x_coord": row.x_coord,
              "y_coord": row.y_coord,
              "z_coord": z,                       # NEW
              "geometry": row.geometry            # 2D geometry is fine
          })
  link_3d_rows = []
  link_id_counter = 0

  for _, row in link_df.iterrows():
      for z in altitudes:
          link_3d_rows.append({
              "link_id": f"{row.link_id}_z{z}",
              "from_node_id": f"{row.from_node_id}_z{z}",
              "to_node_id": f"{row.to_node_id}_z{z}",
              "dir_flag": row.dir_flag,
              "length": row.length,
              "free_speed": 12+0.25*z,
              "link_type": "horizontal",
              "altitude": z,
              "geometry": row.geometry
          })

  CLIMB_COST_PER_METER = 0.02  # example energy cost

  for _, row in node_df.iterrows():
      for z1, z2 in zip(altitudes[:-1], altitudes[1:]):
          dz = z2 - z1

          link_3d_rows.append({
              "link_id": f"V_{row.node_id}_{z1}_{z2}",
              "from_node_id": f"{row.node_id}_z{z1}",
              "to_node_id": f"{row.node_id}_z{z2}",
              "dir_flag": 1,
              "length": dz,
              "free_speed": 4,
              "link_type": "vertical",
              "altitude": f"{z1}->{z2}",
              "energy_cost": dz * CLIMB_COST_PER_METER,
              "geometry": None
          })

  link_3d_df = pd.DataFrame(link_3d_rows)
  node_3d_df = pd.DataFrame(node_3d_rows)

  node2d_to_3d = (
    node_3d_df
    .set_index("original_node_id")["node_id"]
    .to_dict()
  )
  return node_3d_df, link_3d_df, node2d_to_3d

def horizontal_speed(z):
    return 10 + 0.15 * np.sqrt(z)

def vertical_energy(dz):
  return 5 * np.sqrt(abs(dz))

def horizontal_energy_per_meter(z):
    """
    Energy cost per meter as a function of altitude.
    Higher altitude = more efficient cruise.
    """
    base = 1.0
    reduction = 0.001 * z
    return max(0.3, base - reduction)

import networkx as nx

import numpy as np

def build_3d_graph(
    node_3d_df,
    link_3d_df,
    alpha=1.0,
    beta=0.05,
    C_CLIMB=8.0,
    C_DESCENT=1.0
):
    G = nx.DiGraph()

    for _, row in node_3d_df.iterrows():
        G.add_node(
            row["node_id"],
            x=row["x_coord"],
            y=row["y_coord"],
            z=row["z_coord"]
        )

    for _, row in link_3d_df.iterrows():

        length = row["length"]
        if length == 0 or np.isnan(length):
            length = row["geometry"].length if row["geometry"] is not None else 1e-3

        if row["link_type"] == "horizontal":
            z = float(row["altitude"])
            speed = horizontal_speed(z)
            travel_time = length / speed
            energy = horizontal_energy_per_meter(z) * length

        else:  # vertical
            z_from, z_to = map(float, row["altitude"].split("->"))
            dz = z_to - z_from
            travel_time = abs(dz) / 100.
            if dz > 0:
                energy = vertical_energy(dz)
            else:
                energy = C_DESCENT * abs(dz)

        weight = alpha * travel_time + beta * energy
        weight = travel_time

        G.add_edge(
            row["from_node_id"],
            row["to_node_id"],
            weight=weight,
            time=travel_time,
            energy=energy,
            link_type=row["link_type"]
        )

        if row["dir_flag"] == 1:
            G.add_edge(
                row["to_node_id"],
                row["from_node_id"],
                weight=weight,
                time=travel_time,
                energy=energy,
                link_type=row["link_type"]
            )

    return G

def shortest_path(start_node_id, end_node_id, G, verbose=False):
    """
    Optimized for 3D graphs using bidirectional search.
    Returns: cost (total travel time)
    """
    # Bidirectional Dijkstra is ~2x faster on grid-like urban graphs
    try:
        cost, path = nx.bidirectional_dijkstra(
            G,
            source=start_node_id,
            target=end_node_id,
            weight="weight"
        )
        if verbose:
            print(f"Path found from {start_node_id} to {end_node_id}: {path}")
        return cost
    except nx.NetworkXNoPath:
        return float('inf')

from functools import lru_cache

import numpy as np

def shortest_path_cached(src, dst):
    """
    Cache shortest path distances on the current city graph G.

    Your node IDs are strings like '3300_z400', so we MUST NOT cast to int.
    We normalize to stable string keys to avoid cache misses from numpy/object types.
    """
    # Normalize to stable, hashable node IDs
    # (np.str_ / np.int64 / object -> consistent python type)
    if isinstance(src, (np.generic,)):
        src = src.item()
    if isinstance(dst, (np.generic,)):
        dst = dst.item()

    # Your graph nodes are strings -> normalize to str
    src = str(src)
    dst = str(dst)

    return float(shortest_path(src, dst, G))

import numpy as np

from functools import lru_cache

from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import dijkstra

def _dist_from_src(src_node):
    src_node = str(src_node)
    src_idx = NODE_TO_IDX[src_node]
    dist = dijkstra(CSR, directed=False, indices=src_idx, return_predecessors=False)
    return dist

def shortest_path_fast(src_node, dst_node):
    src_node = str(src_node)
    dst_node = str(dst_node)
    dist = _dist_from_src(src_node)
    d = dist[NODE_TO_IDX[dst_node]]
    return float(d)

def shortest_path_cached(src, dst):
  return shortest_path_fast(src, dst)

class Request:
    def __init__(self, origin, destination, request_time, latest_pickup, earliest_arrival, id, x_coord=None, y_coord=None, dest_x=None, dest_y=None):
        self.origin = origin
        self.destination = destination
        self.trr = request_time            # request time
        self.tplr = latest_pickup          # latest allowed pickup time
        self.t_star = earliest_arrival     # earliest possible dropoff
        self.id = str(id)
        self.x = x_coord # Store x_coord as 'x'
        self.y = y_coord # Store y_coord as 'y'
        self.dest_x = dest_x
        self.dest_y = dest_y

    def __repr__(self):
      return f"R{self.id}"

    # Add comparison methods to make Request objects orderable and hashable
    def __lt__(self, other):
        if not isinstance(other, Request):
            return NotImplemented
        return self.id < other.id

    def __eq__(self, other):
        if not isinstance(other, Request):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

class Vehicle:
    def __init__(self, start_node, start_time, onboard=None, id="vehicle"):
        self.qv = start_node
        self.tv = start_time
        self.Pv = onboard if onboard else []
        self.id = str(id)

    def __repr__(self):
        return f"V{self.id}"

def trip_name(trip_fset):
    req_ids = sorted([r.id for r in trip_fset])
    return "T[" + ",".join(req_ids) + "]"

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

from itertools import permutations

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

import itertools

import networkx as nx

def get_trip_key(trip):
    """
    trip: iterable of request IDs (ints)
    returns: frozenset[int]
    """
    return frozenset(int(r) for r in trip)

from functools import lru_cache

def feasible_cached(vehicle_id, trip_key):
    """
    Cache only FEASIBILITY, not cost.
    trip_key is frozenset of request IDs.
    """
    v = vehicle_lookup[vehicle_id]
    trip = {request_lookup[rid] for rid in trip_key}
    return travel(v, trip) is not None

import itertools

import numpy as np

MAX_WAIT = 1800

def _direct_time(r):
    return shortest_path_cached(node2d_to_3d[r.origin], node2d_to_3d[r.destination])

def _sp(a_node2d, b_node2d):
    return shortest_path_cached(node2d_to_3d[a_node2d], node2d_to_3d[b_node2d])

def travel_single_exact(vehicle, r, delta=1200):
    """
    Exact travel for |trip|=1 using same constraints as travel().
    Cost returned = travel time only (waiting does not add to cost).
    """
    t_to_pick = _sp(vehicle.qv, r.origin)
    if t_to_pick is None:
        return None

    arrival = vehicle.tv + t_to_pick
    pickup = max(arrival, r.trr)

    if pickup > r.tplr:
        return None
    if pickup - r.trr > MAX_WAIT:
        return None

    t_to_drop = _sp(r.origin, r.destination)
    if t_to_drop is None:
        return None

    drop = pickup + t_to_drop

    direct = _direct_time(r)
    if direct is None:
        return None

    # detour constraint (same as travel())
    if (drop - pickup) - direct > delta:
        return None

    # cost = travel time only (matches travel() route_time)
    return float(t_to_pick + t_to_drop)

def travel_pair_exact(vehicle, r1, r2, delta=1200):
    """
    Exact travel for |trip|=2 by enumerating all precedence-valid event sequences.
    Events: P1,P2,D1,D2 with Pi before Di.
    """
    direct1 = _direct_time(r1)
    direct2 = _direct_time(r2)
    if direct1 is None or direct2 is None:
        return None

    events = ["P1", "D1", "P2", "D2"]

    best_cost = None

    def valid(seq):
        return seq.index("P1") < seq.index("D1") and seq.index("P2") < seq.index("D2")

    for seq in itertools.permutations(events):
        if not valid(seq):
            continue

        cur_node = vehicle.qv
        sim_time = vehicle.tv
        cost = 0.0
        pickup_time = {}

        feasible = True

        for e in seq:
            if e == "P1":
                t = _sp(cur_node, r1.origin)
                if t is None:
                    feasible = False; break
                sim_time += t
                cost += t
                sim_time = max(sim_time, r1.trr)
                if sim_time > r1.tplr or sim_time - r1.trr > MAX_WAIT:
                    feasible = False; break
                pickup_time["r1"] = sim_time
                cur_node = r1.origin

            elif e == "P2":
                t = _sp(cur_node, r2.origin)
                if t is None:
                    feasible = False; break
                sim_time += t
                cost += t
                sim_time = max(sim_time, r2.trr)
                if sim_time > r2.tplr or sim_time - r2.trr > MAX_WAIT:
                    feasible = False; break
                pickup_time["r2"] = sim_time
                cur_node = r2.origin

            elif e == "D1":
                t = _sp(cur_node, r1.destination)
                if t is None:
                    feasible = False; break
                sim_time += t
                cost += t
                # detour check exactly as travel()
                if (sim_time - pickup_time["r1"]) - direct1 > delta:
                    feasible = False; break
                cur_node = r1.destination

            elif e == "D2":
                t = _sp(cur_node, r2.destination)
                if t is None:
                    feasible = False; break
                sim_time += t
                cost += t
                if (sim_time - pickup_time["r2"]) - direct2 > delta:
                    feasible = False; break
                cur_node = r2.destination

        if feasible:
            if best_cost is None or cost < best_cost:
                best_cost = cost

    return None if best_cost is None else float(best_cost)

import multiprocessing as mp

from functools import partial

from collections import defaultdict

def check_trip_feasibility(candidate_data, delta, vehicle_list):
    """
    Worker function for parallel processing.
    Checks if any vehicle in the list can serve the candidate trip.
    """
    trip_id_set, req_objects = candidate_data
    v_costs = {}
    for v in vehicle_list:
        # travel() is the heavy lifting
        cost = travel(v, set(req_objects), delta=delta)
        if cost is not None:
            v_costs[v.id] = cost

    if v_costs:
        return (frozenset(trip_id_set), v_costs)
    return None

def is_spatially_compatible(r1, r2, threshold=3.5):
    """
    Relaxed heuristic to ensure 100% service rate.
    Only rejects if the destinations are wildly divergent (over 3.5x origin dist).
    """
    d_origin = np.sqrt((r1.x - r2.x)**2 + (r1.y - r2.y)**2)
    d_dest = np.sqrt((r1.dest_x - r2.dest_x)**2 + (r1.dest_y - r2.dest_y)**2)

    # If the requests are very close to each other, always allow them
    if d_origin < 2.0:
        return True

    return d_dest < (d_origin * threshold)

def check_worker(cand, delta):
    # Unpack the candidate data and call the feasibility check
    trip_ids, req_objs, v_list = cand
    return check_trip_feasibility((trip_ids, req_objs), delta, v_list)

import numpy as np

from collections import defaultdict

import itertools

def build_trips(requests, vehicles, delta=1200, nu=2, max_dist_km=6.0):
    """
    FAST + EXACT for |trip| in {1,2}.

    - Singletons: exact feasibility/cost with constant # shortest path calls.
    - Pairs: exact feasibility/cost by enumerating precedence-valid sequences,
      but using a precomputed distance table (so O(1) shortest path calls per pair).
    - Uses spatial pruning for pair generation.
    - Uses only vehicles that served both singletons (common vehicles).
    """

    # ---------- helpers ----------
    INF = float("inf")
    MAX_WAIT = 1800  # MUST match travel()

    def sp2d(a2d, b2d):
        """Shortest path cost between 2D nodes via your cached 3D mapping."""
        return shortest_path_cached(node2d_to_3d[a2d], node2d_to_3d[b2d])

    def direct_time(r):
        # Use r.t_star if it is already the direct OD travel time; else compute.
        # In your notebook, r.t_star appears to be "direct shortest travel time".
        # If you're not 100% sure, replace with: return sp2d(r.origin, r.destination)
        return float(r.t_star)

    def single_cost_exact(v, r):
        """
        Exact singleton feasibility + cost consistent with travel():
        - cost counts travel time only (not waiting),
        - respects pickup window and MAX_WAIT,
        - respects per-request detour constraint delta.
        """
        t_v_to_o = sp2d(v.qv, r.origin)
        if not np.isfinite(t_v_to_o) or t_v_to_o == INF:
            return None

        arr = v.tv + t_v_to_o
        pick = max(arr, r.trr)

        if pick > r.tplr:
            return None
        if pick - r.trr > MAX_WAIT:
            return None

        t_o_to_d = sp2d(r.origin, r.destination)
        if not np.isfinite(t_o_to_d) or t_o_to_d == INF:
            return None

        drop = pick + t_o_to_d
        ddir = direct_time(r)

        # detour constraint in travel(): (drop - pickup) - direct <= delta
        if (drop - pick) - ddir > delta:
            return None

        # travel() objective is route_time (travel time only)
        return float(t_v_to_o + t_o_to_d)

    def pair_cost_exact_fast(v, r1, r2):
        """
        Exact |trip|=2 feasibility/cost by enumerating precedence-valid sequences,
        but using a precomputed distance table (so ~10 SP calls total).
        """
        # Precompute all needed legs ONCE
        # Nodes involved (2D): start, o1, d1, o2, d2
        S  = v.qv
        O1, D1 = r1.origin, r1.destination
        O2, D2 = r2.origin, r2.destination

        # Distances (some duplicates are fine; cached anyway)
        d = {}
        def get(a, b):
            key = (a, b)
            if key in d:
                return d[key]
            val = sp2d(a, b)
            d[key] = val
            return val

        # direct times
        dir1 = direct_time(r1)
        dir2 = direct_time(r2)
        if not np.isfinite(dir1) or not np.isfinite(dir2):
            return None

        # Valid event sequences for 2 requests (P before D for each)
        # We'll evaluate these 6 permutations:
        # P1 P2 D1 D2
        # P1 P2 D2 D1
        # P1 D1 P2 D2
        # P2 P1 D1 D2
        # P2 P1 D2 D1
        # P2 D2 P1 D1
        sequences = [
            ("P1","P2","D1","D2"),
            ("P1","P2","D2","D1"),
            ("P1","D1","P2","D2"),
            ("P2","P1","D1","D2"),
            ("P2","P1","D2","D1"),
            ("P2","D2","P1","D1"),
        ]

        best_cost = None

        for seq in sequences:
            cur = S
            sim_time = float(v.tv)
            cost = 0.0
            ptime1 = None
            ptime2 = None
            feasible = True

            for e in seq:
                if e == "P1":
                    leg = get(cur, O1)
                    if not np.isfinite(leg) or leg == INF:
                        feasible = False; break
                    sim_time += leg
                    cost += leg
                    sim_time = max(sim_time, r1.trr)
                    if sim_time > r1.tplr or (sim_time - r1.trr) > MAX_WAIT:
                        feasible = False; break
                    ptime1 = sim_time
                    cur = O1

                elif e == "P2":
                    leg = get(cur, O2)
                    if not np.isfinite(leg) or leg == INF:
                        feasible = False; break
                    sim_time += leg
                    cost += leg
                    sim_time = max(sim_time, r2.trr)
                    if sim_time > r2.tplr or (sim_time - r2.trr) > MAX_WAIT:
                        feasible = False; break
                    ptime2 = sim_time
                    cur = O2

                elif e == "D1":
                    leg = get(cur, D1)
                    if not np.isfinite(leg) or leg == INF or ptime1 is None:
                        feasible = False; break
                    sim_time += leg
                    cost += leg
                    if (sim_time - ptime1) - dir1 > delta:
                        feasible = False; break
                    cur = D1

                elif e == "D2":
                    leg = get(cur, D2)
                    if not np.isfinite(leg) or leg == INF or ptime2 is None:
                        feasible = False; break
                    sim_time += leg
                    cost += leg
                    if (sim_time - ptime2) - dir2 > delta:
                        feasible = False; break
                    cur = D2

            if feasible:
                if best_cost is None or cost < best_cost:
                    best_cost = cost

        return None if best_cost is None else float(best_cost)

    # ---------- main ----------
    trips = {}
    viable_vids_for_req = defaultdict(list)
    vehicle_by_id = {v.id: v for v in vehicles}

    # 1) Singletons (exact, fast)
    for r in requests:
        trip = frozenset((r.id,))
        v_costs = {}
        for v in vehicles:
            c = single_cost_exact(v, r)
            if c is not None:
                v_costs[v.id] = c
                viable_vids_for_req[r.id].append(v.id)
        if v_costs:
            trips[trip] = v_costs

    if nu <= 1:
        return trips

    # 2) Pairs (exact, but fast distance-table evaluation)
    R = requests
    n = len(R)
    for i in range(n):
        r1 = R[i]
        for j in range(i + 1, n):
            r2 = R[j]

            # Spatial prune
            dist = np.sqrt((r1.x - r2.x)**2 + (r1.y - r2.y)**2)
            if dist > max_dist_km:
                continue

            # Common vehicles that can serve both as singletons
            common_vids = set(viable_vids_for_req[r1.id]) & set(viable_vids_for_req[r2.id])
            if not common_vids:
                continue

            pair = frozenset((r1.id, r2.id))
            v_costs = {}

            for vid in common_vids:
                v = vehicle_by_id[vid]
                c = pair_cost_exact_fast(v, r1, r2)
                if c is not None:
                    v_costs[vid] = c

            if v_costs:
                trips[pair] = v_costs

    return trips

import matplotlib.pyplot as plt

import networkx as nx

import geopandas as gpd

from shapely.geometry import LineString

def visualize_rv_on_map(requests, vehicles, RV, TRIPS, node_df, link_gdf=None):
    """
    Visualizes the RV graph overlaying the actual GMNS street network.
    """

    # --- 1. Setup Coordinate Mapping ---
    # Create a dictionary to look up (x, y) coordinates for any node ID
    # structure: {node_id: (x_coord, y_coord)}
    node_coords = node_df.set_index('node_id')[['x_coord', 'y_coord']].to_dict('index')

    # Define positions for graph nodes based on their real-world location
    pos = {}

    # Requests are positioned at their Origin Node
    for r in requests:
        if r.origin in node_coords:
            coords = node_coords[r.origin]
            pos[r] = (coords['x_coord'], coords['y_coord'])

    # Vehicles are positioned at their Current Node (qv)
    for v in vehicles:
        if v.qv in node_coords:
            coords = node_coords[v.qv]
            pos[v] = (coords['x_coord'], coords['y_coord'])

    # --- 2. Build the Graph Structure ---
    G = nx.Graph()
    G.add_nodes_from(requests)
    G.add_nodes_from(vehicles)

    # Add Servability Edges (Green) - Vehicle to Request
    rv_edges = []
    for r, v in RV:
        if r in pos and v in pos: # Ensure both have valid coords
            rv_edges.append((r, v))

    # Add Shareability Edges (Red) - Request to Request
    rr_edges = set()
    for trip_requests in TRIPS.keys():
        if len(trip_requests) >= 2:
            import itertools
            for r1_id, r2_id in itertools.combinations(trip_requests, 2):
                r1 = request_lookup[r1_id]
                r2 = request_lookup[r2_id]
                if r1 in pos and r2 in pos:
                    rr_edges.add((r1, r2))

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(15, 15))

    # A. Draw the Background Map (Street Network)
    if link_gdf is not None:
        # If you have the geometry parsed from earlier steps
        link_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)
    else:
        print("Warning: link_gdf not provided, skipping background map.")

    # B. Draw the Edges (Connections)
    # Shareability (Red Dotted)
    nx.draw_networkx_edges(G, pos, edgelist=list(rr_edges), ax=ax,
                           edge_color='red', style='dotted', width=2, alpha=0.6, label='Shareability')

    # Servability (Green Solid)
    nx.draw_networkx_edges(G, pos, edgelist=rv_edges, ax=ax,
                           edge_color='green', style='solid', width=1.5, alpha=0.5, label='Servability')

    # C. Draw the Nodes
    # Requests (Orange Stars)
    nx.draw_networkx_nodes(G, pos, nodelist=requests, ax=ax,
                           node_color='#FF6347', node_shape='*', node_size=150, label='Requests')

    # Vehicles (Green Circles)
    nx.draw_networkx_nodes(G, pos, nodelist=vehicles, ax=ax,
                           node_color='#32CD32', node_shape='o', node_size=150, label='Vehicles')

    # D. Labels (Optional - can be messy on a map)
    # Only label if there are few items, otherwise it overlaps too much
    if len(requests) + len(vehicles) < 50:
        labels = {node: getattr(node, 'id', str(node)) for node in G.nodes() if node in pos}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', ax=ax)

    # Aesthetics
    plt.title("Geo-Spatial RV Graph: Shareability & Servability on GMNS Network", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Create a custom legend manually because nx.draw handles legends poorly
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#FF6347', markersize=15, label='Request (Origin)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#32CD32', markersize=10, label='Vehicle (Location)'),
        Line2D([0], [0], color='red', linestyle=':', linewidth=2, label='Shareable (R-R)'),
        Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='Servable (R-V)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

def ilp_model_stats(model):
    vars_ = model.variables()
    num_vars = len(vars_)
    num_constraints = len(model.constraints)

    # Type counts (PuLP categories: "Binary", "Integer", "Continuous")
    num_bin = sum(1 for v in vars_ if v.cat == "Binary")
    num_int = sum(1 for v in vars_ if v.cat == "Integer")
    num_cont = sum(1 for v in vars_ if v.cat == "Continuous")

    # Nonzeros in constraint matrix (approx): count variable appearances in constraints
    # This uses PuLP's internal constraint representation.
    nnz = 0
    for cname, c in model.constraints.items():
        # c is an LpConstraint, c.items() gives (LpVariable, coeff)
        nnz += len(c.items())

    return {
        "ilp_num_vars": num_vars,
        "ilp_num_constraints": num_constraints,
        "ilp_num_binary_vars": num_bin,
        "ilp_num_integer_vars": num_int,
        "ilp_num_continuous_vars": num_cont,
        "ilp_num_nonzero_coeffs": nnz,
    }

import pulp

def optimal_assignment(
    requests,
    vehicles,
    trips,
    trip_costs,
    trip_to_vehicle,
    baseline_vehicle_costs,
    ignore_cost=10000
):
    """
    PNAS-style ILP for trip-vehicle assignment with a GREEDY WARM START,
    not a fixed greedy solution.

    Returns
    -------
    Sigma_opt : list of (t_id, v_id)
    ignored_requests : list of request IDs
    stats : dict
    """

    # -------------------------------------------------
    # 1. Create ILP model
    # -------------------------------------------------
    model = pulp.LpProblem("OptimalAssignment", pulp.LpMinimize)

    # -------------------------------------------------
    # 2. Decision variables
    # ε_(i,j) and χ_k
    # -------------------------------------------------
    epsilon = {}
    for t_id in trips:
        for v_id in trip_to_vehicle[t_id]:
            if (t_id, v_id) in trip_costs:
                epsilon[(t_id, v_id)] = pulp.LpVariable(
                    f"eps_{t_id}_{v_id}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpBinary
                )

    chi = {
        r_id: pulp.LpVariable(
            f"chi_{r_id}",
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary
        )
        for r_id in requests
    }

    # -------------------------------------------------
    # 3. Objective
    # incremental delay + ignore penalty
    # -------------------------------------------------
    model += (
        pulp.lpSum(
            (trip_costs[(t_id, v_id)] - baseline_vehicle_costs[v_id])
            * epsilon[(t_id, v_id)]
            for (t_id, v_id) in epsilon
        )
        +
        pulp.lpSum(ignore_cost * chi[r_id] for r_id in requests)
    )

    # -------------------------------------------------
    # 4. Constraints
    # -------------------------------------------------

    # (8) Each vehicle assigned at most one trip
    for v_id in vehicles:
        model += (
            pulp.lpSum(
                epsilon[(t_id, v_id)]
                for (t_id, vv) in epsilon
                if vv == v_id
            ) <= 1
        )

    # (9) Each request served exactly once or ignored
    for r_id in requests:
        model += (
            pulp.lpSum(
                epsilon[(t_id, v_id)]
                for (t_id, v_id) in epsilon
                if r_id in trips[t_id]
            ) + chi[r_id] == 1
        )

    # -------------------------------------------------
    # 5. Greedy warm start (Algorithm 2 idea)
    # IMPORTANT: warm start only, DO NOT fix variables
    # -------------------------------------------------

    # Initialize all variables to zero
    for var in epsilon.values():
        var.setInitialValue(0)

    for var in chi.values():
        var.setInitialValue(0)

    sorted_eps = sorted(
        epsilon.keys(),
        key=lambda x: (
            -len(trips[x[0]]),   # trip size descending
            trip_costs[x]        # cost ascending
        )
    )

    assigned_requests = set()
    assigned_vehicles = set()

    for (t_id, v_id) in sorted_eps:
        if v_id in assigned_vehicles:
            continue
        if any(r in assigned_requests for r in trips[t_id]):
            continue

        # Warm start only
        epsilon[(t_id, v_id)].setInitialValue(1)

        assigned_vehicles.add(v_id)
        assigned_requests |= trips[t_id]

    for r_id in requests:
        if r_id not in assigned_requests:
            chi[r_id].setInitialValue(1)

    # -------------------------------------------------
    # 6. Solve
    # -------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True)
    model.solve(solver)

    # -------------------------------------------------
    # 7. Extract solution
    # -------------------------------------------------
    Sigma_opt = [
        (t_id, v_id)
        for (t_id, v_id), var in epsilon.items()
        if var.value() is not None and var.value() > 0.5
    ]

    ignored_requests = [
        r_id
        for r_id, var in chi.items()
        if var.value() is not None and var.value() > 0.5
    ]

    stats = ilp_model_stats(model)

    return Sigma_opt, ignored_requests, stats

import time

import math

import numpy as np

from scipy.optimize import linear_sum_assignment

def greedy_assignment_pnas_instrumented(trips, trip_costs, trip_to_vehicle, nu):
    """
    PNAS Algorithm 2 greedy assignment + instrumentation.

    Returns:
      Sigma: list[(trip_id, vehicle_id)]
      ignored: list[request_id]
      stats: dict with timing + growth counters
    """
    Rok, Vok = set(), set()
    Sigma = []

    # ---- counters ----
    edges_total = 0
    sort_work = 0.0
    accepted = 0

    # ---- timing ----
    t_build = 0.0
    t_sort = 0.0
    t_select = 0.0

    for k in range(int(nu), 0, -1):
        # (A) Build candidate list S_k
        tb0 = time.perf_counter()
        Sk = []
        for t_id in trips.keys():
            if len(t_id) != k:
                continue
            for v_id in trip_to_vehicle.get(t_id, []):
                c = trip_costs.get((t_id, v_id), None)
                if c is not None:
                    Sk.append((float(c), t_id, v_id))
        tb1 = time.perf_counter()
        t_build += (tb1 - tb0)

        m = len(Sk)
        edges_total += m
        if m > 1:
            sort_work += m * math.log2(m)

        # (B) Sort by increasing cost
        ts0 = time.perf_counter()
        Sk.sort(key=lambda x: x[0])
        ts1 = time.perf_counter()
        t_sort += (ts1 - ts0)

        # (C) Select feasible assignments
        tl0 = time.perf_counter()
        for _, t_id, v_id in Sk:
            if v_id in Vok:
                continue
            if any(r in Rok for r in t_id):
                continue
            Sigma.append((t_id, v_id))
            Vok.add(v_id)
            Rok |= set(t_id)
            accepted += 1
        tl1 = time.perf_counter()
        t_select += (tl1 - tl0)

    all_req_ids = [r.id for r in requests]
    ignored = [rid for rid in all_req_ids if rid not in Rok]

    stats = {
        "time_greedy_build_candidates": t_build,
        "time_greedy_sort": t_sort,
        "time_greedy_select": t_select,
        "time_greedy_total": (t_build + t_sort + t_select),
        "greedy_edges_total": int(edges_total),
        "greedy_sort_work": float(sort_work),
        "greedy_assignments": int(accepted),
    }
    return Sigma, ignored, stats

def compute_baseline_vehicle_costs(vehicles):
    costs = {}
    for v in vehicles:
        cost = travel(v, set())
        costs[v.id] = cost if cost is not None else 0
    return costs

import time

import math

def greedy_assignment_pnas(
    requests,
    vehicles,
    trips,
    trip_costs,
    trip_to_vehicle,
    nu
):
    """
    PNAS Algorithm 2 greedy assignment.

    Returns
    -------
    Sigma_greedy : list of (trip_id, vehicle_id)
    ignored_requests : list of request_ids
    greedy_stats : dict
    ilp_stats : dict   # schema-compatible placeholder
    """

    Rok = set()
    Vok = set()
    Sigma_greedy = []

    edges_total = 0
    sort_work   = 0.0
    accepted    = 0

    t_build  = 0.0
    t_sort   = 0.0
    t_select = 0.0

    # (A) Pre-index trip_costs by t_id to avoid repeated (frozenset, v_id) lookups.
    # trip_costs is keyed by (frozenset, v_id), which at large n suffers hash
    # collisions as the dict grows to ~1M entries. Re-indexing as
    # costs_by_trip[t_id][v_id] = cost reduces each lookup to two small-dict
    # accesses and keeps per-edge cost O(1) regardless of total dict size.
    tb0 = time.perf_counter()
    costs_by_trip = {}
    for t_id in trips:
        v_costs = {}
        for v_id in trip_to_vehicle.get(t_id, []):
            c = trip_costs.get((t_id, v_id), None)
            if c is not None:
                v_costs[v_id] = float(c)
        if v_costs:
            costs_by_trip[t_id] = v_costs

    # Build buckets by trip size in the same pass — O(T) total, one scan only.
    buckets = {}
    for t_id, v_costs in costs_by_trip.items():
        k = len(t_id)
        bucket = buckets.setdefault(k, [])
        for v_id, c in v_costs.items():
            bucket.append((c, t_id, v_id))
    tb1 = time.perf_counter()
    t_build = tb1 - tb0

    # (B) Process buckets from largest trip size down to 1
    for k in range(int(nu), 0, -1):
        Sk = buckets.get(k, [])

        m = len(Sk)
        edges_total += m
        if m > 1:
            sort_work += m * math.log2(m)

        ts0 = time.perf_counter()
        Sk.sort(key=lambda x: x[0])
        ts1 = time.perf_counter()
        t_sort += (ts1 - ts0)

        tl0 = time.perf_counter()
        for _, t_id, v_id in Sk:
            if v_id in Vok:
                continue
            if any(r in Rok for r in t_id):
                continue
            Sigma_greedy.append((t_id, v_id))
            Vok.add(v_id)
            Rok.update(t_id)
            accepted += 1
        tl1 = time.perf_counter()
        t_select += (tl1 - tl0)

    ignored_requests = [r_id for r_id in requests if r_id not in Rok]

    greedy_stats = {
        "time_greedy_build_candidates": t_build,
        "time_greedy_sort":             t_sort,
        "time_greedy_select":           t_select,
        "time_greedy_total":            (t_build + t_sort + t_select),
        "greedy_edges_total":           int(edges_total),
        "greedy_sort_work":             float(sort_work),
        "greedy_assignments":           int(accepted),
    }

    ilp_stats = {
        "ilp_num_vars":           None,
        "ilp_num_constraints":    None,
        "ilp_num_integer_vars":   None,
        "ilp_num_nonzero_coeffs": None,
    }

    return Sigma_greedy, ignored_requests, greedy_stats, ilp_stats

import time

def classical_greedy_run(
    metadata,
    baseline_vehicle_costs,
    csv_filename="results.csv",
    stats={},
    seed=123,
    trial=1
):
    """
    Classical greedy run with clean timing semantics:

    - total_run_time: end-to-end time inside THIS function (solver pipeline only)
    - solve_time: time spent inside greedy_assignment_pnas() (greedy solve time)
    - rtv_graph_build_time: scenario precompute time from final_trips (t1 - t0), if available
    - scenario_total_time: scenario pipeline time (if provided via stats), optional
    """

    # Per-solver wall time (does NOT include scenario generation unless you call this inside it)
    run_start = time.perf_counter()

    # ----------------------------
    # GREEDY SOLVE
    # ----------------------------
    t2 = time.perf_counter()

    Sigma_greedy, ignored, greedy_stats, ilp_stats = greedy_assignment_pnas(
        requests=[r.id for r in requests],
        vehicles=[v.id for v in vehicles],
        trips=trips,
        trip_costs=trip_costs,
        trip_to_vehicle=trip_to_vehicle,
        nu=max(len(t) for t in trips) if trips else 0
    )

    t3 = time.perf_counter()

    # ----------------------------
    # POST-PROCESSING
    # ----------------------------
    served_requests = set()
    served_trips = []

    for trip_ids, vid in Sigma_greedy:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        v_obj = vehicle_lookup[vid]
        served_trips.append((trip_objs, v_obj))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = 100.0 * len(served_requests) / len(requests) if requests else 0.0

    # ----------------------------
    # METRICS (RAW)
    # ----------------------------
    detours = []
    waiting = []
    VMT = 0.0

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

    run_end = time.perf_counter()

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print("Number of nodes in the graph:", node_df["node_id"].nunique())
    print("Number of vehicles:", len(vehicles))
    print("Number of requests:", len(requests))
    print(f"% Serviced Requests: {percent_serviced:.2f}%")

    print("Greedy Assignment:")
    for trip, v in served_trips:
        print(" ", trip_name(trip), "→", v)

    print("\nIgnored Requests:", ignored)

    print("\n--- Performance Metrics ---")
    print(f"Average Waiting Time: {avg_wait:.2f} s")
    print(f"Maximum Waiting Time: {max_wait:.2f} s")
    print(f"Average Detour Factor: {avg_detour:.3f}")
    print(f"Maximum Detour Factor: {max_detour:.3f}")
    print(f"Vehicle Miles Traveled (VMT): {VMT:.2f}")

    # IMPORTANT: do not report "t3 - t0" as solver runtime; that's scenario-cumulative.
    print(f"Total Run Time (solver only): {run_end - run_start:.9f} s")
    print(f"Greedy Solve Time:           {t3 - t2:.9f} s")

    if "t0" in globals() and "t1" in globals():
        print(f"RTV Graph Build Time:        {t1 - t0:.2f} s")

    # ----------------------------
    # SAVE METRICS
    # ----------------------------
    metrics_to_save = {
        "city": metadata.get("city", ""),
        "run_type": "ClassicalGreedy",
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
        "total_run_time": run_end - run_start,   # solver pipeline wall time
        "solve_time": t3 - t2,                   # greedy solver call time

        # Scenario build time (if available from final_trips):
        "rtv_graph_build_time": (t1 - t0) if ("t0" in globals() and "t1" in globals()) else None,

        "seed": seed,
        "trial": trial,
    }

    # Add scenario-level time if you computed it in the outer loop and stuffed into stats
    # (recommended): stats["scenario_total_time"] = scenario_end - scenario_start

    metrics_to_save.update(stats)
    metrics_to_save.update(ilp_stats)
    metrics_to_save.update(greedy_stats)

    save_metrics_to_csv(csv_filename, metrics_to_save)
    return Sigma_greedy, ignored

from collections import defaultdict

import numpy as np

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
      maximize benefit w_t = ignore_cost*|t| - trip_costs[t]
      QUBO minimization uses diagonal -w_t
    """

    rng = np.random.default_rng(seed)

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
        benefit = float(ignore_cost) * len(tkey) - c
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

    def addQ(i, j, v):
        # store symmetric in dict (your downstream code sometimes assumes full dict)
        Qdict[(i, j)] += float(v)
        if i != j:
            Qdict[(j, i)] += float(v)

    def add_square_constraint(u, a, b, weight):
        """
        Add weight*(u - a - b)^2 to Q.
        Expansion: weight*(u + a + b - 2ua - 2ub + 2ab)
        """
        addQ(u, u, weight)
        addQ(a, a, weight)
        addQ(b, b, weight)

        addQ(u, a, -2 * weight)
        addQ(u, b, -2 * weight)
        addQ(a, b,  2 * weight)

    # -------------------------
    # 4) Objective on diagonal
    # -------------------------
    for t in trip_vars:
        i = idx[t]
        w_t = float(ignore_cost) * len(t) - float(trip_costs[t])
        addQ(i, i, -w_t)

    # --------------------------------------------------------
    # 5) For each request: sum(x in S_r) <= 1 via merge tree
    # --------------------------------------------------------
    for r, tlist in req_to_trip_idxs.items():
        if len(tlist) <= 1:
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
                add_square_constraint(u, a, b, M)
                nxt.append(u)

            current = nxt
            level += 1

    if next_aux != N:
        raise RuntimeError(f"Aux mismatch: predicted {aux_needed}, used {next_aux - T}")

    # -------------------------
    # 6) Return format
    # -------------------------
    if not return_numpy:
        return dict(Qdict), all_vars

    # Dense convert only at the end (still O(N^2), so keep N small via caps)
    Q = np.zeros((N, N), dtype=float)
    for (i, j), v in Qdict.items():
        Q[i, j] = v
    return Q, all_vars

import numpy as np

from scipy.optimize import minimize

import dimod

from dwave.samplers import SimulatedAnnealingSampler

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

def solve_qubo_qiskit_real(Q, reps=2, max_iter=10, real=False):
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
        sampleset = sampler.sample_qubo(qubo, num_reads=1000)

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

import numpy as np

import dimod

from dwave.samplers import SimulatedAnnealingSampler

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

from collections import defaultdict

import heapq

import numpy as np

def compress_qubo_fast_topk(
    Q,
    target_max=1.0,
    K=20,
    keep_diagonal=True,
    min_abs_keep=0.0,
    rescale_after=True,
    return_stats=False,
):
    def to_upper_dict(Q_in):
        if isinstance(Q_in, dict):
            out = {}
            for (i, j), c in Q_in.items():
                if c == 0:
                    continue
                i = int(i); j = int(j); c = float(c)
                if i <= j:
                    out[(i, j)] = out.get((i, j), 0.0) + c
                else:
                    out[(j, i)] = out.get((j, i), 0.0) + c
            return out

        if isinstance(Q_in, np.ndarray):
            # FAST: grab nonzeros from upper triangle (no Python N^2 scan)
            Qtri = np.triu(Q_in)
            ii, jj = np.nonzero(Qtri)
            out = {(int(i), int(j)): float(Qtri[i, j]) for i, j in zip(ii, jj)}
            return out

        raise TypeError("Q must be dict or numpy ndarray")

    def scale_to_target(Qd, tgt):
        if not Qd:
            return Qd, 1.0
        m = max(abs(c) for c in Qd.values())
        if m == 0:
            return Qd, 1.0
        s = float(tgt) / m
        if s == 1.0:
            return Qd, 1.0
        return {k: v * s for k, v in Qd.items()}, s

    Qd0 = to_upper_dict(Q)
    Qd, s1 = scale_to_target(Qd0, target_max)

    out = {}
    heaps = defaultdict(list)

    for (i, j), c in Qd.items():
        if i == j:
            if keep_diagonal and abs(c) >= min_abs_keep:
                out[(i, j)] = c
            continue

        a = abs(c)
        if a < min_abs_keep:
            continue

        hi = heaps[i]
        if len(hi) < K:
            heapq.heappush(hi, (a, i, j))
        elif a > hi[0][0]:
            heapq.heapreplace(hi, (a, i, j))

        hj = heaps[j]
        if len(hj) < K:
            heapq.heappush(hj, (a, i, j))
        elif a > hj[0][0]:
            heapq.heapreplace(hj, (a, i, j))

    keep_edges = set()
    for h in heaps.values():
        for _, i, j in h:
            keep_edges.add((i, j))

    for (i, j), c in Qd.items():
        if i != j and (i, j) in keep_edges:
            out[(i, j)] = c

    if rescale_after:
        out, s2 = scale_to_target(out, target_max)
    else:
        s2 = 1.0

    if not return_stats:
        return out

    n_vars = (max(max(i, j) for (i, j) in out.keys()) + 1) if out else 0
    n_couplers = sum(1 for (i, j) in out.keys() if i != j)
    n_diag = sum(1 for (i, j) in out.keys() if i == j)

    return out, {
        "K": K,
        "scale_factor_1": s1,
        "scale_factor_2": s2,
        "n_vars": n_vars,
        "n_diag": n_diag,
        "n_couplers": n_couplers,
    }

import random

import numpy as np

def generate_requests_and_vehicles(
    num_requests,
    num_vehicles,
    node2d_to_3d,
    shortest_path_cached,
    node_df, # New argument
    T_max = 3*3600,
    seed=42,
    min_slack=1200,
    max_slack=4800,
):
    random.seed(seed)
    np.random.seed(seed)

    nodes_2d = list(node2d_to_3d.keys())
    # Create a map for 2D node IDs to their coordinates
    node_coords_map = node_df.set_index('node_id')[['x_coord', 'y_coord']].to_dict('index')


    # Vehicles
    vehicles = [
        Vehicle(
            random.choice(nodes_2d),
            random.randint(0, T_max // 2),
            [],
            vid + 1
        )
        for vid in range(num_vehicles)
    ]

    # Requests
    requests = []
    for rid in range(1, num_requests + 1):
        origin, destination = random.sample(nodes_2d, 2)

        # Get coordinates for the origin
        origin_coords = node_coords_map.get(origin)
        if origin_coords is None:
            # This case implies an origin node was chosen that isn't in node_df, which shouldn't happen
            # if nodes_2d is derived from node_df. If it does, skipping this request.
            continue

        # Get coordinates for the destination
        destination_coords = node_coords_map.get(destination)
        if destination_coords is None:
            # Skip request if destination coordinates are not found
            continue

        release_time = random.randint(0, T_max - max_slack)

        travel_time = shortest_path_cached(
            node2d_to_3d[origin],
            node2d_to_3d[destination]
        )

        if not np.isfinite(travel_time):
            continue

        slack = random.randint(min_slack, max_slack)

        requests.append(
            Request(
                origin,
                destination,
                release_time,
                release_time + slack,   # latest pickup
                travel_time,            # earliest arrival
                rid,
                origin_coords['x_coord'], # Pass x_coord
                origin_coords['y_coord'],  # Pass y_coord
                destination_coords['x_coord'], # Pass dest_x
                destination_coords['y_coord']  # Pass dest_y
            )
        )

    return requests, vehicles

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

import time

def classical_ilp_run(
    metadata,
    baseline_vehicle_costs,
    csv_filename="results.csv",
    stats={},
    seed=123,
    trial=1
):
    """
    Classical ILP run with clean timing semantics:

    - total_run_time: end-to-end time inside THIS function (solver pipeline only)
    - solve_time: time spent inside optimal_assignment() (ILP solve/model time)
    - rtv_graph_build_time: scenario precompute time from final_trips (t1 - t0), if available
    - scenario_total_time: scenario pipeline time (if provided via stats), optional
    """

    # Per-solver wall time (does NOT include scenario generation unless you call this inside it)
    run_start = time.perf_counter()

    # ----------------------------
    # ILP SOLVE
    # ----------------------------
    t2 = time.perf_counter()

    Sigma_opt, ignored, ilp_stats = optimal_assignment(
        requests=[r.id for r in requests],
        vehicles=[v.id for v in vehicles],
        trips=trips,
        trip_costs=trip_costs,
        trip_to_vehicle=trip_to_vehicle,
        baseline_vehicle_costs=baseline_vehicle_costs,
        ignore_cost=10000
    )

    t3 = time.perf_counter()

    # ----------------------------
    # POST-PROCESSING
    # ----------------------------
    served_requests = set()
    served_trips = []

    for trip_ids, vid in Sigma_opt:
        trip_objs = frozenset(request_lookup[r] for r in trip_ids)
        v_obj = vehicle_lookup[vid]
        served_trips.append((trip_objs, v_obj))
        for r in trip_objs:
            served_requests.add(r.id)

    percent_serviced = 100.0 * len(served_requests) / len(requests) if requests else 0.0

    # ----------------------------
    # METRICS (RAW)
    # ----------------------------
    detours = []
    waiting = []
    VMT = 0.0

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

    run_end = time.perf_counter()

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print("Number of nodes in the graph:", node_df["node_id"].nunique())
    print("Number of vehicles:", len(vehicles))
    print("Number of requests:", len(requests))
    print(f"% Serviced Requests: {percent_serviced:.2f}%")

    print("Optimal Assignment:")
    for trip, v in served_trips:
        print(" ", trip_name(trip), "→", v)

    print("\nIgnored Requests:", ignored)

    print("\n--- Performance Metrics ---")
    print(f"Average Waiting Time: {avg_wait:.2f} s")
    print(f"Maximum Waiting Time: {max_wait:.2f} s")
    print(f"Average Detour Factor: {avg_detour:.3f}")
    print(f"Maximum Detour Factor: {max_detour:.3f}")
    print(f"Vehicle Miles Traveled (VMT): {VMT:.2f}")

    # IMPORTANT: do not report "t3 - t0" as solver runtime; that's scenario-cumulative.
    print(f"Total Run Time (solver only): {run_end - run_start:.9f} s")
    print(f"ILP Solve Time:              {t3 - t2:.9f} s")

    if "t0" in globals() and "t1" in globals():
        print(f"RTV Graph Build Time:        {t1 - t0:.2f} s")

    # ----------------------------
    # SAVE METRICS
    # ----------------------------
    metrics_to_save = {
        "city": metadata.get("city", ""),
        "run_type": "Classical",
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
        "total_run_time": run_end - run_start,   # solver pipeline wall time
        "solve_time": t3 - t2,                   # ILP solver/model call time

        # Scenario build time (if available from final_trips):
        "rtv_graph_build_time": (t1 - t0) if ("t0" in globals() and "t1" in globals()) else None,

        "seed": seed,
        "trial": trial,
    }

    # Add scenario-level time if you computed it in the outer loop and stuffed into stats
    # (recommended): stats["scenario_total_time"] = scenario_end - scenario_start

    metrics_to_save.update(stats)
    metrics_to_save.update(ilp_stats)

    save_metrics_to_csv(csv_filename, metrics_to_save)
    return Sigma_opt, ignored

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

def quantum_mwis_run(
    metadata,
    baseline_vehicle_costs,   # kept for signature parity; not used here
    csv_filename="results.csv",
    real=False,
    request_stats={},
    seed=123,
    trial=1
):
    """
    Quantum MWIS run with clean timing semantics.

    - total_run_time: end-to-end wall time inside THIS function (solver pipeline only)
    - solve_time: time spent in solve_qubo_qiskit_real (sampler/QAOA/SA)
    - rtv_graph_build_time: scenario build time from final_trips (t1 - t0), if available
    - qubo_build_time: QUBO build time (min-cost prep + QUBO generation)
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
    t_qubo_gen_start = time.perf_counter()
    Q_matrix, all_vars = generate_qubo(
        trips,
        trip_costs=minimal_qubo_trip_costs,
        return_numpy=False,
        seed=seed
    )
    t_qubo_gen_end = time.perf_counter()

    # ----------------------------
    # 3) Solve QUBO
    # ----------------------------
    t_quantum_solve_start = time.perf_counter()
    quantum_bitstring = solve_qubo_qiskit_real(Q_matrix, reps=2, real=real)
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

        "energy_gap": float(energygap),
        "condition_number": float(conditionnumber),
        "real_quantum_hardware": 1 if real else 0,
        "seed": seed,
        "trial": trial,
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

import pandas as pd

import os

import numpy as np

CSV_COLUMNS = [

    # =============================
    # Core metadata
    # =============================
    "city",
    "run_type",
    "nodes",
    "num_vehicles",
    "num_requests",
    "seed",
    "trial",

    # =============================
    # Scenario-level timing (measured ONCE around final_trips in the outer loop)
    # =============================
    #"scenario_total_time",          # NEW: includes request generation + RTV build + trip enumeration
    "rtv_graph_build_time",         # scenario component (from final_trips)

    # =============================
    # Performance outcomes
    # =============================
    "percent_serviced",
    "avg_waiting_time",
    "max_waiting_time",
    "avg_detour_factor",
    "max_detour_factor",
    "vmt",

    # =============================
    # Solver-level timing (measured INSIDE each solver function)
    # =============================
    "total_run_time",               # solver pipeline wall time (per function call)
    "solve_time",                   # optimizer call only (ILP solve or quantum solve)
    "qubo_build_time",              # all pre-solve QUBO work (prep + gen [+ compress])

    # =============================
    # Quantum pipeline breakdown
    # (These will be blank/None for Classical + Greedy)
    # =============================
    "time_min_cost_prep",
    "time_qubo_gen",
    "time_compress",
    "time_quantum_solve",
    "time_decode",
    "time_vehicle_assignment",
    "time_metrics_calc",
    "time_struct_stats",
    "time_total_quantum_block",     # should match total_run_time for Quantum/QC

    # =============================
    # ILP structural size (Classical only; else 0/None)
    # =============================
    "ilp_num_vars",
    "ilp_num_constraints",
    "ilp_num_integer_vars",
    "ilp_num_nonzero_coeffs",

    # =============================
    # Quantum diagnostics (Quantum/QC only; else None)
    # =============================
    "energy_gap",
    "argmin_preservation",
    "condition_number",
    "real_quantum_hardware",

    # =============================
    # Compression parameter (QC only; else None)
    # =============================
    "K",

    # =============================
    # Slack diagnostics (scenario-level; copied into each row via stats)
    # =============================
    "infeasible_windows",
    "arrival_violations",
    "mean_slack",
    "min_slack",
    "max_slack",
    "p25_slack",
    "median_slack",
    "p75_slack",

    # =============================
    # Baseline QUBO stats (Quantum + QC; else None)
    # =============================
    "base_qubo_vars",
    "base_qubo_couplers",
    "base_qubo_graph_density",
    "base_node_density_avg",
    "base_node_density_max",
    "base_node_density_min",
    "base_degree_avg",
    "base_degree_max",
    "base_degree_min",
    "base_qubo_max_abs",
    "base_qubo_min_nonzero_abs",
    "base_qubo_dynamic_range",

    # =============================
    # Compressed QUBO stats (QC only; else None)
    # =============================
    "comp_qubo_vars",
    "comp_qubo_couplers",
    "comp_qubo_graph_density",
    "comp_node_density_avg",
    "comp_node_density_max",
    "comp_node_density_min",
    "comp_degree_avg",
    "comp_degree_max",
    "comp_degree_min",
    "comp_qubo_max_abs",
    "comp_qubo_min_nonzero_abs",
    "comp_qubo_dynamic_range",

    # =============================
    # Qubit counts (Quantum + QC; else None)
    # =============================
    "qubo_logical_qubits",

    # =============================
    # Greedy instrumentation (Greedy only; else None/0)
    # =============================
    "time_greedy_build_candidates",
    "greedy_edges_total",
    "time_greedy_sort",
    "time_greedy_select",
    "time_greedy_total",
    "greedy_sort_work",
    "greedy_assignments",
]

def save_metrics_to_csv(filename, metrics_dict):
    """
    Appends metrics to CSV using a fixed schema.
    Missing columns are filled with 0.
    """

    # Convert single row to DataFrame
    df_new = pd.DataFrame([metrics_dict])

    # Enforce fixed schema
    df_new = df_new.reindex(columns=CSV_COLUMNS, fill_value=np.nan)

    # Append safely
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)

        # Ensure old file also matches schema
        df_existing = df_existing.reindex(columns=CSV_COLUMNS, fill_value=np.nan)

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(filename, index=False)

    print(f"Data successfully saved to {filename}")

def final_trips(num_requests: int, num_vehicles: int, seed=123):
    global requests, vehicles, vehicle_lookup, request_lookup, trips, trip_to_vehicle, trip_costs, t0, t1

    travel_cached.cache_clear()

    requests, vehicles = generate_requests_and_vehicles(
        num_requests=int(num_requests),
        num_vehicles=int(num_vehicles),
        node2d_to_3d=node2d_to_3d,
        shortest_path_cached=shortest_path_cached,
        node_df=node_df,
        T_max=T_max,
        seed=seed
    )

    vehicle_lookup = {v.id: v for v in vehicles}
    request_lookup = {r.id: r for r in requests}

    t0 = time.perf_counter()
    # Iterate through each Request object to add 3D origin and destination attributes
    for req_obj in requests:
        req_obj.origin_3d = node2d_to_3d[req_obj.origin]
        req_obj.dest_3d = node2d_to_3d[req_obj.destination]
    all_trips_with_costs = build_trips(requests=requests, vehicles=vehicles, nu=4)
    t1 = time.perf_counter()

    trips = {}
    trip_to_vehicle = {}
    trip_costs = {}

    # NEW: Initialize baseline dictionary
    # Format: {(request_id, vehicle_id): cost}


    for t_key, v_costs_dict in all_trips_with_costs.items():
        trips[t_key] = t_key
        trip_to_vehicle[t_key] = list(v_costs_dict.keys())
        for v_id, cost in v_costs_dict.items():
            trip_costs[(t_key, v_id)] = cost

            # LOGIC: If the trip contains only 1 request, it is a baseline cost
            # Removed this section as baseline_vehicle_costs should be per vehicle, not per (request, vehicle) pair
            # if len(t_key) == 1:
            #     req_id = list(t_key)[0]
            #     baseline_vehicle_costs[(req_id, v_id)] = cost

    # Correctly compute baseline_vehicle_costs using the dedicated function
    baseline_vehicle_costs = compute_baseline_vehicle_costs(vehicles)

    # Return the new baseline dictionary along with the other data
    return (
        requests, vehicles, vehicle_lookup, request_lookup,
        trips, trip_to_vehicle, trip_costs,
        baseline_vehicle_costs, t0, t1
    )

import numpy as np

def summarize_requests(requests):
    """
    Aggregate feasibility + cost-variance-sensitive diagnostics.
    Safe for dict/list inputs and numerically stable.
    """

    trr = np.array([r.trr for r in requests])
    tplr = np.array([r.tplr for r in requests])
    t_star = np.array([r.t_star for r in requests])

    slack = tplr - trr

    summary = {
        # ---- Feasibility ----
        "num_requests": len(requests),
        "infeasible_windows": int(np.sum(slack < 0)),
        "arrival_violations": int(np.sum(t_star < trr)),

        # ---- Time slack distribution ----
        "mean_slack": float(np.mean(slack)),
        "min_slack": float(np.min(slack)),
        "p25_slack": float(np.percentile(slack, 25)),
        "median_slack": float(np.median(slack)),
        "p75_slack": float(np.percentile(slack, 75)),
        "max_slack": float(np.max(slack))
    }

    return summary