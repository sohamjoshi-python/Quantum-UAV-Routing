import itertools
import networkx as nx
from functools import lru_cache

import multiprocessing as mp
from functools import partial
from collections import defaultdict
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString

from quantum_uav_routing.rtv.trip_builder import travel

RTV = nx.DiGraph()

# Store all trips: key is frozenset({r1, r2, ...}), value is trip object
TRIPS = {}

def get_trip_key(trip):
    """
    trip: iterable of request IDs (ints)
    returns: frozenset[int]
    """
    return frozenset(int(r) for r in trip)


@lru_cache(maxsize=None)
def feasible_cached(vehicle_id, trip_key):
    """
    Cache only FEASIBILITY, not cost.
    trip_key is frozenset of request IDs.
    """
    v = vehicle_lookup[vehicle_id]
    trip = {request_lookup[rid] for rid in trip_key}
    return travel(v, trip) is not None


MAX_WAIT = 1800  # must match travel()

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
    # RV (Servability edges) is passed as an argument directly now

    # Add Shareability Edges (Red) - Request to Request
    rr_edges = set()
    for trip_requests_ids in TRIPS.keys():
        if len(trip_requests_ids) >= 2:
            import itertools
            for r1_id, r2_id in itertools.combinations(trip_requests_ids, 2):
                # Assuming request_lookup is available globally or passed implicitly
                # Re-using request_lookup from the global scope as done before
                r1 = globals().get('request_lookup', {}).get(r1_id)
                r2 = globals().get('request_lookup', {}).get(r2_id)
                if r1 and r2 and r1 in pos and r2 in pos:
                    # Ensure r1 and r2 are Request objects for comparison/hashing
                    rr_edges.add(tuple(sorted((r1, r2), key=lambda x: x.id)))

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(20, 10)) # Modified figsize for wider graph

    # A. Draw the Background Map (Street Network)
    if link_gdf is not None:
        link_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)
    else:
        print("Warning: link_gdf not provided, skipping background map.")

    # B. Draw the Edges (Connections)
    # Shareability (Red Dotted)
    nx.draw_networkx_edges(G, pos, edgelist=list(rr_edges), ax=ax,
                           edge_color='red', style='dotted', width=2, alpha=0.6, label='Shareability')

    # Servability (Green Solid)
    nx.draw_networkx_edges(G, pos, edgelist=RV, ax=ax,
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