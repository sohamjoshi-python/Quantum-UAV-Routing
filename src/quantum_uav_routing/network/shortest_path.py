import networkx as nx
import numpy as np
from functools import lru_cache
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


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



# Cache: compute distances from a source node ONCE (fast C Dijkstra)
@lru_cache(maxsize=5000)
def _dist_from_src(src_node):
    src_node = str(src_node)
    src_idx = NODE_TO_IDX[src_node]
    dist = dijkstra(CSR, directed=False, indices=src_idx, return_predecessors=False)
    return dist  # numpy array length N

def shortest_path_fast(src_node, dst_node):
    src_node = str(src_node)
    dst_node = str(dst_node)
    dist = _dist_from_src(src_node)
    d = dist[NODE_TO_IDX[dst_node]]
    return float(d)

def shortest_path_cached(src, dst):
  return shortest_path_fast(src, dst)