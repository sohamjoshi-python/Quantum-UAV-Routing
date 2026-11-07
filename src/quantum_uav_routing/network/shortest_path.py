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

        edge_attrs = dict(
            weight=weight,
            time=travel_time,
            energy=energy,
            length=float(length),
            link_type=row["link_type"],
            altitude=row["altitude"],
        )

        G.add_edge(
            row["from_node_id"],
            row["to_node_id"],
            **edge_attrs,
        )

        if row["dir_flag"] == 1:
            G.add_edge(
                row["to_node_id"],
                row["from_node_id"],
                **edge_attrs,
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


def path_length_by_altitude(G, source, target, altitudes=(0, 50, 100, 200, 400)):
    """Share of shortest-path length at each cruise altitude.

    Horizontal length is charged to that link's altitude. Climb and descent
    length is reported as ``vertical_length``. The path weight is travel time,
    so a ground-to-ground trip can still cruise higher when the faster layer
    pays for the climb.
    """
    path = nx.shortest_path(G, source, target, weight="weight")
    by_alt = {int(z): 0.0 for z in altitudes}
    vertical = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.edges[u, v]
        length = float(data.get("length", 0.0))
        if data.get("link_type") == "vertical":
            vertical += length
        else:
            z = int(data.get("altitude", G.nodes[u].get("z")))
            by_alt[z] = by_alt.get(z, 0.0) + length
    total = vertical + sum(by_alt.values())
    share = {
        z: (length / total if total else 0.0)
        for z, length in by_alt.items()
    }
    return {
        "path": path,
        "length_by_altitude": by_alt,
        "vertical_length": vertical,
        "total_length": total,
        "share_by_altitude": share,
        "vertical_share": (vertical / total if total else 0.0),
    }