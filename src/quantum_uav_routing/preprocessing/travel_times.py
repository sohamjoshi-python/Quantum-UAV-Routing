
from __future__ import annotations

from functools import lru_cache
import networkx as nx

G = None

def configure_runtime(**kwargs):
    globals().update(kwargs)

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
