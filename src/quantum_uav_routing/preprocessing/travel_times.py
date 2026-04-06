"""
travel_times.py
===============
Shortest-path oracles for the 3-D UAM airspace graph.

This module provides two complementary interfaces:

Fast batch oracle (recommended for large experiments)
    :func:`build_csr` compiles the NetworkX graph into a SciPy CSR sparse
    matrix.  :func:`make_shortest_path_fn` wraps it in a row-cached Dijkstra
    call that computes all distances from a source node in a single C-level
    sweep and caches the resulting distance array.  Amortised O(1) per query
    once a source has been seen.

NetworkX fallback oracle (small graphs / debugging)
    :func:`make_nx_shortest_path_fn` wraps ``nx.bidirectional_dijkstra`` with
    an LRU cache.  Slower but requires no CSR pre-compilation.

Usage example
-------------
.. code-block:: python

    from src.quantum_uav_routing.preprocessing.graph_3d import build_3d_graph
    from src.quantum_uav_routing.preprocessing.travel_times import (
        build_csr, make_shortest_path_fn
    )

    G = build_3d_graph(node_3d_df, link_3d_df)
    csr, node_list, node_to_idx = build_csr(G)
    shortest_path_cached = make_shortest_path_fn(csr, node_list, node_to_idx)

    cost = shortest_path_cached("1234_z0", "5678_z100")

Design notes
------------
* Node IDs are always strings (e.g. ``"3300_z400"``); ``int`` or
  ``numpy`` scalar IDs are coerced to ``str`` before lookup to prevent
  cache misses.
* ``_dist_from_src`` is intentionally module-private so callers cannot
  accidentally hold a stale reference after the CSR is rebuilt.
* The CSR is built as an *undirected* symmetric matrix (both (u,v) and (v,u)
  are inserted) to allow ``scipy.sparse.csgraph.dijkstra`` with
  ``directed=False``.  If you need strictly directed shortest paths, set
  ``directed=True`` in the Dijkstra call and remove the symmetry step in
  :func:`build_csr`.

Dependencies
------------
    networkx, numpy, scipy
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

__all__ = [
    "build_csr",
    "make_shortest_path_fn",
    "make_nx_shortest_path_fn",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 – Build CSR sparse matrix
# ---------------------------------------------------------------------------


def build_csr(
    G: nx.DiGraph,
) -> Tuple[csr_matrix, list, dict]:
    """Compile a NetworkX graph into a SciPy CSR sparse matrix.

    The matrix is built as an *undirected* symmetric graph so that
    ``scipy.sparse.csgraph.dijkstra`` can be called efficiently without the
    overhead of maintaining a directed predecessor table.

    Parameters
    ----------
    G:
        Weighted NetworkX DiGraph (output of
        ``graph_3d.build_3d_graph``).  Edge attribute ``"weight"`` is used.

    Returns
    -------
    csr : csr_matrix
        Sparse adjacency / weight matrix of shape ``(N, N)``, dtype ``float64``.
    node_list : list[str]
        Ordered list of node IDs; ``node_list[i]`` ↔ row / column ``i``.
    node_to_idx : dict[str, int]
        Inverse mapping ``node_id → matrix index``.

    Notes
    -----
    Building the CSR is an O(E) operation and should be done **once per
    city** in the outer experiment loop, not per scenario.
    """
    node_list = list(G.nodes())
    node_to_idx: dict = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for u, v, attr in G.edges(data=True):
        w = float(attr.get("weight", 1.0))
        iu = node_to_idx[u]
        iv = node_to_idx[v]
        # Both directions → undirected Dijkstra
        rows.append(iu); cols.append(iv); data.append(w)
        rows.append(iv); cols.append(iu); data.append(w)

    csr = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    logger.info(
        "Built CSR matrix: %d nodes, %d non-zeros (undirected).",
        n, csr.nnz,
    )
    return csr, node_list, node_to_idx


# ---------------------------------------------------------------------------
# Step 2 – Fast row-cached Dijkstra oracle
# ---------------------------------------------------------------------------


def make_shortest_path_fn(
    csr: csr_matrix,
    node_list: list,
    node_to_idx: dict,
    cache_size: int = 5000,
) -> Callable[[str | int, str | int], float]:
    """Return a shortest-path callable backed by a row-cached C Dijkstra.

    The returned function computes all distances from a source node on the
    first call and caches the full distance array.  Subsequent queries from
    the same source are O(1) dictionary lookups.

    Parameters
    ----------
    csr:
        Sparse adjacency matrix from :func:`build_csr`.
    node_list:
        Ordered node IDs from :func:`build_csr`.
    node_to_idx:
        Inverse mapping from :func:`build_csr`.
    cache_size:
        LRU cache capacity for the per-source distance arrays.  Each entry
        holds a ``float64`` array of length N.  At 3 000 nodes per city and
        float64, one entry ≈ 24 KB; 5 000 entries ≈ 120 MB.

    Returns
    -------
    Callable[[str, str], float]
        ``shortest_path_cached(src, dst) -> float``
        Returns ``float('inf')`` when no path exists.

    Examples
    --------
    .. code-block:: python

        sp = make_shortest_path_fn(csr, node_list, node_to_idx)
        cost = sp("1234_z0", "5678_z100")
    """

    @lru_cache(maxsize=cache_size)
    def _dist_from_src(src_node: str) -> np.ndarray:
        """Compute and cache the distance vector from *src_node* (all targets)."""
        src_idx = node_to_idx[src_node]
        dist = dijkstra(
            csr,
            directed=False,
            indices=src_idx,
            return_predecessors=False,
        )
        return dist  # shape (N,)

    def shortest_path_cached(
        src: str | int,
        dst: str | int,
    ) -> float:
        """Return the shortest-path cost from *src* to *dst*.

        Parameters
        ----------
        src, dst:
            3-D node IDs (strings like ``"3300_z400"``).  NumPy scalar types
            are coerced to ``str`` automatically.

        Returns
        -------
        float
            Travel time in seconds, or ``float('inf')`` if unreachable.
        """
        # Normalise: numpy scalars → Python str (avoid cache misses)
        if isinstance(src, np.generic):
            src = src.item()
        if isinstance(dst, np.generic):
            dst = dst.item()
        src = str(src)
        dst = str(dst)

        dist = _dist_from_src(src)
        return float(dist[node_to_idx[dst]])

    # Expose cache management on the returned callable
    shortest_path_cached.cache_clear = _dist_from_src.cache_clear  # type: ignore[attr-defined]
    shortest_path_cached.cache_info = _dist_from_src.cache_info    # type: ignore[attr-defined]

    return shortest_path_cached


# ---------------------------------------------------------------------------
# Fallback – NetworkX bidirectional Dijkstra oracle
# ---------------------------------------------------------------------------


def make_nx_shortest_path_fn(
    G: nx.DiGraph,
    cache_size: int = 500_000,
) -> Callable[[str | int, str | int], float]:
    """Return a shortest-path callable backed by NetworkX bidirectional Dijkstra.

    Useful for small graphs or debugging.  For production experiments with
    thousands of nodes, prefer :func:`make_shortest_path_fn`.

    Parameters
    ----------
    G:
        Weighted NetworkX DiGraph.  Edge attribute ``"weight"`` is used.
    cache_size:
        LRU cache capacity for individual (src, dst) pairs.

    Returns
    -------
    Callable[[str, str], float]
        ``shortest_path_nx(src, dst) -> float``
        Returns ``float('inf')`` when no path exists.
    """

    @lru_cache(maxsize=cache_size)
    def shortest_path_nx(
        src: str | int,
        dst: str | int,
    ) -> float:
        """Return shortest-path cost using NetworkX bidirectional Dijkstra."""
        if isinstance(src, np.generic):
            src = src.item()
        if isinstance(dst, np.generic):
            dst = dst.item()
        src = str(src)
        dst = str(dst)
        try:
            cost, _ = nx.bidirectional_dijkstra(G, source=src, target=dst, weight="weight")
            return float(cost)
        except nx.NetworkXNoPath:
            return float("inf")
        except nx.NodeNotFound:
            logger.warning("Node not found in graph: src=%s, dst=%s", src, dst)
            return float("inf")

    return shortest_path_nx
