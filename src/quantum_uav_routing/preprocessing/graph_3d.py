"""
graph_3d.py
===========
Expand a 2-D GMNS-Plus node/link dataset into a multi-altitude 3-D airspace
graph and assemble a weighted NetworkX DiGraph suitable for shortest-path
queries.

Pipeline
--------
1. :func:`convert_to_3d`  – replicate every 2-D node at each altitude layer
   and generate horizontal + vertical links.
2. :func:`build_3d_graph` – compile a weighted ``nx.DiGraph`` from the 3-D
   DataFrames, embedding travel-time and energy costs on edges.

Physics model
-------------
The UAV physics helpers below implement the altitude-dependent speed and
energy model from the original notebook.  They are module-level functions so
they can be imported and overridden independently in tests or parameter sweeps.

* :func:`horizontal_speed`           – cruise speed (m/s) at altitude z.
* :func:`horizontal_energy_per_meter` – energy per metre at altitude z.
* :func:`vertical_energy`            – energy for a climb/descent of dz metres.

Altitude layers
---------------
The default layers ``[0, 50, 100, 200, 400]`` m correspond to ground, low,
mid, high, and cruise tiers.  Adjust via the *altitudes* parameter.

Dependencies
------------
    pandas, numpy, networkx
"""

from __future__ import annotations

import logging
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    # physics
    "horizontal_speed",
    "horizontal_energy_per_meter",
    "vertical_energy",
    # pipeline
    "convert_to_3d",
    "build_3d_graph",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default altitude layers (metres).  Each 2-D node is replicated once per
#: layer; adjacent layers are connected by vertical (climb/descent) links.
DEFAULT_ALTITUDES: list[int] = [0, 50, 100, 200, 400]

#: Energy cost (arbitrary units) per metre of vertical displacement.
_CLIMB_COST_PER_METER: float = 0.02

#: Free-speed (m/s) for vertical (climb / descent) links.
_VERTICAL_SPEED_MS: float = 4.0


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def horizontal_speed(z: float) -> float:
    """Horizontal cruise speed (m/s) as a function of altitude *z* (m).

    Modelled as a concave-increasing function of altitude reflecting eVTOL
    efficiency gains at higher cruise layers::

        speed(z) = 10 + 0.15 * sqrt(z)

    Parameters
    ----------
    z:
        Altitude in metres.

    Returns
    -------
    float
        Cruise speed in m/s.

    Examples
    --------
    >>> round(horizontal_speed(0), 2)
    10.0
    >>> round(horizontal_speed(400), 2)
    13.0
    """
    return 10.0 + 0.15 * float(np.sqrt(z))


def horizontal_energy_per_meter(z: float) -> float:
    """Energy cost per metre of horizontal flight at altitude *z* (m).

    Higher altitudes yield lower per-metre energy, floored at 0.3 units/m::

        e(z) = max(0.3, 1.0 - 0.001 * z)

    Parameters
    ----------
    z:
        Altitude in metres.

    Returns
    -------
    float
        Energy per metre in arbitrary units; calibrate to your cost model.

    Examples
    --------
    >>> horizontal_energy_per_meter(0)
    1.0
    >>> horizontal_energy_per_meter(800)  # floored
    0.3
    """
    return max(0.3, 1.0 - 0.001 * z)


def vertical_energy(dz: float) -> float:
    """Energy required to climb (or descend) *dz* metres.

    Uses a square-root model that reflects diminishing marginal cost for
    larger altitude changes::

        E(dz) = 5 * sqrt(|dz|)

    The sign of *dz* is ignored; climb and descent use the same model here.
    A separate descent coefficient ``C_DESCENT`` is applied in
    :func:`build_3d_graph` for descent links.

    Parameters
    ----------
    dz:
        Vertical displacement in metres (sign ignored).

    Returns
    -------
    float
        Energy cost in arbitrary units.

    Examples
    --------
    >>> round(vertical_energy(100), 4)
    50.0
    """
    return 5.0 * float(np.sqrt(abs(dz)))


# ---------------------------------------------------------------------------
# Step 1 – 3-D expansion
# ---------------------------------------------------------------------------


def convert_to_3d(
    node_df: pd.DataFrame,
    link_df: pd.DataFrame,
    altitudes: list[int] = DEFAULT_ALTITUDES,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Expand a 2-D node/link dataset into a multi-altitude 3-D airspace graph.

    Each 2-D node is replicated once at every altitude layer (horizontal
    copies).  Horizontal links at each layer receive altitude-dependent free
    speeds.  Adjacent layers at the same geographic position are connected by
    vertical (climb/descent) links.

    Node ID convention
    ------------------
    A 2-D node ``1234`` at altitude 100 m becomes ``"1234_z100"``.

    Parameters
    ----------
    node_df:
        2-D node DataFrame; required columns:
        ``node_id``, ``x_coord``, ``y_coord``, ``zone_id``, ``geometry``.
    link_df:
        2-D link DataFrame; required columns:
        ``link_id``, ``from_node_id``, ``to_node_id``,
        ``dir_flag``, ``length``, ``geometry``.
    altitudes:
        Ordered list of altitude layers in metres (must be sorted ascending).
        Defaults to ``[0, 50, 100, 200, 400]``.

    Returns
    -------
    node_3d_df : pd.DataFrame
        3-D node table.  Columns:
        ``node_id``, ``original_node_id``, ``zone_id``,
        ``x_coord``, ``y_coord``, ``z_coord``, ``geometry``.
    link_3d_df : pd.DataFrame
        3-D link table.  Columns:
        ``link_id``, ``from_node_id``, ``to_node_id``,
        ``dir_flag``, ``length``, ``free_speed``,
        ``link_type`` (``"horizontal"`` or ``"vertical"``),
        ``altitude``, ``geometry``.
        Vertical links additionally carry an ``energy_cost`` column.
    node2d_to_3d : dict
        Maps each 2-D ``node_id`` to its ground-level (``altitudes[0]``)
        3-D ``node_id``.  Used by the RTV pipeline to convert 2-D origins /
        destinations to routable 3-D nodes.

    Notes
    -----
    * Horizontal link free speeds follow ``12 + 0.25 * z`` (m/s), slightly
      different from :func:`horizontal_speed` which is used in the weighted
      graph.  The DataFrames carry the free-speed annotation for reference;
      the actual travel-time weights are computed in :func:`build_3d_graph`.
    * Vertical links use a fixed ``dir_flag = 1`` (bidirectional) so that
      descent is always possible.
    """
    ground_z = altitudes[0]

    # ------------------------------------------------------------------
    # Horizontal node copies
    # ------------------------------------------------------------------
    node_3d_rows: list[dict] = []
    for _, row in node_df.iterrows():
        for z in altitudes:
            node_3d_rows.append(
                {
                    "node_id": f"{row.node_id}_z{z}",
                    "original_node_id": row.node_id,
                    "zone_id": row.zone_id,
                    "x_coord": row.x_coord,
                    "y_coord": row.y_coord,
                    "z_coord": z,
                    "geometry": row.geometry,
                }
            )

    # ------------------------------------------------------------------
    # Horizontal link copies (one per altitude layer)
    # ------------------------------------------------------------------
    link_3d_rows: list[dict] = []
    for _, row in link_df.iterrows():
        for z in altitudes:
            link_3d_rows.append(
                {
                    "link_id": f"{row.link_id}_z{z}",
                    "from_node_id": f"{row.from_node_id}_z{z}",
                    "to_node_id": f"{row.to_node_id}_z{z}",
                    "dir_flag": row.dir_flag,
                    "length": row.length,
                    # Reference free-speed annotation (m/s); not used for weighting
                    "free_speed": 12.0 + 0.25 * z,
                    "link_type": "horizontal",
                    "altitude": z,
                    "geometry": row.geometry,
                }
            )

    # ------------------------------------------------------------------
    # Vertical links (climb / descent) between adjacent altitude layers
    # ------------------------------------------------------------------
    for _, row in node_df.iterrows():
        for z_lo, z_hi in zip(altitudes[:-1], altitudes[1:]):
            dz = z_hi - z_lo
            link_3d_rows.append(
                {
                    "link_id": f"V_{row.node_id}_{z_lo}_{z_hi}",
                    "from_node_id": f"{row.node_id}_z{z_lo}",
                    "to_node_id": f"{row.node_id}_z{z_hi}",
                    "dir_flag": 1,             # bidirectional → descent added in build_3d_graph
                    "length": dz,
                    "free_speed": _VERTICAL_SPEED_MS,
                    "link_type": "vertical",
                    "altitude": f"{z_lo}->{z_hi}",
                    "energy_cost": dz * _CLIMB_COST_PER_METER,
                    "geometry": None,
                }
            )

    node_3d_df = pd.DataFrame(node_3d_rows)
    link_3d_df = pd.DataFrame(link_3d_rows)

    # Map each 2-D node ID to its ground-level 3-D node ID
    node2d_to_3d: dict = (
        node_3d_df[node_3d_df["z_coord"] == ground_z]
        .set_index("original_node_id")["node_id"]
        .to_dict()
    )

    logger.info(
        "3-D expansion: %d 2-D nodes → %d 3-D nodes; %d 3-D links (%d horizontal, %d vertical).",
        len(node_df),
        len(node_3d_df),
        len(link_3d_df),
        len(link_df) * len(altitudes),
        len(node_df) * (len(altitudes) - 1),
    )
    return node_3d_df, link_3d_df, node2d_to_3d


# ---------------------------------------------------------------------------
# Step 2 – Weighted NetworkX graph
# ---------------------------------------------------------------------------


def build_3d_graph(
    node_3d_df: pd.DataFrame,
    link_3d_df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.05,
    C_DESCENT: float = 1.0,
) -> nx.DiGraph:
    """Build a weighted NetworkX DiGraph from 3-D node and link DataFrames.

    Edge weight formula
    -------------------
    Current implementation sets ``weight = travel_time`` (i.e. ``alpha`` and
    ``beta`` are exposed for future use but only travel-time is active)::

        weight = alpha * travel_time   # extend: + beta * energy

    To enable a full bi-criteria weight, replace the ``weight`` assignment
    inside the function body.

    Parameters
    ----------
    node_3d_df:
        Output of :func:`convert_to_3d`.
    link_3d_df:
        Output of :func:`convert_to_3d`.
    alpha:
        Coefficient on travel time in the composite cost (default 1.0).
    beta:
        Coefficient on energy in the composite cost (default 0.05).
        Currently inactive – set to 0 to disable entirely.
    C_DESCENT:
        Energy multiplier applied to *descent* vertical links (J/m equivalent).
        Climb links use :func:`vertical_energy` directly.

    Returns
    -------
    nx.DiGraph
        Directed graph.  Node attributes: ``x``, ``y``, ``z``.
        Edge attributes: ``weight`` (s), ``time`` (s), ``energy`` (a.u.),
        ``link_type`` (``"horizontal"`` or ``"vertical"``).

    Notes
    -----
    * GMNS ``dir_flag = 1`` means **two-way**: both the forward and reverse
      edges are inserted.  Adjust the condition if your dataset uses a
      different convention.
    * Horizontal travel times are computed from :func:`horizontal_speed` and
      the link ``length`` in metres.
    * Vertical travel times use a fixed ascent/descent rate of 100 m/s
      (i.e. 1 s per 100 m, matching the notebook assumption).  Override by
      subclassing or monkey-patching ``_vertical_travel_time``.
    """
    G: nx.DiGraph = nx.DiGraph()

    # --- Nodes ---
    for _, row in node_3d_df.iterrows():
        G.add_node(
            row["node_id"],
            x=row["x_coord"],
            y=row["y_coord"],
            z=row["z_coord"],
        )

    # --- Edges ---
    for _, row in link_3d_df.iterrows():
        length = _resolve_length(row)

        if row["link_type"] == "horizontal":
            z = float(row["altitude"])
            travel_time = length / horizontal_speed(z)
            energy = horizontal_energy_per_meter(z) * length

        else:  # vertical
            z_from, z_to = _parse_altitude_range(str(row["altitude"]))
            dz = z_to - z_from
            travel_time = abs(dz) / 100.0  # 100 m/s ascent/descent rate
            energy = vertical_energy(dz) if dz > 0 else C_DESCENT * abs(dz)

        # Active weight = travel_time only.
        # To include energy: weight = alpha * travel_time + beta * energy
        weight = alpha * travel_time

        edge_attrs = dict(
            weight=weight,
            time=travel_time,
            energy=energy,
            link_type=row["link_type"],
        )

        G.add_edge(row["from_node_id"], row["to_node_id"], **edge_attrs)

        # dir_flag == 1 → bidirectional link (add reverse direction)
        if row["dir_flag"] == 1:
            G.add_edge(row["to_node_id"], row["from_node_id"], **edge_attrs)

    logger.info(
        "Built 3-D graph: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_length(row: "pd.Series") -> float:
    """Return a positive finite link length, falling back to geometry or a sentinel."""
    length = row["length"]
    try:
        l_float = float(length)
        if l_float > 0 and np.isfinite(l_float):
            return l_float
    except (TypeError, ValueError):
        pass
    geom = row.get("geometry")
    if geom is not None and hasattr(geom, "length") and geom.length > 0:
        return float(geom.length)
    return 1e-3  # sentinel: 1 mm to avoid division-by-zero


def _parse_altitude_range(altitude_str: str) -> tuple[float, float]:
    """Parse a vertical link altitude string like ``'50->100'`` → ``(50.0, 100.0)``."""
    parts = altitude_str.split("->")
    if len(parts) != 2:
        raise ValueError(
            f"Vertical link altitude must be formatted as 'z_from->z_to'; got: {altitude_str!r}"
        )
    return float(parts[0]), float(parts[1])
