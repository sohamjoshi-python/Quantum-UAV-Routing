"""
load_network.py
===============
Utilities for loading, preprocessing, and building 2-D and 3-D transport
networks from GMNS-Plus CSV datasets.

Pipeline
--------
1. ``create_graph``   – generate H3 zone file from raw node data (run once).
2. ``parse_graph``    – load node / link / zone CSVs → GeoDataFrames + Folium map.
3. ``convert_to_3d``  – expand a 2-D network into a multi-altitude 3-D graph.
4. ``build_3d_graph`` – build a weighted NetworkX DiGraph from 3-D dataframes.
5. ``build_csr``      – compile a SciPy CSR matrix for fast Dijkstra queries.

Physics helpers
---------------
* ``horizontal_speed``          – cruise speed as a function of altitude (m/s).
* ``horizontal_energy_per_meter`` – energy cost per metre at a given altitude.
* ``vertical_energy``           – energy cost for a vertical climb of dz metres.

Dependencies
------------
    pandas, geopandas, shapely, h3, numpy, folium, networkx, scipy

Install
-------
    pip install pandas geopandas shapely h3 numpy folium networkx scipy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import folium
import h3.api.basic_int as h3
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon

__all__ = [
    # physics helpers
    "horizontal_speed",
    "horizontal_energy_per_meter",
    "vertical_energy",
    # pipeline steps
    "create_graph",
    "parse_graph",
    "convert_to_3d",
    "build_3d_graph",
    "build_csr",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Altitude layers used when expanding a 2-D network into 3-D (metres).
DEFAULT_ALTITUDES: list[int] = [0, 50, 100, 200, 400]

#: H3 hex resolution used when tessellating the study area.
DEFAULT_H3_RESOLUTION: int = 7

#: Buffer applied around the node convex hull before tessellating (km).
DEFAULT_BUFFER_KM: float = 1.0

#: Grid scan step when filling H3 hexagons (~300 m at mid-latitudes).
_GRID_STEP_DEG: float = 0.003

#: Energy cost per metre of vertical climb (placeholder; tune as needed).
_CLIMB_COST_PER_METER: float = 0.02

#: Free-speed for vertical (climb/descent) links (m/s).
_VERTICAL_SPEED_MS: float = 4.0


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def horizontal_speed(z: float) -> float:
    """Return horizontal cruise speed (m/s) at altitude *z* (metres).

    Speed increases with altitude, modelling eVTOL energy-efficiency
    improvements at higher cruise layers.

    Parameters
    ----------
    z:
        Altitude in metres.

    Returns
    -------
    float
        Cruise speed in m/s.
    """
    return 10.0 + 0.15 * np.sqrt(z)


def horizontal_energy_per_meter(z: float) -> float:
    """Return energy cost per metre of horizontal flight at altitude *z*.

    Higher altitudes yield lower per-metre energy due to thinner air and
    more efficient cruise conditions, floored at 0.3 units/m.

    Parameters
    ----------
    z:
        Altitude in metres.

    Returns
    -------
    float
        Energy per metre (arbitrary units; calibrate to your cost model).
    """
    base = 1.0
    reduction = 0.001 * z
    return max(0.3, base - reduction)


def vertical_energy(dz: float) -> float:
    """Return energy required to climb (or descend) *dz* metres.

    Parameters
    ----------
    dz:
        Vertical displacement in metres (sign is ignored; use absolute value).

    Returns
    -------
    float
        Energy cost (arbitrary units).
    """
    return 5.0 * np.sqrt(abs(dz))


# ---------------------------------------------------------------------------
# Step 1 – create_graph
# ---------------------------------------------------------------------------


def create_graph(
    network_name: str,
    dataset_root: str = "/content/GMNS_Plus_Dataset",
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    buffer_km: float = DEFAULT_BUFFER_KM,
) -> None:
    """Generate the H3 zone file for a GMNS-Plus network.

    Reads ``node.csv`` from *network_path*, tessellates the study area with
    H3 hexagons, and writes ``zone.csv`` back to the same directory.  Run
    this **once** before calling :func:`parse_graph`.

    Parameters
    ----------
    network_name:
        Subdirectory name inside *dataset_root* (e.g. ``"32_Phoenix_City"``).
    dataset_root:
        Absolute path to the root of the GMNS-Plus dataset clone.
    h3_resolution:
        H3 resolution for the hexagonal tessellation (default 7, ~1 km²).
    buffer_km:
        Buffer in kilometres applied around the convex hull of the nodes
        before tessellating (default 1 km).

    Raises
    ------
    FileNotFoundError
        If ``node.csv`` does not exist at the expected path.
    """
    network_path = Path(dataset_root) / network_name
    node_file = network_path / "node.csv"
    zone_file = network_path / "zone.csv"

    if not node_file.exists():
        raise FileNotFoundError(f"node.csv not found at {node_file}")

    logger.info("Creating graph for %s", network_name)

    # --- Load nodes ---
    node_df = pd.read_csv(node_file)

    if "geometry" not in node_df.columns:
        if {"x_coord", "y_coord"}.issubset(node_df.columns):
            node_df["geometry"] = node_df.apply(
                lambda row: Point(row["x_coord"], row["y_coord"]), axis=1
            )
        else:
            raise ValueError(
                "node.csv has neither a 'geometry' column nor "
                "'x_coord'/'y_coord' columns."
            )
    else:
        node_df["geometry"] = node_df["geometry"].apply(wkt.loads)

    node_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs="EPSG:4326")

    # --- Convex hull + buffer ---
    convex_hull = node_gdf.geometry.union_all().convex_hull
    buffer_deg = buffer_km / 111.0  # ~1° ≈ 111 km
    buffered_area = convex_hull.buffer(buffer_deg)
    minx, miny, maxx, maxy = buffered_area.bounds

    # --- H3 tessellation via grid scan ---
    lats = np.arange(miny, maxy, _GRID_STEP_DEG)
    lons = np.arange(minx, maxx, _GRID_STEP_DEG)

    h3_cells: set[int] = set()
    for lat in lats:
        for lon in lons:
            if buffered_area.contains(Point(lon, lat)):
                h3_cells.add(h3.latlng_to_cell(lat, lon, h3_resolution))

    # --- Build zone DataFrame ---
    zone_rows = []
    for cell in h3_cells:
        lat, lon = h3.cell_to_latlng(cell)
        boundary_latlon = h3.cell_to_boundary(cell)
        boundary_lonlat = [(lng, lat_) for lat_, lng in boundary_latlon]
        hex_polygon = Polygon(boundary_lonlat)

        zone_rows.append(
            {
                "zone_id": cell,
                "x_coord": lon,
                "y_coord": lat,
                "geometry": Point(lon, lat).wkt,
                "H3_geometry": hex_polygon.wkt,
            }
        )

    zone_df = pd.DataFrame(zone_rows)
    zone_df.to_csv(zone_file, index=False)
    logger.info("Wrote %d H3 zones to %s", len(zone_df), zone_file)


# ---------------------------------------------------------------------------
# Step 2 – parse_graph
# ---------------------------------------------------------------------------


def parse_graph(
    network_name: str,
    dataset_root: str = "/content/GMNS_Plus_Dataset",
) -> Tuple[folium.Map, pd.DataFrame, pd.DataFrame, Optional[gpd.GeoDataFrame]]:
    """Load a GMNS-Plus network and return an interactive Folium map.

    Parameters
    ----------
    network_name:
        Subdirectory name inside *dataset_root* (e.g. ``"32_Phoenix_City"``).
    dataset_root:
        Absolute path to the root of the GMNS-Plus dataset clone.

    Returns
    -------
    m : folium.Map
        Interactive map with nodes, links, and H3 zone polygons rendered.
    node_df : pd.DataFrame
        Raw node table with a ``geometry`` column populated.
    link_df : pd.DataFrame
        Raw link table (geometry column populated where available).
    link_gdf : gpd.GeoDataFrame or None
        Link GeoDataFrame; ``None`` if the link file has no geometry column.
    """
    network_path = Path(dataset_root) / network_name

    node_df = pd.read_csv(
        network_path / "node.csv", encoding="latin1", low_memory=False
    )
    link_df = pd.read_csv(
        network_path / "link.csv", encoding="latin1", low_memory=False
    )
    zone_df = pd.read_csv(
        network_path / "zone.csv", encoding="latin1", low_memory=False
    )

    logger.info(
        "Loaded %d nodes, %d links, %d zones",
        len(node_df), len(link_df), len(zone_df),
    )

    # --- Node geometry ---
    node_df["geometry"] = node_df.apply(
        lambda row: Point(row["x_coord"], row["y_coord"]), axis=1
    )
    node_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs="EPSG:4326")

    # --- Zone geometry ---
    if "H3_geometry" in zone_df.columns:
        if isinstance(zone_df["H3_geometry"].iloc[0], str):
            zone_df["H3_geometry"] = zone_df["H3_geometry"].apply(wkt.loads)
        zone_df["geometry"] = zone_df["H3_geometry"]
    else:
        zone_df["geometry"] = zone_df["geometry"].apply(wkt.loads)

    zone_gdf = gpd.GeoDataFrame(zone_df, geometry="geometry", crs="EPSG:4326")

    # --- Link geometry ---
    link_gdf: Optional[gpd.GeoDataFrame] = None
    if (
        "geometry" in link_df.columns
        and isinstance(link_df["geometry"].iloc[0], str)
    ):
        link_df["geometry"] = link_df["geometry"].apply(wkt.loads)
        link_gdf = gpd.GeoDataFrame(link_df, geometry="geometry", crs="EPSG:4326")

    # --- Build Folium map ---
    center_lat = float(node_df["y_coord"].mean())
    center_lon = float(node_df["x_coord"].mean())
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="cartodbpositron",
    )

    # Zone polygons
    for _, row in zone_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, Polygon):
            folium.Polygon(
                locations=[(lat, lon) for lon, lat in geom.exterior.coords],
                color="purple",
                weight=2,
                fill=True,
                fill_opacity=0.3,
                tooltip=f"Zone ID: {row['zone_id']}",
            ).add_to(m)

    # Link polylines
    if link_gdf is not None:
        for _, row in link_gdf.iterrows():
            if isinstance(row.geometry, LineString):
                travel_time = row.get("vdf_fftt", 5)
                color = (
                    "green" if travel_time < 1 else "orange" if travel_time < 2 else "red"
                )
                folium.PolyLine(
                    locations=[(lat, lon) for lon, lat in row.geometry.coords],
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"vdf_fftt: {row.get('vdf_fftt', 'N/A')}",
                ).add_to(m)

    # Node markers
    for _, row in node_df.iterrows():
        folium.CircleMarker(
            location=[row["y_coord"], row["x_coord"]],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Node ID: {row['node_id']}",
        ).add_to(m)

    return m, node_df, link_df, link_gdf


# ---------------------------------------------------------------------------
# Step 3 – convert_to_3d
# ---------------------------------------------------------------------------


def convert_to_3d(
    node_df: pd.DataFrame,
    link_df: pd.DataFrame,
    altitudes: list[int] = DEFAULT_ALTITUDES,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Expand a 2-D node/link dataset into a multi-altitude 3-D graph.

    Each 2-D node is replicated at every altitude layer.  Horizontal links
    exist at each layer (with altitude-dependent free speeds).  Vertical
    links connect adjacent layers at each node position.

    Parameters
    ----------
    node_df:
        2-D node DataFrame; must contain ``node_id``, ``x_coord``,
        ``y_coord``, ``zone_id``, and ``geometry``.
    link_df:
        2-D link DataFrame; must contain ``link_id``, ``from_node_id``,
        ``to_node_id``, ``dir_flag``, ``length``, and ``geometry``.
    altitudes:
        Sorted list of altitude layers in metres (default ``[0, 50, 100, 200, 400]``).

    Returns
    -------
    node_3d_df : pd.DataFrame
        3-D node table with columns ``node_id``, ``original_node_id``,
        ``zone_id``, ``x_coord``, ``y_coord``, ``z_coord``, ``geometry``.
    link_3d_df : pd.DataFrame
        3-D link table including both horizontal and vertical links.
    node2d_to_3d : dict[Any, str]
        Mapping from a 2-D ``node_id`` to its ground-level (z=0) 3-D ``node_id``.
    """
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

    link_3d_rows: list[dict] = []

    # Horizontal links – one copy per altitude layer
    for _, row in link_df.iterrows():
        for z in altitudes:
            link_3d_rows.append(
                {
                    "link_id": f"{row.link_id}_z{z}",
                    "from_node_id": f"{row.from_node_id}_z{z}",
                    "to_node_id": f"{row.to_node_id}_z{z}",
                    "dir_flag": row.dir_flag,
                    "length": row.length,
                    "free_speed": 12.0 + 0.25 * z,
                    "link_type": "horizontal",
                    "altitude": z,
                    "geometry": row.geometry,
                }
            )

    # Vertical links – connect adjacent altitude layers at each node
    for _, row in node_df.iterrows():
        for z1, z2 in zip(altitudes[:-1], altitudes[1:]):
            dz = z2 - z1
            link_3d_rows.append(
                {
                    "link_id": f"V_{row.node_id}_{z1}_{z2}",
                    "from_node_id": f"{row.node_id}_z{z1}",
                    "to_node_id": f"{row.node_id}_z{z2}",
                    "dir_flag": 1,
                    "length": dz,
                    "free_speed": _VERTICAL_SPEED_MS,
                    "link_type": "vertical",
                    "altitude": f"{z1}->{z2}",
                    "energy_cost": dz * _CLIMB_COST_PER_METER,
                    "geometry": None,
                }
            )

    node_3d_df = pd.DataFrame(node_3d_rows)
    link_3d_df = pd.DataFrame(link_3d_rows)

    # Map each 2-D node ID to its ground-level 3-D node ID (z = altitudes[0])
    ground_z = altitudes[0]
    node2d_to_3d: dict = (
        node_3d_df[node_3d_df["z_coord"] == ground_z]
        .set_index("original_node_id")["node_id"]
        .to_dict()
    )

    logger.info(
        "3-D expansion: %d nodes → %d 3-D nodes, %d 3-D links",
        len(node_df), len(node_3d_df), len(link_3d_df),
    )
    return node_3d_df, link_3d_df, node2d_to_3d


# ---------------------------------------------------------------------------
# Step 4 – build_3d_graph
# ---------------------------------------------------------------------------


def build_3d_graph(
    node_3d_df: pd.DataFrame,
    link_3d_df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.05,
    C_CLIMB: float = 8.0,
    C_DESCENT: float = 1.0,
) -> nx.DiGraph:
    """Build a weighted NetworkX DiGraph from 3-D node and link DataFrames.

    Edge weight is::

        weight = alpha * travel_time + beta * energy   [seconds + energy units]

    In the current implementation ``weight`` is set to ``travel_time`` only
    (matching the notebook); expose *alpha* / *beta* to re-introduce energy.

    Parameters
    ----------
    node_3d_df:
        Output of :func:`convert_to_3d` – 3-D node table.
    link_3d_df:
        Output of :func:`convert_to_3d` – 3-D link table.
    alpha:
        Weight on travel-time in the composite edge cost (default 1.0).
    beta:
        Weight on energy in the composite edge cost (default 0.05).
    C_CLIMB:
        Energy multiplier for climbing links (unused in current weight formula).
    C_DESCENT:
        Energy multiplier for descent links (J/m equivalent).

    Returns
    -------
    nx.DiGraph
        Directed graph with node attributes ``x``, ``y``, ``z`` and edge
        attributes ``weight``, ``time``, ``energy``, ``link_type``.
    """
    G: nx.DiGraph = nx.DiGraph()

    for _, row in node_3d_df.iterrows():
        G.add_node(
            row["node_id"],
            x=row["x_coord"],
            y=row["y_coord"],
            z=row["z_coord"],
        )

    for _, row in link_3d_df.iterrows():
        length = row["length"]
        if not length or np.isnan(length):
            length = (
                row["geometry"].length if row["geometry"] is not None else 1e-3
            )

        if row["link_type"] == "horizontal":
            z = float(row["altitude"])
            speed = horizontal_speed(z)
            travel_time = length / speed
            energy = horizontal_energy_per_meter(z) * length

        else:  # vertical
            z_from, z_to = map(float, str(row["altitude"]).split("->"))
            dz = z_to - z_from
            travel_time = abs(dz) / 100.0
            energy = vertical_energy(dz) if dz > 0 else C_DESCENT * abs(dz)

        # Composite cost (alpha * time + beta * energy); weight = time for now.
        weight = alpha * travel_time  # extend with `+ beta * energy` as needed

        common_attrs = dict(
            weight=weight,
            time=travel_time,
            energy=energy,
            link_type=row["link_type"],
        )

        G.add_edge(row["from_node_id"], row["to_node_id"], **common_attrs)

        # Add reverse edge for bidirectional links (dir_flag == 1 in GMNS means
        # two-way; adjust the condition if your dataset uses a different convention).
        if row["dir_flag"] == 1:
            G.add_edge(row["to_node_id"], row["from_node_id"], **common_attrs)

    logger.info(
        "Built 3-D graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ---------------------------------------------------------------------------
# Step 5 – build_csr
# ---------------------------------------------------------------------------


def build_csr(
    G: nx.DiGraph,
) -> Tuple[csr_matrix, list, dict]:
    """Compile a SciPy CSR matrix from a NetworkX graph for fast Dijkstra.

    The CSR matrix is undirected (both (u, v) and (v, u) are inserted) to
    allow ``scipy.sparse.csgraph.dijkstra`` to be called without a directed
    flag.  Adjust if your use-case requires strictly directed shortest paths.

    Parameters
    ----------
    G:
        Weighted NetworkX DiGraph (output of :func:`build_3d_graph`).

    Returns
    -------
    csr : csr_matrix
        Sparse adjacency matrix of shape (N, N) with ``float`` edge weights.
    node_list : list[str]
        Ordered list of node IDs; ``node_list[i]`` corresponds to row/column
        *i* in *csr*.
    node_to_idx : dict[str, int]
        Inverse mapping from node ID → matrix index.
    """
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for u, v, attr in G.edges(data=True):
        w = float(attr.get("weight", 1.0))
        iu, iv = node_to_idx[u], node_to_idx[v]
        rows.append(iu); cols.append(iv); data.append(w)
        rows.append(iv); cols.append(iu); data.append(w)  # symmetric / undirected

    csr = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    logger.info("Built CSR matrix: shape %s, nnz=%d", csr.shape, csr.nnz)
    return csr, node_list, node_to_idx
