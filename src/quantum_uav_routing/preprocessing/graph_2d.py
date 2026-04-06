"""
graph_2d.py
===========
Load and visualise a 2-D GMNS-Plus transport network.

This module handles everything that lives in 2-D geographic space:

* ``create_zones`` – tessellate the node convex-hull with H3 hexagons and
  write a ``zone.csv`` file.  Call this **once** when setting up a new city
  dataset; subsequent runs read the cached file.
* ``load_network`` – read ``node.csv``, ``link.csv``, and ``zone.csv``,
  parse geometries, and return a Folium map together with raw DataFrames for
  downstream processing.

The two functions correspond to the notebook's ``create_graph`` /
``parse_graph`` cells, renamed to avoid collision with NetworkX's own
``create_graph`` helpers and to better reflect their roles in the pipeline.

Dependencies
------------
    pandas, geopandas, shapely, h3, numpy, folium
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import folium
import geopandas as gpd
import h3.api.basic_int as h3
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon

__all__ = [
    "create_zones",
    "load_network",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: H3 hexagonal resolution used when tessellating the study area.
#: Resolution 7 ≈ 1.2 km² per cell, suitable for city-scale UAM.
DEFAULT_H3_RESOLUTION: int = 7

#: Buffer (km) applied around the convex hull of nodes before tessellating.
DEFAULT_BUFFER_KM: float = 1.0

#: Grid scan step (degrees) when seeding H3 cells; ~300 m at mid-latitudes.
_GRID_STEP_DEG: float = 0.003


# ---------------------------------------------------------------------------
# Step 1 – Zone generation (run once per city)
# ---------------------------------------------------------------------------


def create_zones(
    network_name: str,
    dataset_root: str | Path = "data/raw",
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    buffer_km: float = DEFAULT_BUFFER_KM,
    overwrite: bool = False,
) -> Path:
    """Generate an H3 hexagonal zone file for a GMNS-Plus network.

    Reads ``node.csv`` from *<dataset_root>/<network_name>/*, tessellates the
    bounding area with H3 hexagons at *h3_resolution*, and writes the result
    to ``zone.csv`` in the same directory.

    Parameters
    ----------
    network_name:
        Subdirectory name inside *dataset_root* (e.g. ``"phoenix"``).
    dataset_root:
        Root directory of the GMNS-Plus datasets.  Defaults to ``data/raw``
        relative to the working directory.
    h3_resolution:
        H3 resolution for tessellation (default 7, ≈ 1.2 km² / cell).
    buffer_km:
        Buffer in kilometres around the node convex hull (default 1 km).
    overwrite:
        If ``False`` (default) and ``zone.csv`` already exists, the function
        returns immediately without regenerating.

    Returns
    -------
    Path
        Absolute path to the written ``zone.csv``.

    Raises
    ------
    FileNotFoundError
        If ``node.csv`` does not exist at the expected path.
    ValueError
        If ``node.csv`` has neither a ``geometry`` column nor both
        ``x_coord`` and ``y_coord`` columns.
    """
    network_path = Path(dataset_root) / network_name
    node_file = network_path / "node.csv"
    zone_file = network_path / "zone.csv"

    if not node_file.exists():
        raise FileNotFoundError(f"node.csv not found: {node_file}")

    if zone_file.exists() and not overwrite:
        logger.info("zone.csv already exists at %s – skipping generation.", zone_file)
        return zone_file.resolve()

    logger.info("Generating H3 zones for '%s' (resolution=%d).", network_name, h3_resolution)

    # --- Load node geometry ---
    node_df = pd.read_csv(node_file)
    node_df = _ensure_node_geometry(node_df)
    node_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs="EPSG:4326")

    # --- Convex hull + buffer ---
    convex_hull = node_gdf.geometry.union_all().convex_hull
    buffer_deg = buffer_km / 111.0  # approximate: 1° ≈ 111 km
    buffered_area = convex_hull.buffer(buffer_deg)
    minx, miny, maxx, maxy = buffered_area.bounds

    # --- Grid-scan to seed H3 cells ---
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
        # h3 returns (lat, lon); shapely wants (lon, lat)
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
    network_path.mkdir(parents=True, exist_ok=True)
    zone_df.to_csv(zone_file, index=False)
    logger.info("Wrote %d H3 zones → %s", len(zone_df), zone_file)
    return zone_file.resolve()


# ---------------------------------------------------------------------------
# Step 2 – Network loading
# ---------------------------------------------------------------------------


def load_network(
    network_name: str,
    dataset_root: str | Path = "data/raw",
) -> Tuple[folium.Map, pd.DataFrame, pd.DataFrame, Optional[gpd.GeoDataFrame]]:
    """Load a GMNS-Plus network from CSV files and return a Folium map.

    Reads ``node.csv``, ``link.csv``, and ``zone.csv`` from
    *<dataset_root>/<network_name>/*.  Call :func:`create_zones` first if
    ``zone.csv`` does not yet exist.

    Parameters
    ----------
    network_name:
        Subdirectory name inside *dataset_root* (e.g. ``"phoenix"``).
    dataset_root:
        Root directory of the GMNS-Plus datasets.

    Returns
    -------
    map : folium.Map
        Interactive Folium map with H3 zone polygons, links, and node markers.
    node_df : pd.DataFrame
        Node table with a ``geometry`` (shapely ``Point``) column added.
    link_df : pd.DataFrame
        Link table; geometry column parsed to shapely objects where present.
    link_gdf : gpd.GeoDataFrame or None
        GeoDataFrame version of the link table; ``None`` if the link file
        has no parseable geometry column.

    Raises
    ------
    FileNotFoundError
        If any of the three required CSV files is missing.
    """
    network_path = Path(dataset_root) / network_name

    for fname in ("node.csv", "link.csv", "zone.csv"):
        fpath = network_path / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"{fname} not found at {fpath}.  "
                f"Run create_zones('{network_name}') first if zone.csv is missing."
            )

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
        "Loaded '%s': %d nodes, %d links, %d zones.",
        network_name, len(node_df), len(link_df), len(zone_df),
    )

    # --- Parse geometries ---
    node_df = _ensure_node_geometry(node_df)
    node_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs="EPSG:4326")

    zone_gdf = _parse_zone_geometry(zone_df)
    link_gdf = _parse_link_geometry(link_df)

    # --- Build Folium map ---
    center_lat = float(node_df["y_coord"].mean())
    center_lon = float(node_df["x_coord"].mean())
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="cartodbpositron",
    )

    _draw_zones(fmap, zone_gdf)
    _draw_links(fmap, link_gdf)
    _draw_nodes(fmap, node_df)

    return fmap, node_df, link_df, link_gdf


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_node_geometry(node_df: pd.DataFrame) -> pd.DataFrame:
    """Return *node_df* with a populated shapely ``geometry`` column."""
    df = node_df.copy()
    if "geometry" in df.columns:
        if df["geometry"].dtype == object:
            df["geometry"] = df["geometry"].apply(wkt.loads)
    elif {"x_coord", "y_coord"}.issubset(df.columns):
        df["geometry"] = df.apply(
            lambda row: Point(row["x_coord"], row["y_coord"]), axis=1
        )
    else:
        raise ValueError(
            "node.csv must contain either a 'geometry' column or both "
            "'x_coord' and 'y_coord' columns."
        )
    return df


def _parse_zone_geometry(zone_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame from a zone CSV, preferring H3 polygon geometry."""
    df = zone_df.copy()
    if "H3_geometry" in df.columns:
        if isinstance(df["H3_geometry"].iloc[0], str):
            df["H3_geometry"] = df["H3_geometry"].apply(wkt.loads)
        df["geometry"] = df["H3_geometry"]
    else:
        df["geometry"] = df["geometry"].apply(
            lambda g: wkt.loads(g) if isinstance(g, str) else g
        )
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def _parse_link_geometry(link_df: pd.DataFrame) -> Optional[gpd.GeoDataFrame]:
    """Return a GeoDataFrame if the link table has a parseable geometry column."""
    if "geometry" not in link_df.columns:
        return None
    df = link_df.copy()
    if isinstance(df["geometry"].iloc[0], str):
        df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def _link_color(vdf_fftt: float) -> str:
    """Map free-flow travel time to a Folium color string."""
    if vdf_fftt < 1:
        return "green"
    if vdf_fftt < 2:
        return "orange"
    return "red"


def _draw_zones(fmap: folium.Map, zone_gdf: gpd.GeoDataFrame) -> None:
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
            ).add_to(fmap)


def _draw_links(
    fmap: folium.Map, link_gdf: Optional[gpd.GeoDataFrame]
) -> None:
    if link_gdf is None:
        return
    for _, row in link_gdf.iterrows():
        if isinstance(row.geometry, LineString):
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in row.geometry.coords],
                color=_link_color(row.get("vdf_fftt", 5)),
                weight=3,
                opacity=0.8,
                popup=f"vdf_fftt: {row.get('vdf_fftt', 'N/A')}",
            ).add_to(fmap)


def _draw_nodes(fmap: folium.Map, node_df: pd.DataFrame) -> None:
    for _, row in node_df.iterrows():
        folium.CircleMarker(
            location=[row["y_coord"], row["x_coord"]],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Node ID: {row['node_id']}",
        ).add_to(fmap)
