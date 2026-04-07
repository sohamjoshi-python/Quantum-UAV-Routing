from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import h3.api.basic_int as h3
import numpy as np
import pandas as pd
from shapely import geometry, wkt
from shapely.geometry import LineString, Point, Polygon


def _ensure_geometry_from_xy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "geometry" not in out.columns:
        if "x_coord" not in out.columns or "y_coord" not in out.columns:
            raise ValueError("Need either geometry column or x_coord/y_coord columns.")
        out["geometry"] = out.apply(lambda row: Point(row["x_coord"], row["y_coord"]), axis=1)
    else:
        if len(out) > 0 and isinstance(out["geometry"].iloc[0], str):
            out["geometry"] = out["geometry"].apply(wkt.loads)
    return out


def create_graph(network_dir: str | Path, h3_resolution: int = 7, buffer_km: float = 1.0) -> Path:
    network_dir = Path(network_dir)
    node_file = network_dir / "node.csv"
    zone_file = network_dir / "zone.csv"

    node_df = pd.read_csv(node_file)
    node_df = _ensure_geometry_from_xy(node_df)
    node_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs="EPSG:4326")

    convex_hull = node_gdf.geometry.union_all().convex_hull
    buffer_degree = buffer_km / 111.0
    buffered_area = convex_hull.buffer(buffer_degree)
    minx, miny, maxx, maxy = buffered_area.bounds

    step = 0.003
    lats = np.arange(miny, maxy, step)
    lons = np.arange(minx, maxx, step)

    h3_cells: set[int] = set()
    for lat in lats:
        for lon in lons:
            if buffered_area.contains(geometry.Point(lon, lat)):
                h3_cells.add(h3.latlng_to_cell(lat, lon, h3_resolution))

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
    return zone_file


def parse_graph(network_dir: str | Path):
    network_dir = Path(network_dir)

    node_path = network_dir / "node.csv"
    link_path = network_dir / "link.csv"
    zone_path = network_dir / "zone.csv"

    if not node_path.exists():
        raise FileNotFoundError(f"Missing {node_path}")
    if not link_path.exists():
        raise FileNotFoundError(f"Missing {link_path}")
    if not zone_path.exists():
        create_graph(network_dir)

    node_df = pd.read_csv(node_path, encoding="latin1", low_memory=False)
    link_df = pd.read_csv(link_path, encoding="latin1", low_memory=False)
    zone_df = pd.read_csv(zone_path, encoding="latin1", low_memory=False)

    node_df = _ensure_geometry_from_xy(node_df)
    node_gdf = gpd.GeoDataFrame(node_df, geometry="geometry", crs="EPSG:4326")

    if "H3_geometry" in zone_df.columns:
        if len(zone_df) > 0 and isinstance(zone_df["H3_geometry"].iloc[0], str):
            zone_df["H3_geometry"] = zone_df["H3_geometry"].apply(wkt.loads)
        zone_df["geometry"] = zone_df["H3_geometry"]
    else:
        zone_df = _ensure_geometry_from_xy(zone_df)
    zone_gdf = gpd.GeoDataFrame(zone_df, geometry="geometry", crs="EPSG:4326")

    if "geometry" in link_df.columns and len(link_df) > 0 and isinstance(link_df["geometry"].iloc[0], str):
        link_df["geometry"] = link_df["geometry"].apply(wkt.loads)
        link_gdf = gpd.GeoDataFrame(link_df, geometry="geometry", crs="EPSG:4326")
    else:
        link_gdf = None

    center_lat = float(node_df["y_coord"].mean())
    center_lon = float(node_df["x_coord"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")

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

    def get_color_by_travel_time(time_val):
        if time_val < 1:
            return "green"
        if time_val < 2:
            return "orange"
        return "red"

    if link_gdf is not None:
        for _, row in link_gdf.iterrows():
            if isinstance(row.geometry, LineString):
                folium.PolyLine(
                    locations=[(lat, lon) for lon, lat in row.geometry.coords],
                    color=get_color_by_travel_time(row.get("vdf_fftt", 5)),
                    weight=3,
                    opacity=0.8,
                    popup=f"vdf_fftt: {row.get('vdf_fftt', 'N/A')}",
                ).add_to(fmap)

    for _, row in node_df.iterrows():
        folium.CircleMarker(
            location=[row["y_coord"], row["x_coord"]],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Node ID: {row['node_id']}",
        ).add_to(fmap)

    return fmap, node_df, link_df, link_gdf