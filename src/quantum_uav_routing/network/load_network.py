"""
load_gmns.py

Loads the GMNS network (node.csv, link.csv) from the GMNS_Plus_Dataset repo.

"""

import argparse
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

import folium
import geopandas as gpd
import h3.api.basic_int as h3
import numpy as np
import pandas as pd
from branca.element import Element
from shapely import geometry, wkt
from shapely.geometry import LineString, Point, Polygon

REPO_URL = "https://github.com/HanZhengIntelliTransport/GMNS_Plus_Dataset.git"
REPO_DIR = "data/GMNS_Plus_Dataset"

H3_RESOLUTION = 7
BUFFER_KM = 1
GRID_STEP_DEGREES = 0.003  # ~300m


def _download_repo_archive() -> bytes:
    """Download the GMNS dataset repository from GitHub as a zip archive."""
    archive_urls = [
        "https://codeload.github.com/HanZhengIntelliTransport/GMNS_Plus_Dataset/zip/refs/heads/main",
        "https://codeload.github.com/HanZhengIntelliTransport/GMNS_Plus_Dataset/zip/refs/heads/master",
    ]

    last_error = None
    for archive_url in archive_urls:
        try:
            with urllib.request.urlopen(archive_url, timeout=60) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - exercised in real environments
            last_error = exc

    raise RuntimeError(f"Could not download GMNS dataset archive: {last_error}")


def clone_dataset_repo(force: bool = False) -> None:
    """Clone or extract the GMNS_Plus_Dataset into the configured data folder."""
    if force and os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)

    if os.path.isdir(REPO_DIR):
        print(f"{REPO_DIR} already exists, skipping clone.")
        return

    parent_dir = os.path.dirname(REPO_DIR)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if shutil.which("git"):
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
        return

    print("Git not found; downloading the dataset archive instead...")
    archive_bytes = _download_repo_archive()

    with tempfile.TemporaryDirectory(prefix="gmns_repo_", dir=os.getcwd()) as temp_dir:
        archive_path = os.path.join(temp_dir, "repo.zip")
        with open(archive_path, "wb") as archive_file:
            archive_file.write(archive_bytes)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_dir)

        extracted_candidates = [
            os.path.join(temp_dir, entry)
            for entry in os.listdir(temp_dir)
            if entry != os.path.basename(archive_path)
        ]
        extracted_dirs = [path for path in extracted_candidates if os.path.isdir(path)]
        if not extracted_dirs:
            raise RuntimeError("Downloaded GMNS archive did not contain an extracted folder.")

        shutil.move(extracted_dirs[0], REPO_DIR)


def create_graph(network_name):

    # === CONFIGURATION ===
    H3_RESOLUTION = 7
    BUFFER_KM = 1
    network_path = f"{REPO_DIR}/{network_name}"
    node_file = f"{network_path}/node.csv"
    zone_file = f"{network_path}/zone.csv"

    # === 1. Load nodes ===
    node_df = pd.read_csv(node_file, low_memory=False)

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


def _add_title(map_obj, title: str) -> None:
    if not title:
        return

    title_html = f"""
    <h3 style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 9999;
        background: white;
        padding: 6px 10px;
        border: 1px solid #cccccc;
        border-radius: 4px;
        font-family: sans-serif;
    ">{title}</h3>
    """
    map_obj.get_root().html.add_child(Element(title_html))


def _color_by_travel_time(time):
    if time < 1:
        return "green"
    elif time < 2:
        return "orange"
    else:
        return "red"


def _link_style(row):
    facility_type = row.get("facility_type", "default")
    if pd.isna(facility_type):
        facility_type = "default"

    if facility_type == "bus":
        return "#1f77b4", 5, "Bus corridor"
    return "#666666", 3, "GMNS link"


def _has_bike_facility(row):
    bike_facility = row.get("bike_facility", None)
    return bike_facility is not None and pd.notna(bike_facility) and bool(bike_facility)


def _has_signal_priority(row):
    signal_priority = row.get("signal_priority_mode", None)
    return signal_priority is not None and pd.notna(signal_priority) and bool(signal_priority)


def _add_legend(map_obj, include_zones: bool = False) -> None:
    zone_row = """
        <div><span style="background: rgba(128, 0, 128, 0.3); border: 2px solid purple;"></span> H3 zone</div>
    """ if include_zones else ""

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border: 1px solid #cccccc;
        border-radius: 4px;
        font-family: sans-serif;
        font-size: 13px;
        line-height: 1.6;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
    ">
        <div style="font-weight: 600; margin-bottom: 4px;">Legend</div>
        <div><span style="background: #666666;"></span> GMNS link</div>
        <div><span style="background: #1f77b4;"></span> Bus corridor</div>
        <div><span style="border-top: 3px dashed #2ca02c; background: transparent;"></span> Bike facility</div>
        <div><span style="background: #ff9900; border-radius: 50%;"></span> Signal priority node</div>
        <div><span style="background: #0000ff; border-radius: 50%;"></span> GMNS node</div>
        {zone_row}
    </div>
    <style>
        .legend span {{
            display: inline-block;
            width: 24px;
            height: 4px;
            margin-right: 8px;
            vertical-align: middle;
        }}
    </style>
    """
    legend_html = legend_html.replace('<div style="', '<div class="legend" style="', 1)
    map_obj.get_root().html.add_child(Element(legend_html))


def _parse_wkt_if_needed(value):
    if isinstance(value, str):
        return wkt.loads(value)
    return value


def build_folium_map(node_df, link_df, zone_df=None, title: str = ""):
    """
    Build the Folium GMNS visualization used by parse_graph and render_gmns.

    Returns:
        m: folium.Map
        node_df: pd.DataFrame
        link_df: pd.DataFrame
        link_gdf: gpd.GeoDataFrame | None
    """
    node_df = node_df.copy()
    link_df = link_df.copy()

    if node_df.empty:
        raise ValueError("Cannot render an empty GMNS node table.")

    if "geometry" not in node_df.columns:
        node_df["geometry"] = node_df.apply(
            lambda row: Point(row["x_coord"], row["y_coord"]), axis=1
        )
    else:
        node_df["geometry"] = node_df["geometry"].apply(_parse_wkt_if_needed)

    center_lat = node_df["y_coord"].mean()
    center_lon = node_df["x_coord"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")
    _add_title(m, title)
    include_zones = zone_df is not None and not zone_df.empty

    if zone_df is not None and not zone_df.empty:
        zone_df = zone_df.copy()
        if "H3_geometry" in zone_df.columns:
            zone_df["H3_geometry"] = zone_df["H3_geometry"].apply(_parse_wkt_if_needed)
            zone_df["geometry"] = zone_df["H3_geometry"]
        elif "geometry" in zone_df.columns:
            zone_df["geometry"] = zone_df["geometry"].apply(_parse_wkt_if_needed)

        if "geometry" in zone_df.columns:
            zone_gdf = gpd.GeoDataFrame(zone_df, geometry="geometry", crs="EPSG:4326")
            for _, row in zone_gdf.iterrows():
                geom = row.geometry
                if isinstance(geom, Polygon):
                    folium.Polygon(
                        locations=[(lat, lon) for lon, lat in geom.exterior.coords],
                        color="purple",
                        weight=2,
                        fill=True,
                        fill_opacity=0.3,
                        tooltip=f"Zone ID: {row.get('zone_id', '')}",
                    ).add_to(m)

    link_gdf = None
    if "geometry" in link_df.columns and not link_df["geometry"].isna().all():
        link_df["geometry"] = link_df["geometry"].apply(_parse_wkt_if_needed)
        link_gdf = gpd.GeoDataFrame(link_df, geometry="geometry", crs="EPSG:4326")
    else:
        node_lookup = node_df.set_index("node_id")[["x_coord", "y_coord"]]
        geometries = []
        for _, row in link_df.iterrows():
            try:
                x0, y0 = node_lookup.loc[row["from_node_id"]]
                x1, y1 = node_lookup.loc[row["to_node_id"]]
                geometries.append(LineString([(x0, y0), (x1, y1)]))
            except KeyError:
                geometries.append(None)
        link_df["geometry"] = geometries
        link_gdf = gpd.GeoDataFrame(link_df.dropna(subset=["geometry"]), geometry="geometry", crs="EPSG:4326")

    for _, row in link_gdf.iterrows():
        if isinstance(row.geometry, LineString):
            color, weight, label = _link_style(row)
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in row.geometry.coords],
                color=color,
                weight=weight,
                opacity=0.8,
                popup=f"{label}; link_id: {row.get('link_id', 'N/A')}",
            ).add_to(m)
            if _has_bike_facility(row):
                folium.PolyLine(
                    locations=[(lat, lon) for lon, lat in row.geometry.coords],
                    color="#2ca02c",
                    weight=2,
                    opacity=0.9,
                    dash_array="6, 6",
                    popup=f"Bike facility: {row.get('bike_facility')}",
                ).add_to(m)

    for _, row in node_df.iterrows():
        has_signal_priority = _has_signal_priority(row)
        folium.CircleMarker(
            location=[row["y_coord"], row["x_coord"]],
            radius=5 if has_signal_priority else 3,
            color="#ff9900" if has_signal_priority else "blue",
            fill=True,
            fill_opacity=0.9 if has_signal_priority else 0.7,
            tooltip=f"Node ID: {row['node_id']}",
        ).add_to(m)

    _add_legend(m, include_zones=include_zones)
    return m, node_df, link_df, link_gdf


def parse_graph(network_name):
    # === CONFIGURATION ===
    network_path = f"{REPO_DIR}/{network_name}"

    # === LOAD FILES ===
    # Specify encoding as 'latin1' and low_memory=False to handle potential encoding issues and mixed types
    node_df = pd.read_csv(f"{network_path}/node.csv", encoding='latin1', low_memory=False)
    link_df = pd.read_csv(f"{network_path}/link.csv", encoding='latin1', low_memory=False)
    zone_df = pd.read_csv(f"{network_path}/zone.csv", encoding='latin1', low_memory=False)

    print(f"Nodes loaded: {len(node_df)}")
    print(f"Zones loaded: {len(zone_df)}")
    print(f"Links loaded: {len(link_df)}")

    return build_folium_map(node_df, link_df, zone_df)



def convert_to_3d(network_name, input_dir):
  # Convert to 3d
  node_df = pd.read_csv(f"{input_dir}/node.csv", encoding='latin1', low_memory=False)
  link_df = pd.read_csv(f"{input_dir}/link.csv", encoding='latin1', low_memory=False)

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

  output_dir = os.path.join(os.getcwd(), "data", "raw")
  os.makedirs(output_dir, exist_ok=True)

  node2d_mapping_df = pd.DataFrame(list(node2d_to_3d.items()), columns=['original_node_id', 'node_3d_id'])
  node2d_mapping_df.to_csv(os.path.join(output_dir, "node2d_to_3d_mapping.csv"), index=False)

  node_3d_df.to_csv(os.path.join(output_dir, "node_3d.csv"), index=False)
  link_3d_df.to_csv(os.path.join(output_dir, "link_3d.csv"), index=False)

  return node_3d_df, link_3d_df, node2d_to_3d



def load_gmns_network(network_dir):
    node_file = os.path.join(network_dir, "node.csv")
    link_file = os.path.join(network_dir, "link.csv")
    mapping_file = os.path.join(network_dir, "node2d_to_3d_mapping.csv")

    if not os.path.isfile(node_file):
        raise FileNotFoundError(f"Node file not found: {node_file}")
    if not os.path.isfile(link_file):
        raise FileNotFoundError(f"Link file not found: {link_file}")
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    node_df = pd.read_csv(node_file, low_memory=False)
    link_df = pd.read_csv(link_file, low_memory=False)
    mapping_df = pd.read_csv(mapping_file, low_memory=False)

    return node_df, link_df, mapping_df