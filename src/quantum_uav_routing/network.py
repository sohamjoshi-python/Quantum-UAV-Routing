from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import h3.api.basic_int as h3
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import geometry, wkt
from shapely.geometry import Point, Polygon


ALTITUDES = [0, 50, 100, 200, 400]


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


def create_zone_file(network_dir: str | Path, h3_resolution: int = 7, buffer_km: float = 1.0) -> Path:
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


def load_gmns_network(network_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    network_dir = Path(network_dir)

    node_df = pd.read_csv(network_dir / "node.csv", encoding="latin1", low_memory=False)
    link_df = pd.read_csv(network_dir / "link.csv", encoding="latin1", low_memory=False)

    zone_path = network_dir / "zone.csv"
    if not zone_path.exists():
        create_zone_file(network_dir)

    zone_df = pd.read_csv(zone_path, encoding="latin1", low_memory=False)

    node_df = _ensure_geometry_from_xy(node_df)

    if "H3_geometry" in zone_df.columns:
        if len(zone_df) > 0 and isinstance(zone_df["H3_geometry"].iloc[0], str):
            zone_df["H3_geometry"] = zone_df["H3_geometry"].apply(wkt.loads)
        zone_df["geometry"] = zone_df["H3_geometry"]
    else:
        zone_df = _ensure_geometry_from_xy(zone_df)

    if "geometry" in link_df.columns and len(link_df) > 0 and isinstance(link_df["geometry"].iloc[0], str):
        link_df["geometry"] = link_df["geometry"].apply(wkt.loads)

    return node_df, link_df, zone_df


def convert_to_3d(node_df: pd.DataFrame, link_df: pd.DataFrame, altitudes: list[int] | None = None):
    if altitudes is None:
        altitudes = ALTITUDES

    node_rows = []
    for _, row in node_df.iterrows():
        for z in altitudes:
            node_rows.append(
                {
                    "node_id": f"{row.node_id}_z{z}",
                    "original_node_id": row.node_id,
                    "zone_id": row.zone_id if "zone_id" in row else None,
                    "x_coord": row.x_coord,
                    "y_coord": row.y_coord,
                    "z_coord": z,
                    "geometry": row.geometry if "geometry" in row else None,
                }
            )

    link_rows = []
    for _, row in link_df.iterrows():
        for z in altitudes:
            link_rows.append(
                {
                    "link_id": f"{row.link_id}_z{z}",
                    "from_node_id": f"{row.from_node_id}_z{z}",
                    "to_node_id": f"{row.to_node_id}_z{z}",
                    "dir_flag": row.dir_flag,
                    "length": row.length,
                    "free_speed": 12 + 0.25 * z,
                    "link_type": "horizontal",
                    "altitude": z,
                    "geometry": row.geometry if "geometry" in row else None,
                }
            )

    for _, row in node_df.iterrows():
        for z1, z2 in zip(altitudes[:-1], altitudes[1:]):
            dz = z2 - z1
            link_rows.append(
                {
                    "link_id": f"V_{row.node_id}_{z1}_{z2}",
                    "from_node_id": f"{row.node_id}_z{z1}",
                    "to_node_id": f"{row.node_id}_z{z2}",
                    "dir_flag": 1,
                    "length": dz,
                    "free_speed": 4,
                    "link_type": "vertical",
                    "altitude": f"{z1}->{z2}",
                    "geometry": None,
                }
            )

    node_3d_df = pd.DataFrame(node_rows)
    link_3d_df = pd.DataFrame(link_rows)

    node2d_to_3d = (
        node_3d_df[node_3d_df["z_coord"] == max(altitudes)]
        .set_index("original_node_id")["node_id"]
        .to_dict()
    )

    return node_3d_df, link_3d_df, node2d_to_3d


def horizontal_speed(z: float) -> float:
    return 10 + 0.15 * np.sqrt(z)


def horizontal_energy_per_meter(z: float) -> float:
    base = 1.0
    reduction = 0.001 * z
    return max(0.3, base - reduction)


def vertical_energy(dz: float) -> float:
    return 5 * np.sqrt(abs(dz))


def build_3d_graph(
    node_3d_df: pd.DataFrame,
    link_3d_df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.05,
    c_descent: float = 1.0,
) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, row in node_3d_df.iterrows():
        G.add_node(row["node_id"], x=row["x_coord"], y=row["y_coord"], z=row["z_coord"])

    for _, row in link_3d_df.iterrows():
        length = row["length"]
        if pd.isna(length) or length == 0:
            geom = row.get("geometry")
            length = geom.length if geom is not None else 1e-3

        if row["link_type"] == "horizontal":
            z = float(row["altitude"])
            speed = horizontal_speed(z)
            travel_time = length / speed
            energy = horizontal_energy_per_meter(z) * length
        else:
            z_from, z_to = map(float, str(row["altitude"]).split("->"))
            dz = z_to - z_from
            travel_time = abs(dz) / 100.0
            energy = vertical_energy(dz) if dz > 0 else c_descent * abs(dz)

        weight = alpha * travel_time + beta * energy

        G.add_edge(
            row["from_node_id"],
            row["to_node_id"],
            weight=weight,
            time=travel_time,
            energy=energy,
            link_type=row["link_type"],
        )

        if int(row["dir_flag"]) == 1:
            G.add_edge(
                row["to_node_id"],
                row["from_node_id"],
                weight=weight,
                time=travel_time,
                energy=energy,
                link_type=row["link_type"],
            )

    return G


class ShortestPathOracle:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.csr = self._build_csr(G)
        self._cache: dict[str, np.ndarray] = {}

    def _build_csr(self, G: nx.Graph) -> csr_matrix:
        rows = []
        cols = []
        data = []

        for u, v, attrs in G.edges(data=True):
            ui = self.node_to_idx[u]
            vi = self.node_to_idx[v]
            w = float(attrs.get("weight", 1.0))
            rows.append(ui)
            cols.append(vi)
            data.append(w)

        n = len(self.nodes)
        return csr_matrix((data, (rows, cols)), shape=(n, n))

    def dist_from(self, src: str) -> np.ndarray:
        src = str(src)
        if src not in self._cache:
            src_idx = self.node_to_idx[src]
            self._cache[src] = dijkstra(self.csr, directed=True, indices=src_idx, return_predecessors=False)
        return self._cache[src]

    def shortest_path(self, src: str, dst: str) -> float:
        src = str(src)
        dst = str(dst)
        dist = self.dist_from(src)
        out = float(dist[self.node_to_idx[dst]])
        return out if np.isfinite(out) else float("inf")

    def clear(self) -> None:
        self._cache.clear()