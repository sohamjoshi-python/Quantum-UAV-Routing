from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

ALTITUDES = [0, 50, 100, 200, 400]


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
                    "geometry": row.geometry,
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

    climb_cost_per_meter = 0.02
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
                    "energy_cost": dz * climb_cost_per_meter,
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


def horizontal_speed(z):
    return 10 + 0.15 * np.sqrt(z)


def vertical_energy(dz):
    return 5 * np.sqrt(abs(dz))


def horizontal_energy_per_meter(z):
    base = 1.0
    reduction = 0.001 * z
    return max(0.3, base - reduction)


def build_3d_graph(
    node_3d_df,
    link_3d_df,
    alpha=1.0,
    beta=0.05,
    c_climb=8.0,
    c_descent=1.0,
):
    G = nx.DiGraph()

    for _, row in node_3d_df.iterrows():
        G.add_node(
            row["node_id"],
            x=row["x_coord"],
            y=row["y_coord"],
            z=row["z_coord"],
        )

    for _, row in link_3d_df.iterrows():
        length = row["length"]
        geom = row.get("geometry", None)
        if pd.isna(length) or length == 0:
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
        weight = travel_time

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