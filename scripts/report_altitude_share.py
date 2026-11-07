"""
Share of shortest-path length at each altitude, on the Phoenix requests
the experiments actually sample.

Request origins and destinations are the ground-level nodes (_z0) from the
same scenarios, trials, and seeds as scripts/run_experiment.py. Each path is
the undirected Dijkstra path the experiment uses for that request's travel
time. Horizontal length is the GMNS link length in meters. Vertical length is
the altitude change, stored in feet on the links and converted here at
0.3048 m/ft so the shares use one unit.

Usage (from the repo root):
  python scripts/report_altitude_share.py
  python scripts/report_altitude_share.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from scipy.sparse.csgraph import dijkstra

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import quantum_uav_routing.network.shortest_path as sp_mod  # noqa: E402
import run_experiment as R  # noqa: E402

ALTITUDES_FT = (0, 50, 100, 200, 400)
FEET_TO_METERS = 0.3048
OUT_DIR = PROJECT_ROOT / "results" / "analysis_tables"


def edge_length_meters(link_3d_df: pd.DataFrame) -> dict[tuple[str, str], tuple[float, str, int | None]]:
    """Map each directed link to (length_m, link_type, cruise altitude or None)."""
    frame = link_3d_df.copy()
    frame["length"] = frame["length"].fillna(1e-3).replace(0, 1e-3)
    edges: dict[tuple[str, str], tuple[float, str, int | None]] = {}
    for row in frame.itertuples(index=False):
        kind = str(row.link_type)
        if kind == "vertical":
            length_m = float(row.length) * FEET_TO_METERS
            altitude = None
        else:
            length_m = float(row.length)
            altitude = int(float(row.altitude))
        hop = (length_m, kind, altitude)
        src, dst = str(row.from_node_id), str(row.to_node_id)
        edges[(src, dst)] = hop
        if int(row.dir_flag) == 1:
            edges[(dst, src)] = hop
    return edges


def install_predecessor_cache(store: dict[int, object]):
    """Record Dijkstra predecessors while the request generator asks for distances."""

    def dist_with_pred(src_node):
        src_node = str(src_node)
        src_idx = sp_mod.NODE_TO_IDX[src_node]
        cached = store.get(src_idx)
        if cached is not None:
            return cached[0]
        dist, pred = dijkstra(
            sp_mod.CSR,
            directed=False,
            indices=src_idx,
            return_predecessors=True,
        )
        store[src_idx] = (dist, pred)
        return dist

    sp_mod._dist_from_src = dist_with_pred
    if hasattr(sp_mod._dist_from_src, "cache_clear"):
        sp_mod._dist_from_src.cache_clear()


def recover_path(pred, src_idx: int, dst_idx: int) -> list[int] | None:
    if src_idx == dst_idx:
        return [src_idx]
    nodes = [dst_idx]
    cur = dst_idx
    for _ in range(len(pred)):
        if cur == src_idx:
            nodes.reverse()
            return nodes
        nxt = int(pred[cur])
        if nxt < 0:
            return None
        nodes.append(nxt)
        cur = nxt
    return None


def accumulate_path(node_ids: list[str], edges, totals: dict) -> bool:
    climbed = False
    for src, dst in zip(node_ids, node_ids[1:]):
        hop = edges.get((src, dst))
        if hop is None:
            hop = edges.get((dst, src))
        if hop is None:
            totals["missing_hops"] += 1
            continue
        length_m, kind, altitude = hop
        totals["total_m"] += length_m
        if kind == "vertical":
            totals["vertical_m"] += length_m
        else:
            totals["by_altitude_m"][altitude] = (
                totals["by_altitude_m"].get(altitude, 0.0) + length_m
            )
            if altitude != 0:
                climbed = True
    return climbed


def scenario_pairs(quick: bool):
    if quick:
        return [(5, 5, 0)]
    pairs = []
    for num_vehicles, request_list in R.SCENARIOS_BY_VEHICLES.items():
        for num_requests in request_list:
            for trial in range(R.DEFAULT_TRIALS):
                pairs.append((num_vehicles, num_requests, trial))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Altitude share of experimental shortest paths.")
    parser.add_argument("--city", default="32_Phoenix_City")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only the 5-vehicle, 5-request, trial-1 scenario.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    node_df = R.load_or_build_network(args.city)
    link_3d_df = pd.read_csv(R.LINK_3D_PATH, low_memory=False)
    edges = edge_length_meters(link_3d_df)
    idx_to_node = {idx: node_id for node_id, idx in sp_mod.NODE_TO_IDX.items()}
    node2d_to_3d = R.trip_builder.node2d_to_3d
    if not all(str(node_id).endswith("_z0") for node_id in node2d_to_3d.values()):
        raise SystemExit("Endpoint map is not ground level (_z0). Refusing to measure.")

    totals = {
        "by_altitude_m": {z: 0.0 for z in ALTITUDES_FT},
        "vertical_m": 0.0,
        "total_m": 0.0,
        "missing_hops": 0,
        "n_paths": 0,
        "n_climbed": 0,
    }
    pred_store: dict[int, object] = {}
    install_predecessor_cache(pred_store)

    pairs = scenario_pairs(args.quick)
    for done, (num_vehicles, num_requests, trial) in enumerate(pairs, start=1):
        pred_store.clear()
        seed = R.scenario_seed(num_vehicles, num_requests, 0, trial)
        requests, _vehicles = R._generate_requests_and_vehicles(
            num_requests=num_requests,
            num_vehicles=num_vehicles,
            node2d_to_3d=node2d_to_3d,
            shortest_path_cached=R.shortest_path_cached,
            node_df=node_df,
            seed=seed,
        )
        for request in requests:
            src = str(node2d_to_3d[request.origin])
            dst = str(node2d_to_3d[request.destination])
            src_idx = sp_mod.NODE_TO_IDX[src]
            dst_idx = sp_mod.NODE_TO_IDX[dst]
            cached = pred_store.get(src_idx)
            if cached is None:
                totals["missing_hops"] += 1
                continue
            path_idx = recover_path(cached[1], src_idx, dst_idx)
            if path_idx is None:
                totals["missing_hops"] += 1
                continue
            path_ids = [str(idx_to_node[i]) for i in path_idx]
            climbed = accumulate_path(path_ids, edges, totals)
            totals["n_paths"] += 1
            totals["n_climbed"] += int(climbed)
        print(
            f"[{done}/{len(pairs)}] v={num_vehicles} r={num_requests} trial={trial + 1} "
            f"paths={len(requests)}",
            flush=True,
        )

    total = totals["total_m"]
    rows = []
    for z in ALTITUDES_FT:
        length = totals["by_altitude_m"].get(z, 0.0)
        rows.append({
            "component": f"horizontal_{z}ft",
            "altitude_ft": z,
            "length_m": length,
            "share_of_total": (length / total) if total else 0.0,
        })
    rows.append({
        "component": "vertical_climb_and_descent",
        "altitude_ft": "",
        "length_m": totals["vertical_m"],
        "share_of_total": (totals["vertical_m"] / total) if total else 0.0,
    })
    horizontal = sum(totals["by_altitude_m"].values())
    summary = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "altitude_length_share.csv"
    summary.to_csv(csv_path, index=False)

    def pct(part: float) -> str:
        return f"{100.0 * part / total:.1f}%" if total else "n/a"

    horizontal_bits = ", ".join(
        f"{z} ft {pct(totals['by_altitude_m'].get(z, 0.0))}" for z in ALTITUDES_FT
    )
    climbed_pct = (
        100.0 * totals["n_climbed"] / totals["n_paths"] if totals["n_paths"] else 0.0
    )
    lines = [
        f"Paths: {totals['n_paths']} request origin-destination shortest paths.",
        f"Paths that leave the ground layer: {totals['n_climbed']} ({climbed_pct:.1f}%).",
        f"Total length: {total:.1f} m "
        f"(horizontal {horizontal:.1f} m, vertical {totals['vertical_m']:.1f} m).",
        f"Share of path length: {horizontal_bits}; "
        f"climb and descent {pct(totals['vertical_m'])}.",
        "Vertical length uses 0.3048 m per foot. Horizontal length is the GMNS link length.",
        f"Missing hops: {totals['missing_hops']}.",
    ]
    text = "\n".join(lines)
    txt_path = OUT_DIR / "altitude_length_share.txt"
    txt_path.write_text(text + "\n", encoding="utf-8")
    print()
    print(text)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
