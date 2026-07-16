import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum_uav_routing.rtv.load_network import clone_dataset_repo, parse_graph, create_graph, REPO_DIR, convert_to_3d

network_name = "32_Phoenix_City"
skip_zones = False

clone_dataset_repo()
node_3d_df, link_3d_df, node2d_to_3d = convert_to_3d(network_name, "data/GMNS_Plus_Dataset/32_Phoenix_City")