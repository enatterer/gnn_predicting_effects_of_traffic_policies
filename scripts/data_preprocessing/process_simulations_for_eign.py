"""
Process simulation data (from MATSim) for GNNs. Load basecase and simulated graphs (with policies applied in various district combinations),
convert them to dual line graphs, and compute specified edge features. Save as PyTorch Geometric data batches for efficient loading and training.

Here we specify all features, then run_models can be called with a reduced set. Note that, for example, the flag "use_allowed_modes" is accessed from the run_models script.
"""

import os
import sys
from enum import IntEnum

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import torch
from torch_geometric.data import Data

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.help_functions import *
from data_preprocessing.process_simulations_for_gnn import compute_result_dic

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths to raw simulation data
sim_input_paths = [os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_1000'),
                   os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_2000'),
                   os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_3000'),
                   os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_4000'),
                   os.path.join(project_root, 'data', 'raw_data', 'exp_dist_not_connected_5k', 'output_networks_5000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_1000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_2000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_3000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_4000'),
                   os.path.join(project_root, 'data', 'raw_data', 'norm_dist_not_connected_5k', 'output_networks_5000')]

# Path to save the processed simulation data
result_path = os.path.join(
    project_root, "data", "train_data", "edge_features_with_net_flow"
)

# Path to the basecase links and stats
basecase_links_path = os.path.join(project_root, 'data', 'links_and_stats', 'pop_1pct_basecase_average_output_links.geojson')
basecase_stats_path = os.path.join(project_root, 'data', 'links_and_stats', 'pop_1pct_basecase_average_mode_stats.csv')

# Flag to use allowed modes or not
use_allowed_modes = False

class EdgeFeatures(IntEnum):
    VOL_BASE_CASE = 0
    CAPACITY_BASE_CASE = 1
    CAPACITY_REDUCTION = 2
    FREESPEED = 3
    HIGHWAY = 4
    LENGTH = 5
    ALLOWED_MODE_CAR = 6
    ALLOWED_MODE_BUS = 7
    ALLOWED_MODE_PT = 8
    ALLOWED_MODE_TRAIN = 9
    ALLOWED_MODE_RAIL = 10
    ALLOWED_MODE_SUBWAY = 11
    NET_FLOW = 12


def create_aggregated_edges_for_eign(links_gdf):
    """
    Create aggregated edges for EIGN where each unique node pair (u,v) has only one edge.
    For undirected pairs, we aggregate the features. For directed edges, we keep them as is.

    Returns:
    - edge_index: (2, num_edges) tensor with unique edges
    - edge_features: dict with aggregated features for each edge
    - edge_is_directed: boolean tensor indicating if each edge is directed
    - edge_positions: positions for each aggregated edge
    """
    # Create node mapping
    nodes = pd.concat([links_gdf["from_node"], links_gdf["to_node"]]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Create edge data with indices
    links_gdf = links_gdf.copy()
    links_gdf["from_idx"] = links_gdf["from_node"].map(node_to_idx)
    links_gdf["to_idx"] = links_gdf["to_node"].map(node_to_idx)

    # Group by edge pairs to identify undirected vs directed edges
    edge_groups = {}
    edge_positions = {}

    for idx, row in links_gdf.iterrows():
        u, v = row["from_idx"], row["to_idx"]

        # Create canonical edge representation (smaller node first)
        canonical_edge = tuple(sorted([u, v]))

        if canonical_edge not in edge_groups:
            edge_groups[canonical_edge] = []
            edge_positions[canonical_edge] = []

        edge_groups[canonical_edge].append(
            {
                "direction": (u, v),
                "vol_car": row["vol_car"] if pd.notna(row["vol_car"]) else 0.0,
                "capacity": (
                    row["capacity"]
                    if pd.notna(row["capacity"]) and "car" in str(row.get("modes", ""))
                    else 0.0
                ),
                "length": row["length"] if pd.notna(row["length"]) else 0.0,
                "freespeed": row["freespeed"] if pd.notna(row["freespeed"]) else 0.0,
                "highway": row["highway"] if pd.notna(row["highway"]) else "unknown",
                "modes": row["modes"] if pd.notna(row["modes"]) else "",
            }
        )

    # Process aggregated edges
    aggregated_edges = []
    aggregated_features = {
        "vol_base_case": [],
        "capacity_base_case": [],
        "length": [],
        "freespeed": [],
        "highway": [],
        "net_flow": [],
        "allowed_modes": [],
    }
    aggregated_positions = []
    edge_is_directed = []

    for canonical_edge, edge_data_list in edge_groups.items():
        u_canonical, v_canonical = canonical_edge

        # Determine if this is a directed or undirected edge
        directions = [data["direction"] for data in edge_data_list]
        has_both_directions = (u_canonical, v_canonical) in directions and (
            v_canonical,
            u_canonical,
        ) in directions

        if has_both_directions:
            # Undirected edge - aggregate features
            edge_is_directed.append(False)

            # Find data for both directions
            forward_data = next(
                data
                for data in edge_data_list
                if data["direction"] == (u_canonical, v_canonical)
            )
            backward_data = next(
                data
                for data in edge_data_list
                if data["direction"] == (v_canonical, u_canonical)
            )

            # Aggregate features
            vol_forward = (
                forward_data["vol_car"] if forward_data["vol_car"] is not None else 0.0
            )
            vol_backward = (
                backward_data["vol_car"]
                if backward_data["vol_car"] is not None
                else 0.0
            )
            net_flow = (
                vol_forward - vol_backward
            )  # Net flow in u_canonical -> v_canonical direction

            # For other features, take average or sum as appropriate
            # Handle potential None/NaN values before division
            forward_capacity = (
                forward_data["capacity"]
                if forward_data["capacity"] is not None
                else 0.0
            )
            backward_capacity = (
                backward_data["capacity"]
                if backward_data["capacity"] is not None
                else 0.0
            )
            capacity = (forward_capacity + backward_capacity) / 2

            forward_length = (
                forward_data["length"] if forward_data["length"] is not None else 0.0
            )
            backward_length = (
                backward_data["length"] if backward_data["length"] is not None else 0.0
            )
            length = (forward_length + backward_length) / 2

            forward_freespeed = (
                forward_data["freespeed"]
                if forward_data["freespeed"] is not None
                else 0.0
            )
            backward_freespeed = (
                backward_data["freespeed"]
                if backward_data["freespeed"] is not None
                else 0.0
            )
            freespeed = (forward_freespeed + backward_freespeed) / 2

            # For highway type, take the more significant one (higher numeric value)
            highway_forward = highway_mapping.get(forward_data["highway"], -1)
            highway_backward = highway_mapping.get(backward_data["highway"], -1)
            highway = max(highway_forward, highway_backward)

            # For modes, combine them
            modes_combined = (
                str(forward_data["modes"]) + "," + str(backward_data["modes"])
            )

        else:
            # Directed edge
            edge_is_directed.append(True)
            edge_data = edge_data_list[0]  # Only one direction exists

            vol_forward = (
                edge_data["vol_car"] if edge_data["vol_car"] is not None else 0.0
            )
            net_flow = vol_forward  # No backward flow for directed edges

            # Handle potential None/NaN values for directed edges
            capacity = (
                edge_data["capacity"] if edge_data["capacity"] is not None else 0.0
            )
            length = edge_data["length"] if edge_data["length"] is not None else 0.0
            freespeed = (
                edge_data["freespeed"] if edge_data["freespeed"] is not None else 0.0
            )
            highway = highway_mapping.get(edge_data["highway"], -1)
            modes_combined = (
                str(edge_data["modes"]) if edge_data["modes"] is not None else ""
            )

            # Ensure edge orientation matches direction for directed edges
            if edge_data["direction"] != (u_canonical, v_canonical):
                # Flip the canonical edge to match the actual direction
                u_canonical, v_canonical = v_canonical, u_canonical
                net_flow = -net_flow  # Flip net flow to match new orientation

        # Add to aggregated data
        aggregated_edges.append([u_canonical, v_canonical])
        aggregated_features["vol_base_case"].append(
            abs(vol_forward) if has_both_directions else vol_forward
        )
        aggregated_features["capacity_base_case"].append(capacity)
        aggregated_features["length"].append(length)
        aggregated_features["freespeed"].append(freespeed)
        aggregated_features["highway"].append(highway)
        aggregated_features["net_flow"].append(net_flow)

        # Encode allowed modes for aggregated edge
        allowed_modes = encode_modes_from_string(modes_combined)
        aggregated_features["allowed_modes"].append(allowed_modes)

    # Convert to tensors
    edge_index = torch.tensor(aggregated_edges, dtype=torch.long).t().contiguous()
    edge_is_directed = torch.tensor(edge_is_directed, dtype=torch.bool)

    # Convert features to tensors
    for key in aggregated_features:
        if key == "allowed_modes":
            # Handle allowed modes separately as it's a list of lists
            aggregated_features[key] = torch.stack(
                [torch.tensor(modes) for modes in aggregated_features[key]]
            )
        else:
            aggregated_features[key] = torch.tensor(
                aggregated_features[key], dtype=torch.float
            )

    return edge_index, aggregated_features, edge_is_directed


def encode_modes_from_string(modes_str):
    """Helper function to encode allowed modes from a string"""
    allowed_modes = [0, 0, 0, 0, 0, 0]  # car, bus, pt, train, rail, subway

    modes_lower = modes_str.lower()
    if "car" in modes_lower:
        allowed_modes[0] = 1
    if "bus" in modes_lower:
        allowed_modes[1] = 1
    if "pt" in modes_lower:
        allowed_modes[2] = 1
    if "train" in modes_lower:
        allowed_modes[3] = 1
    if "rail" in modes_lower:
        allowed_modes[4] = 1
    if "subway" in modes_lower:
        allowed_modes[5] = 1

    return allowed_modes


def process_result_dic_eign(
    result_dic,
    result_dic_mode_stats,
    save_path=None,
    batch_size=500,
    links_base_case=None,
    gdf_basecase_mean_mode_stats=None,
):

    # Create aggregated edges for EIGN - only one edge per node pair
    aggregated_data = create_aggregated_edges_for_eign(links_base_case)
    edge_index, base_features, edge_is_directed_base = (
        aggregated_data
    )
    _, stacked_edge_geometries_tensor, _, _ = get_link_geometries(links_base_case)

    os.makedirs(save_path, exist_ok=True)
    datalist = []

    batch_counter = 0
    for key, df in tqdm(
        result_dic.items(), desc="Processing result_dic", unit="dataframe"
    ):
        if isinstance(df, pd.DataFrame) and key != "base_network_no_policies":
            # Create aggregated edges for the current simulation data
            sim_aggregated_data = create_aggregated_edges_for_eign(df)
            sim_edge_index, sim_features, _ = sim_aggregated_data

            # Ensure consistent edge ordering between base case and simulation
            if not torch.equal(edge_index, sim_edge_index):
                print(f"Warning: Edge indices don't match for {key}. Skipping.")
                continue

            # Calculate signed features (net flow changes)
            net_flow_base = base_features["net_flow"]
            net_flow_sim = sim_features["net_flow"]
            net_flow_change = net_flow_sim - net_flow_base

            # Calculate unsigned features changes
            vol_change = sim_features["vol_base_case"] - base_features["vol_base_case"]
            capacity_change = (
                sim_features["capacity_base_case"] - base_features["capacity_base_case"]
            )

            # Prepare unsigned features (invariant)
            edge_feature_dict = {
                EdgeFeatures.VOL_BASE_CASE: base_features["vol_base_case"],
                EdgeFeatures.CAPACITY_BASE_CASE: base_features["capacity_base_case"],
                EdgeFeatures.CAPACITY_REDUCTION: capacity_change,
                EdgeFeatures.FREESPEED: base_features["freespeed"],
                EdgeFeatures.HIGHWAY: base_features["highway"],
                EdgeFeatures.LENGTH: base_features["length"],
            }

            # Prepare signed features (equivariant)
            edge_feature_dict_signed = {
                EdgeFeatures.NET_FLOW: net_flow_base,
            }

            if use_allowed_modes:
                allowed_modes_tensor = base_features["allowed_modes"]
                edge_feature_dict.update(
                    {
                        EdgeFeatures.ALLOWED_MODE_CAR: allowed_modes_tensor[:, 0],
                        EdgeFeatures.ALLOWED_MODE_BUS: allowed_modes_tensor[:, 1],
                        EdgeFeatures.ALLOWED_MODE_PT: allowed_modes_tensor[:, 2],
                        EdgeFeatures.ALLOWED_MODE_TRAIN: allowed_modes_tensor[:, 3],
                        EdgeFeatures.ALLOWED_MODE_RAIL: allowed_modes_tensor[:, 4],
                        EdgeFeatures.ALLOWED_MODE_SUBWAY: allowed_modes_tensor[:, 5],
                    }
                )

            # Create the edge tensors by iterating through the EdgeFeatures enum
            edge_tensor = [
                edge_feature_dict[feature]
                for feature in EdgeFeatures
                if feature in edge_feature_dict
            ]

            edge_tensor_signed = [
                edge_feature_dict_signed[feature]
                for feature in EdgeFeatures
                if feature in edge_feature_dict_signed
            ]

            # Stack the tensors
            edge_tensor = torch.stack(edge_tensor, dim=1)
            edge_tensor_signed = torch.stack(edge_tensor_signed, dim=1)

            # Create data object
            data = Data(edge_index=edge_index)
            data.num_nodes = edge_index.shape[1]
            data.x = edge_tensor
            data.x_signed = edge_tensor_signed
            data.pos = stacked_edge_geometries_tensor
            data.y = vol_change.unsqueeze(1)  # Volume change as target
            data.y_signed = net_flow_change.unsqueeze(
                1
            )  # Net flow change as signed target
            data.edge_is_directed = edge_is_directed_base

            df_mode_stats = result_dic_mode_stats.get(key)
            if df_mode_stats is not None:
                pd.set_option("display.float_format", lambda x: "%.10f" % x)
                numeric_cols_base_case = gdf_basecase_mean_mode_stats.select_dtypes(
                    include=[np.number]
                ).columns
                numeric_cols = df_mode_stats.select_dtypes(include=[np.number]).columns
                mode_stats_diff = (
                    df_mode_stats[numeric_cols].values
                    - gdf_basecase_mean_mode_stats[numeric_cols_base_case].values
                )
                mode_stats_tensor = torch.tensor(mode_stats_diff, dtype=torch.float)
                data.mode_stats_diff = mode_stats_tensor
                mode_stats_diff_perc = (
                    mode_stats_tensor
                    / gdf_basecase_mean_mode_stats[numeric_cols_base_case].values
                    * 100
                )
                data.mode_stats_diff_perc = mode_stats_diff_perc

            if validate_data_eign(data):
                datalist.append(data)
                batch_counter += 1

                # Save intermediate result every batch_size data points
                if batch_counter % batch_size == 0:
                    batch_index = batch_counter // batch_size
                    torch.save(
                        datalist,
                        os.path.join(save_path, f"datalist_batch_{batch_index}.pt"),
                    )
                    print(os.path.join(save_path, f"datalist_batch_{batch_index}.pt"))
                    datalist = []
            else:
                print("Invalid EIGNgraph data")
                return

    # Save any remaining data points
    if datalist:
        batch_index = (batch_counter // batch_size) + 1
        torch.save(
            datalist, os.path.join(save_path, f"datalist_batch_{batch_index}.pt")
        )
        print(os.path.join(save_path, f"datalist_batch_{batch_index}.pt"))


def add_edge_is_directed(data: Data) -> Data:
    edge_index = data.edge_index
    src, dst = edge_index
    edges = torch.stack([src, dst], dim=1)

    # Create a set of reversed edges
    reversed_edges_set = {
        tuple(edge.tolist()) for edge in torch.stack([dst, src], dim=1)
    }

    # Mark each edge as directed if its reverse is not in the set
    is_directed = torch.tensor(
        [tuple(edge.tolist()) not in reversed_edges_set for edge in edges],
        dtype=torch.bool,
        device=edge_index.device,
    )

    return is_directed


def validate_data_eign(data: Data) -> bool:
    """Validation function for EIGN data with aggregated edges"""
    num_edges = data.edge_index.shape[1]

    # Check x (unsigned features)
    expected_unsigned_features = 6 if not use_allowed_modes else 12
    if data.x.shape != (num_edges, expected_unsigned_features):
        print(
            f"Invalid shape for data.x: expected ({num_edges}, {expected_unsigned_features}), got {data.x.shape}"
        )
        return False

    # Check x_signed (signed features)
    if data.x_signed.shape != (num_edges, 1):
        print(
            f"Invalid shape for data.x_signed: expected ({num_edges}, 1), got {data.x_signed.shape}"
        )
        return False

    # Check y (unsigned target)
    if data.y.shape != (num_edges, 1):
        print(
            f"Invalid shape for data.y: expected ({num_edges}, 1), got {data.y.shape}"
        )
        return False

    # Check y_signed (signed target)
    if data.y_signed.shape != (num_edges, 1):
        print(
            f"Invalid shape for data.y_signed: expected ({num_edges}, 1), got {data.y_signed.shape}"
        )
        return False

    # Check edge_index
    if data.edge_index.shape != (2, num_edges):
        print(
            f"Invalid shape for data.edge_index: expected (2, {num_edges}), got {data.edge_index.shape}"
        )
        return False

    # Check edge_is_directed
    if data.edge_is_directed.shape != (num_edges,):
        print(
            f"Invalid shape for data.edge_is_directed: expected ({num_edges},), got {data.edge_is_directed.shape}"
        )
        return False

    return True


def main():

    networks = list()

    for path in sim_input_paths:
        networks += [os.path.join(path, network) for network in os.listdir(path)]

    networks = [
        network
        for network in networks
        if os.path.isdir(network) and not network.endswith(".DS_Store")
    ]
    networks.sort()

    gdf_basecase_links = gpd.read_file(basecase_links_path)
    gdf_basecase_links = gdf_basecase_links.set_crs("EPSG:4326", allow_override=True)
    gdf_basecase_mean_mode_stats = pd.read_csv(basecase_stats_path, delimiter=",")

    result_dic_output_links, result_dic_eqasim_trips = compute_result_dic(
        basecase_links=gdf_basecase_links, networks=networks
    )
    base_gdf = result_dic_output_links["base_network_no_policies"]

    gdf_basecase_mean_mode_stats.rename(
        columns={
            "avg_total_travel_time": "total_travel_time",
            "avg_total_routed_distance": "total_routed_distance",
            "avg_trip_count": "trip_count",
        },
        inplace=True,
    )
    process_result_dic_eign(
        result_dic=result_dic_output_links,
        result_dic_mode_stats=result_dic_eqasim_trips,
        save_path=result_path,
        batch_size=50,
        links_base_case=base_gdf,
        gdf_basecase_mean_mode_stats=gdf_basecase_mean_mode_stats,
    )


if __name__ == "__main__":
    main()
