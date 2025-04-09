import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.help_functions import encode_modes
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures, highway_mapping
from gnn.help_functions import compute_r2_torch, compute_r2_torch_with_mean_targets, compute_spearman_pearson

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

districts = gpd.read_file(os.path.join(project_root, "data", "visualisation", "districts_paris.geojson"))

# Professional color palette with good contrast and accessibility
colors = {
    'Trunk Roads': '#1f77b4',         # Muted blue
    'Primary Roads': '#2ca02c',       # Muted green
    'Secondary Roads': '#ff7f0e',     # Muted orange
    'Tertiary Roads': '#9467bd',      # Muted purple
    'Residential Streets': '#8c564b',  # Brown
    'Living Streets': '#e377c2',      # Pink
    'P/S/T Roads with Capacity Reduction': '#7f7f7f',    # Gray
    'P/S/T Roads with No Capacity Reduction': '#bcbd22', # Olive
}

# DATAFRAME FUNCTIONS

def data_to_geodataframe_with_og_values(data, original_gdf, predicted_values, inversed_x, use_all_features=False):
    ' use_all_features is a flag for whether to use all features or not, as shown in the ablation tests'
    target_values = data.y.cpu().numpy()
    predicted_values = predicted_values.cpu().numpy() if isinstance(predicted_values, torch.Tensor) else predicted_values
    
    if use_all_features:
        edge_data = {
        'from_node': original_gdf["from_node"].values,
        'to_node': original_gdf["to_node"].values,
        'vol_base_case': inversed_x[:, EdgeFeatures.VOL_BASE_CASE],  
        'capacity_base_case': inversed_x[:, EdgeFeatures.CAPACITY_BASE_CASE],
        'capacity_reduction': inversed_x[:, EdgeFeatures.CAPACITY_REDUCTION],  
        'freespeed': inversed_x[:, EdgeFeatures.FREESPEED],  
        'highway': original_gdf['highway'].values,
        'length': inversed_x[:, EdgeFeatures.LENGTH-1], # -1 since we didn't use Highway, fix later        
        'allowed_mode_car': inversed_x[:, EdgeFeatures.ALLOWED_MODE_CAR],
        'allowed_mode_bus': inversed_x[:, EdgeFeatures.ALLOWED_MODE_BUS],
        'allowed_mode_pt': inversed_x[:, EdgeFeatures.ALLOWED_MODE_PT],
        'allowed_mode_train': inversed_x[:, EdgeFeatures.ALLOWED_MODE_TRAIN],
        'allowed_mode_rail': inversed_x[:, EdgeFeatures.ALLOWED_MODE_RAIL],
        'allowed_mode_subway': inversed_x[:, EdgeFeatures.ALLOWED_MODE_SUBWAY],
        'vol_car_change_actual': target_values.squeeze(),  
        'vol_car_change_predicted': predicted_values.squeeze(),
        }
    else:
        edge_data = {
            'from_node': original_gdf["from_node"].values,
            'to_node': original_gdf["to_node"].values,
            'vol_base_case': inversed_x[:, EdgeFeatures.VOL_BASE_CASE],  
            'capacity_base_case': inversed_x[:, EdgeFeatures.CAPACITY_BASE_CASE],  
            'capacity_reduction': inversed_x[:, EdgeFeatures.CAPACITY_REDUCTION],  
            'freespeed': inversed_x[:, EdgeFeatures.FREESPEED],  
            'highway': original_gdf['highway'].values,
            'length': inversed_x[:, EdgeFeatures.LENGTH-1], # -1 since we didn't use Highway, fix later
            'vol_car_change_actual': target_values.squeeze(),
            'vol_car_change_predicted': predicted_values.squeeze(),
            'mean_car_vol': original_gdf['vol_car'].values,
            'variance_base_case': original_gdf['variance'].values,
            'std_dev': original_gdf['std_dev'].values,
            'std_dev_multiplied': original_gdf['std_dev_multiplied'].values,
            'cv_percent': original_gdf['cv_percent'].values,
        }
    
    edge_df = pd.DataFrame(edge_data)
    edge_df['geometry'] = original_gdf["geometry"].values
    gdf = gpd.GeoDataFrame(edge_df, geometry='geometry')
    return gdf

def create_test_object(links_base_case, test_data, stacked_edge_geometries_tensor):
    linegraph_transformation = LineGraph()
    vol_base_case = links_base_case['vol_car'].values
    capacity_base_case = np.where(links_base_case['modes'].str.contains('car'), links_base_case['capacity'], 0)

    length = links_base_case['length'].values
    freespeed = links_base_case['freespeed'].values
    allowed_modes = encode_modes(links_base_case)
    edges_base = links_base_case[['from_idx', 'to_idx']].values
    nodes = pd.concat([links_base_case['from_node'], links_base_case['to_node']]).unique()

    edge_index = torch.tensor(edges_base, dtype=torch.long).t().contiguous()
    x = torch.zeros((len(nodes), 1), dtype=torch.float)
    data = Data(edge_index=edge_index, x=x)
    
    capacities_new = test_data['capacity'].values
    capacity_reduction= capacities_new - capacity_base_case

    highway = test_data['highway'].apply(lambda x: highway_mapping.get(x, -1)).values
    
    edge_feature_dict = {
        EdgeFeatures.VOL_BASE_CASE: torch.tensor(vol_base_case),
        EdgeFeatures.CAPACITY_BASE_CASE: torch.tensor(capacity_base_case),
        EdgeFeatures.CAPACITY_REDUCTION: torch.tensor(capacity_reduction),
        EdgeFeatures.FREESPEED: torch.tensor(freespeed),
        EdgeFeatures.HIGHWAY: torch.tensor(highway),
        EdgeFeatures.LENGTH: torch.tensor(length),
        EdgeFeatures.ALLOWED_MODE_CAR: allowed_modes[0],
        EdgeFeatures.ALLOWED_MODE_BUS: allowed_modes[1],
        EdgeFeatures.ALLOWED_MODE_PT:  allowed_modes[2],
        EdgeFeatures.ALLOWED_MODE_TRAIN: allowed_modes[3],
        EdgeFeatures.ALLOWED_MODE_RAIL: allowed_modes[4],
        EdgeFeatures.ALLOWED_MODE_SUBWAY: allowed_modes[5],
    }
    
    edge_tensor = [edge_feature_dict[feature] for feature in EdgeFeatures]
    edge_tensor = torch.stack(edge_tensor, dim=1)  # Shape: (31140, 14)
    
    linegraph_data = linegraph_transformation(data)
    linegraph_data.x = edge_tensor
    linegraph_data.pos = stacked_edge_geometries_tensor
    edge_car_volume_difference = test_data['vol_car'].values - vol_base_case
    linegraph_data.y = torch.tensor(edge_car_volume_difference, dtype=torch.float).unsqueeze(1)
    
    if linegraph_data.validate(raise_on_error=True):
        return linegraph_data
    else:
        print("Invalid line graph data")

def get_road_type_indices(gdf, tolerance=1e-3):
    """
    Get indices for different road types, including dynamic conditions like capacity reduction
    """
    tolerance = 1e-3
    
    indices = {
        # Static conditions (road types)
        "Trunk Roads": gdf[gdf['highway'].isin([0])].index,
        "Primary Roads": gdf[gdf['highway'].isin([1])].index,
        "Secondary Roads": gdf[gdf['highway'].isin([2])].index,
        "Tertiary Roads": gdf[gdf['highway'].isin([3])].index,
        "Residential Streets": gdf[gdf['highway'].isin([4])].index,
        "Living Streets": gdf[gdf['highway'].isin([5])].index,
        "P/S/T Roads": gdf[gdf['highway'].isin([1, 2, 3])].index,
        
        # Dynamic conditions (capacity reduction)
        "Roads with Capacity Reduction": gdf[gdf['capacity_reduction_rounded'] < -tolerance].index,
        "Roads with No Capacity Reduction": gdf[gdf['capacity_reduction_rounded'] >= -tolerance].index,
        "P/S/T Roads with Capacity Reduction": gdf[(gdf['highway'].isin([1, 2, 3])) & (gdf['capacity_reduction_rounded'] < -tolerance)].index,
        "P/S/T Roads with No Capacity Reduction": gdf[(gdf['highway'].isin([1, 2, 3])) & (gdf['capacity_reduction_rounded'] >= -tolerance)].index,
        
        # Combined conditions
        "Primary Roads with Capacity Reduction": gdf[
            (gdf['highway'].isin([1])) & 
            (gdf['capacity_reduction_rounded'] < -tolerance)
        ].index,
        "Primary Roads with No Capacity Reduction": gdf[
            (gdf['highway'].isin([1])) & 
            (gdf['capacity_reduction_rounded'] >= -tolerance)
        ].index,
        "Secondary Roads with Capacity Reduction": gdf[
            (gdf['highway'].isin([2])) & 
            (gdf['capacity_reduction_rounded'] < -tolerance)
        ].index,
        "Secondary Roads with No Capacity Reduction": gdf[
            (gdf['highway'].isin([2])) & 
            (gdf['capacity_reduction_rounded'] >= -tolerance)
        ].index,
        "Tertiary Roads with Capacity Reduction": gdf[
            (gdf['highway'].isin([3])) & 
            (gdf['capacity_reduction_rounded'] < -tolerance)
        ].index,    
        "Tertiary Roads with No Capacity Reduction": gdf[
            (gdf['highway'].isin([3])) & 
            (gdf['capacity_reduction_rounded'] >= -tolerance)
        ].index
    }
    return indices

### NORMALIZATION FUNCTIONS ###

def normalize_tensor(tensor, scaler):
    """
    Normalizes a given tensor using the provided scaler.

    Parameters:
    tensor (torch.Tensor): The tensor to normalize.
    scaler (sklearn.preprocessing.StandardScaler or similar): The scaler to use for normalization.

    Returns:
    torch.Tensor: The normalized tensor.
    """
    # Convert the tensor to a numpy array, apply the scaler, and convert back to a tensor
    tensor_np = tensor.numpy()
    normalized_np = scaler.transform(tensor_np)
    normalized_tensor = torch.tensor(normalized_np, dtype=tensor.dtype)
    return normalized_tensor

def normalize_pos_features(tensor, scaler):
    """
    Normalizes the position features of a given tensor using the provided scaler.

    Parameters:
    tensor (torch.Tensor): The tensor to normalize, expected shape (31140, 3, 2).
    scaler (sklearn.preprocessing.StandardScaler or similar): The scaler to use for normalization.

    Returns:
    torch.Tensor: The normalized tensor.
    """
    # Reshape the tensor to (31140, 6) for normalization
    tensor_reshaped = tensor.view(-1, 6).numpy()
    normalized_np = scaler.transform(tensor_reshaped)
    # Reshape back to (31140, 3, 2) after normalization
    normalized_tensor = torch.tensor(normalized_np, dtype=tensor.dtype).view(31140, 3, 2)
    return normalized_tensor

def validate_model_on_test_set(model, dataset, loss_func, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.inference_mode():
        if isinstance(dataset, list):
            for data in dataset:
                input_node_features, targets = data.x.to(device), data.y.to(device)
                predicted = model(data.to(device))
                loss = loss_func(predicted, targets).item()
                total_loss += loss
                all_preds.append(predicted)
                all_targets.append(targets)
        else:
            input_node_features, targets = dataset.x.to(device), dataset.y.to(device)
            predicted = model(dataset.to(device))
            loss = loss_func(predicted, targets).item()
            total_loss += loss
            all_preds.append(predicted)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    mean_targets = torch.mean(all_targets)
    r_squared = compute_r2_torch_with_mean_targets(mean_targets=mean_targets, preds=all_preds, targets=all_targets)
    baseline_loss = loss_func(all_targets, torch.full_like(all_preds, mean_targets))
    avg_loss = total_loss / len(dataset)
    return avg_loss, r_squared, all_targets, all_preds, baseline_loss

def calculate_error_distribution_metric(actual_changes, predicted_changes):
    sigma = np.std(actual_changes)
    errors = np.abs(predicted_changes - actual_changes)
    within_sigma = np.sum(errors <= sigma)
    return (within_sigma / len(errors)) * 100

def validate_model_with_interpretable_error(indices, gdf, loss_fct, tolerance):
    loss_fct_l1 = torch.nn.L1Loss()
    base_car_vol = gdf.loc[indices, 'vol_base_case']
    actual_vals = gdf.loc[indices, 'vol_car_change_actual']
    predicted_vals = gdf.loc[indices, 'vol_car_change_predicted']
    
    actual_vals = actual_vals.to_numpy()
    predicted_vals = predicted_vals.to_numpy()
    actual_mean = torch.mean(torch.tensor(actual_vals))
    
    baseline_vals = torch.full_like(torch.tensor(actual_vals), actual_mean)
    r_squared = compute_r2_torch(preds=torch.tensor(predicted_vals), targets=torch.tensor(actual_vals))
    r_squared = round(r_squared.item(), 2)
    
    baseline_vals_np = baseline_vals.numpy()
    base_car_vol_np = base_car_vol.to_numpy()
    
    baseline_car_vol = base_car_vol_np + baseline_vals_np
    actual_car_vol = base_car_vol + actual_vals
    predicted_car_vol = base_car_vol + predicted_vals
    
    mse_loss = loss_fct(torch.tensor(actual_vals), torch.tensor(predicted_vals))
    l1_loss = loss_fct_l1(torch.tensor(actual_vals), torch.tensor(predicted_vals))

    baseline_mse = loss_fct(torch.tensor(actual_vals), torch.full_like(torch.tensor(actual_vals), actual_mean))
    baseline_l1 = loss_fct_l1(torch.tensor(actual_vals), torch.full_like(torch.tensor(actual_vals), actual_mean))
    
    error_normalized_by_mean_squared = mse_loss /   torch.mean(torch.tensor(actual_vals)).pow(2)
    baseline_normalized_by_mean_squared = baseline_mse / torch.mean(torch.tensor(actual_vals)).pow(2)
    
    variance_actual_car_vol = torch.var(torch.tensor(actual_vals))
    error_normalized_by_variance = mse_loss / variance_actual_car_vol
    baseline_normalized_by_variance = baseline_mse / variance_actual_car_vol
    
    spearman_corr, pearson_corr = compute_spearman_pearson(torch.tensor(actual_vals), torch.tensor(predicted_vals))
    
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    mean_relative_errors, mean_filtered_relative_errors = compute_relative_error_and_relative_filtered_error(actual_car_vol, predicted_car_vol, tolerance)
    mean_relative_errors_baseline, mean_filtered_relative_errors_baseline = compute_relative_error_and_relative_filtered_error(actual_car_vol, baseline_car_vol, tolerance)
    
    print(f"R-squared: {r_squared}")
    print(f"MSE Loss: {mse_loss}")
    print(f"Baseline Loss: {baseline_mse}")
    print(f"L1 Loss: {l1_loss}")
    print(f"Baseline L1 loss: {baseline_l1}")
    print(f"Error normalized by mean squared: {error_normalized_by_mean_squared:.4f}")
    print(f"Baseline normalized by mean squared: {baseline_normalized_by_mean_squared:.4f}")
    print(f"Error normalized by variance: {error_normalized_by_variance:.4f}")
    print(f"Baseline normalized by variance: {baseline_normalized_by_variance:.4f}")
    print(f"Mean Relative Error: {mean_relative_errors:.4f}")
    print(f"Mean Filtered Relative Error: {mean_filtered_relative_errors:.4f}")
    print(f"Baseline Mean Relative Error: {mean_relative_errors_baseline:.4f}")
    print(f"Baseline Mean Filtered Relative Error: {mean_filtered_relative_errors_baseline:.4f}")
    print(" ")
    
    return

def compute_relative_error_and_relative_filtered_error(actual_car_vol, car_vol_to_compare, tolerance):
    actual_car_vol[actual_car_vol == 0] = 1e-10
    absolute_errors = torch.abs(torch.tensor(car_vol_to_compare) - torch.tensor(actual_car_vol))
    relative_errors = absolute_errors / torch.tensor(actual_car_vol)
    filtered_relative_errors = relative_errors[(relative_errors <= tolerance) & (relative_errors >= -tolerance)]
    mean_relative_errors = torch.mean(relative_errors)
    mean_filtered_relative_errors = torch.mean(filtered_relative_errors)
    return mean_relative_errors,mean_filtered_relative_errors