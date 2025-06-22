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

from data_preprocessing.help_functions import encode_modes, highway_mapping
from data_preprocessing.process_simulations_for_gnn import EdgeFeatures
from gnn.help_functions import compute_r2_torch_with_mean_targets

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
districts = gpd.read_file(os.path.join(project_root, "data", "visualisation", "districts_paris.geojson"))

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
        "All Roads": gdf.index,

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