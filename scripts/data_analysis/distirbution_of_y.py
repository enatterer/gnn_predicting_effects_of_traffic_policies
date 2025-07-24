
import os
import sys
import json
from enum import IntEnum
import shutil
from collections import Counter

import sys
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch_geometric.data import Data

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from data_preprocessing.help_functions import *
from data_preprocessing.process_simulations_for_gnn import *



# Control variables (matching the main processing script)
batch_size = 128
seed = 3
hex_sizes = [500]
required_modes_on_links = ['car', 'car_passenger']
use_destination_activity = False  # Simplified - no activity features needed
all_cities = ['muenchen']
target_feature = 'vol_car'  # Target feature for Y values (options: 'vol_car', 'vol_car_percentage')
target_feature_normalization_type = 'None'  # Normalization type for Y values (options: 'None', 'signed_log_normalization', 'mean_std', 'min_max')

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def compute_target_tensor_only_edge_features_for_distribution(vol_base_case, gdf, column_name: str, normalization_type: str):
    edge_car_volume_difference = gdf['vol_car'].values - vol_base_case
    
    if column_name == 'vol_car':
        # Keep continuous values for training - round only at inference
        # Sign preserved: +500.3 cars vs -200.7 cars
        data_to_normalize = edge_car_volume_difference
        
    elif column_name == 'vol_car_percentage':
        # Division by base case to get percentage change  
        # Sign preserved: +50% vs -30%
        epsilon = 1e-6 #adjust as needed
        base_case_with_epsilon = vol_base_case.copy()
        zero_mask = vol_base_case == 0
        base_case_with_epsilon[zero_mask] = epsilon
        data_to_normalize = edge_car_volume_difference / base_case_with_epsilon
        
    if normalization_type == 'None':
        return data_to_normalize
    else:# Use signed_log_normalized for both - preserves sign, compresses range
        normalized_data = normalization_of_edge_features(
        data_to_normalize, normalization_type
        )
        return normalized_data

def extract_y_and_capacity_reduction(city, result_dic, links_base_case):
    """Extract only Y values and capacity reduction for distribution analysis"""
    
    features_data = []
    vol_base_case = links_base_case['vol_car'].values
    capacity_base_case = get_capacity_base_case(links_base_case, required_modes_on_links)

    # Filter out base_network_no_policies before the loop
    graph_items = {k: v for k, v in result_dic.items() 
                   if isinstance(v, pd.DataFrame) and k != "base_network_no_policies"}
    
    for key, df in graph_items.items():
        gdf = prepare_gdf(df, links_base_case) 
        _, capacity_reduction, _, _ = get_basic_edge_attributes(capacity_base_case, gdf, required_modes_on_links)
        policy_region, scenario = key

        # Only compute Y values and capacity reduction
        y_values = compute_target_tensor_only_edge_features_for_distribution(vol_base_case, gdf, target_feature, target_feature_normalization_type)
        
        features_data.append({
            'y_values': y_values,
            'capacity_reduction': capacity_reduction,
            'policy_region': policy_region,
            'scenario': scenario,
            'city': city
        })
    
    return features_data
    
def plot_y_and_capacity_distributions(features_data, city_name, save_path):
    """Plot distributions of Y values and capacity reduction"""
    
    # Extract Y values and capacity reduction
    y_values_flat = []
    capacity_reduction_flat = []
    
    for data in features_data:
        # Y values
        if isinstance(data['y_values'], torch.Tensor):
            y_values_flat.extend(data['y_values'].cpu().numpy().flatten())
        else:
            y_values_flat.extend(np.array(data['y_values']).flatten())
            
        # Capacity reduction
        capacity_reduction_flat.extend(np.array(data['capacity_reduction']).flatten())
    
    y_values_flat = np.array(y_values_flat)
    capacity_reduction_flat = np.array(capacity_reduction_flat)
    
    # Filter out zero Y values for histogram
    y_values_nonzero = y_values_flat[y_values_flat != 0]
    capacity_reduction_nonzero = capacity_reduction_flat[capacity_reduction_flat != 0]
    
    # Create 2x2 plot: Y values on top row, capacity reduction on bottom row
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    fig.suptitle(f'Y Values & Capacity Reduction Distributions - {city_name}', fontsize=16)
    
    # Y VALUES ROW
    # 1. Y Values Histogram (NON-ZERO ONLY)
    if len(y_values_nonzero) > 0:
        # Create bin edges for ranges like -100 to -95, -95 to -90, etc.
        bin_edges = np.arange(-100, 105, 1)  # From -100 to 100 in steps of 5
        # This creates bins: [-100,-95], [-95,-90], [-90,-85], ..., [95,100]

        axes[0,0].hist(y_values_nonzero, bins=bin_edges, alpha=0.7, edgecolor='black', color='blue')
        axes[0,0].set_title(f'Y Values Histogram - Non-Zero Only ({target_feature})')
        axes[0,0].set_xlim(-100,100)
    else:
        axes[0,0].text(0.5, 0.5, 'No non-zero Y values', ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title(f'Y Values Histogram - No Data ({target_feature})')
    axes[0,0].set_xlabel('Y Value (excluding zeros)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Y Values Box plot (ALL VALUES INCLUDING ZEROS)
    if len(y_values_flat) > 0:
        axes[0,1].boxplot(y_values_flat)
        axes[0,1].set_title('Y Values Box Plot (All Values)')
    else:
        axes[0,1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Y Values Box Plot - No Data')
    axes[0,1].set_ylabel('Y Value')
    axes[0,1].grid(True, alpha=0.3)
    
    if len(y_values_nonzero) > 0:
        axes[0,2].boxplot(y_values_nonzero)
        axes[0,2].set_title('Y Values Box Plot (Non-Zero Only)')
    else:
        axes[0,2].text(0.5, 0.5, 'No non-zero Y values', ha='center', va='center', transform=axes[0,2].transAxes)
        axes[0,2].set_title('Y Values Box Plot - No Data')
    axes[0,2].set_ylabel('Y Value')
    axes[0,2].grid(True, alpha=0.3)
    
    # CAPACITY REDUCTION ROW
    # 3. Capacity Reduction Histogram
    if len(capacity_reduction_flat) > 0:
        axes[1,0].hist(capacity_reduction_nonzero, bins=100, alpha=0.7, edgecolor='black', color='red') 
        axes[1,0].set_title('Capacity Reduction Histogram for non-zero values')
        axes[1,0].set_xlim(-4000,0)
    else:
        axes[1,0].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Capacity Reduction Histogram - No Data')
    axes[1,0].set_xlabel('Capacity Reduction')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Capacity Reduction Box plot
    if len(capacity_reduction_flat) > 0:
        axes[1,1].boxplot(capacity_reduction_flat)
        axes[1,1].set_title('Capacity Reduction Box Plot (All Values)')
    else:
        axes[1,1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Capacity Reduction Box Plot - No Data')
    axes[1,1].set_ylabel('Capacity Reduction')
    axes[1,1].grid(True, alpha=0.3)
    
    if len(capacity_reduction_nonzero) > 0:
        axes[1,2].boxplot(capacity_reduction_nonzero)
        axes[1,2].set_title('Capacity Reduction Box Plot (Non-Zero Only)')
    else:
        axes[1,2].text(0.5, 0.5, 'No non-zero Y values', ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('Capacity Reduction Box Plot - No Data')
    axes[1,2].set_ylabel('Capacity Reduction')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, f'y_{target_feature}_and_capacity_distribution_{city_name}_{target_feature_normalization_type}_nonzero.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {plot_path}")
    
    with open(os.path.join(save_path, f'y_{target_feature}_and_capacity_distribution_{city_name}_{target_feature_normalization_type}_nonzero.txt'), 'w') as f:
        if len(y_values_flat) > 0:
            f.write(f"Y VALUE STATISTICS (ALL VALUES) ({target_feature}) - {city_name}\n")
            f.write(f"Total samples: {len(y_values_flat):,}\n")
            f.write(f"Mean: {np.mean(y_values_flat):.6f}\n")
            f.write(f"Std: {np.std(y_values_flat):.6f}\n")
            f.write(f"Min: {np.min(y_values_flat):.6f}\n")
            f.write(f"Max: {np.max(y_values_flat):.6f}\n")
            f.write(f"Median: {np.median(y_values_flat):.6f}\n")
            
            # Y values sign distribution
            y_positive = np.sum(y_values_flat > 0)
            y_negative = np.sum(y_values_flat < 0)
            y_zero = np.sum(y_values_flat == 0)
            
            f.write(f"Positive (increase): {y_positive:,} ({y_positive/len(y_values_flat)*100:.1f}%)\n")
            f.write(f"Negative (decrease): {y_negative:,} ({y_negative/len(y_values_flat)*100:.1f}%)\n")
            f.write(f"Zero (no change): {y_zero:,} ({y_zero/len(y_values_flat)*100:.1f}%)\n")
            
            if len(y_values_nonzero) > 0:
                f.write(f"\n=== NON-ZERO Y VALUE STATISTICS ({target_feature}) - {city_name} ===")
                f.write(f"Non-zero samples: {len(y_values_nonzero):,}\n")
                f.write(f"Mean (non-zero): {np.mean(y_values_nonzero):.6f}\n")
                f.write(f"Std (non-zero): {np.std(y_values_nonzero):.6f}\n")
                f.write(f"Min (non-zero): {np.min(y_values_nonzero):.6f}\n")
                f.write(f"Max (non-zero): {np.max(y_values_nonzero):.6f}\n")
                f.write(f"Median (non-zero): {np.median(y_values_nonzero):.6f}\n")
            else:
                f.write(f"\n=== NO NON-ZERO Y VALUES FOUND ===")
        else:
            f.write(f"\n=== NO Y VALUE DATA FOUND - {city_name} ===")
            
        if len(capacity_reduction_flat) > 0:
            f.write(f"\n=== CAPACITY REDUCTION STATISTICS - {city_name} ===")
            f.write(f"Total samples: {len(capacity_reduction_flat):,}\n")
            f.write(f"Mean: {np.mean(capacity_reduction_flat):.6f}\n")
            f.write(f"Std: {np.std(capacity_reduction_flat):.6f}\n")
            f.write(f"Min: {np.min(capacity_reduction_flat):.6f}\n")
            f.write(f"Max: {np.max(capacity_reduction_flat):.6f}\n")
            f.write(f"Median: {np.median(capacity_reduction_flat):.6f}\n")
            
            if len(capacity_reduction_nonzero) > 0:
                f.write(f"\n=== NON-ZERO CAPACITY REDUCTION STATISTICS - {city_name} ===")
                f.write(f"Non-zero samples: {len(capacity_reduction_nonzero):,}\n")
                f.write(f"Mean (non-zero): {np.mean(capacity_reduction_nonzero):.6f}\n")
                f.write(f"Std (non-zero): {np.std(capacity_reduction_nonzero):.6f}\n")
                f.write(f"Min (non-zero): {np.min(capacity_reduction_nonzero):.6f}\n")
                f.write(f"Max (non-zero): {np.max(capacity_reduction_nonzero):.6f}\n")
                f.write(f"Median (non-zero): {np.median(capacity_reduction_nonzero):.6f}\n")
            # Capacity reduction distribution
            cap_nonzero = np.sum(capacity_reduction_flat < 0)
            cap_zero = np.sum(capacity_reduction_flat == 0)
        
            f.write(f"Capacity reduced: {cap_nonzero:,} ({cap_nonzero/len(capacity_reduction_flat)*100:.1f}%)\n")
            f.write(f"No capacity reduction: {cap_zero:,} ({cap_zero/len(capacity_reduction_flat)*100:.1f}%)\n")
        else:
            f.write(f"\n=== NO CAPACITY REDUCTION DATA FOUND - {city_name} ===")
        
    # Print statistics for Y values (ALL VALUES)
    if len(y_values_flat) > 0:
        print(f"\n=== Y VALUE STATISTICS (ALL VALUES) ({target_feature}) - {city_name} ===")
        print(f"Total samples: {len(y_values_flat):,}")
        print(f"Mean: {np.mean(y_values_flat):.6f}")
        print(f"Std: {np.std(y_values_flat):.6f}")
        print(f"Min: {np.min(y_values_flat):.6f}")
        print(f"Max: {np.max(y_values_flat):.6f}")
        print(f"Median: {np.median(y_values_flat):.6f}")
        
        # Y values sign distribution
        y_positive = np.sum(y_values_flat > 0)
        y_negative = np.sum(y_values_flat < 0)
        y_zero = np.sum(y_values_flat == 0)
        
        print(f"Positive (increase): {y_positive:,} ({y_positive/len(y_values_flat)*100:.1f}%)")
        print(f"Negative (decrease): {y_negative:,} ({y_negative/len(y_values_flat)*100:.1f}%)")
        print(f"Zero (no change): {y_zero:,} ({y_zero/len(y_values_flat)*100:.1f}%)")
        
        # Statistics for NON-ZERO values only
        if len(y_values_nonzero) > 0:
            print(f"\n=== NON-ZERO Y VALUE STATISTICS ({target_feature}) - {city_name} ===")
            print(f"Non-zero samples: {len(y_values_nonzero):,}")
            print(f"Mean (non-zero): {np.mean(y_values_nonzero):.6f}")
            print(f"Std (non-zero): {np.std(y_values_nonzero):.6f}")
            print(f"Min (non-zero): {np.min(y_values_nonzero):.6f}")
            print(f"Max (non-zero): {np.max(y_values_nonzero):.6f}")
            print(f"Median (non-zero): {np.median(y_values_nonzero):.6f}")
        else:
            print(f"\n=== NO NON-ZERO Y VALUES FOUND ===")
    else:
        print(f"\n=== NO Y VALUE DATA FOUND - {city_name} ===")
    
    # Print statistics for Capacity Reduction
    if len(capacity_reduction_nonzero) > 0:
        print(f"\n=== CAPACITY REDUCTION STATISTICS - {city_name} ===")
        print(f"Total samples: {len(capacity_reduction_flat):,}")
        print(f"Non-zero samples: {len(capacity_reduction_nonzero):,}")
        print(f"Mean (non-zero): {np.mean(capacity_reduction_nonzero):.6f}")
        print(f"Std (non-zero): {np.std(capacity_reduction_nonzero):.6f}")
        print(f"Min (non-zero): {np.min(capacity_reduction_nonzero):.6f}")
        print(f"Max (non-zero): {np.max(capacity_reduction_nonzero):.6f}")
        print(f"Median (non-zero): {np.median(capacity_reduction_nonzero):.6f}")
        
        # Capacity reduction distribution
        cap_nonzero = np.sum(capacity_reduction_flat < 0)
        cap_zero = np.sum(capacity_reduction_flat == 0)
        
        print(f"Capacity reduced: {cap_nonzero:,} ({cap_nonzero/len(capacity_reduction_flat)*100:.1f}%)")
        print(f"No capacity reduction: {cap_zero:,} ({cap_zero/len(capacity_reduction_flat)*100:.1f}%)")
    else:
        print(f"\n=== NO CAPACITY REDUCTION DATA FOUND - {city_name} ===")
    
    plt.show()
    
    return y_values_flat, capacity_reduction_flat

def process_single_city(city, project_root, result_path):
    """Process a single city and collect Y values and capacity reduction for distribution analysis"""
    
    # Initialize features collector
    all_features_data = []
    
    sim_input_paths = list()
    # Paths to compressed directories
    compressed_input_paths = [os.path.join(project_root, 'data','inductive_data',city, f'compressed_{city}_hex_{hex_size}_seed_{seed}') for hex_size in hex_sizes]

    for path in compressed_input_paths:
        if os.path.exists(path) and os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith('.tar.gz'):
                    sim_input_paths.append(os.path.join(path, f))
    
    basecase_links_path = os.path.join(project_root, 'data_new', 'links_and_stats', 'basecases_mean', city, 'basecase_average_output_links.geojson')
    
    # Load basecase data (simplified - no need for stats or trips data)
    gdf_basecase_links = gpd.read_file(basecase_links_path)
    gdf_basecase_links = gdf_basecase_links.set_crs("EPSG:25832", allow_override=True)
    
    sim_input_paths.sort()
    
    print(f"Processing {len(sim_input_paths)} files for {city}")
    
    for i in tqdm(range(0, len(sim_input_paths), batch_size), desc="Processing in batches ...", unit="batch"):
        sliced_inputs = sim_input_paths[i:i+batch_size]
        
        try:
            networks, temp_dirs = extract_and_get_networks(sliced_inputs)
            networks = [network for network in networks if not network.endswith(".DS_Store")]
            
            result_dic_output_links, _ = compute_result_dic(basecase_links=gdf_basecase_links, networks=networks, use_destination_activity=use_destination_activity)
            base_gdf = result_dic_output_links["base_network_no_policies"]
            
            # Extract only Y values and capacity reduction
            features_data = extract_y_and_capacity_reduction(city, result_dic=result_dic_output_links, links_base_case=base_gdf)
            
            # Collect features data
            all_features_data.extend(features_data)
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            temp_dirs = []  # Ensure temp_dirs is defined for cleanup
            
        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_dir}: {e}")
            
    print(f"Collected {len(all_features_data)} graphs' features for {city}")
    
    # Plot the distributions
    y_values_array, capacity_reduction_array = plot_y_and_capacity_distributions(all_features_data, city, result_path)
    
    return all_features_data, y_values_array, capacity_reduction_array
    
    
def main():
    """Main function to analyze Y values and capacity reduction distributions for all cities"""
    
    # Create the result base path
    result_base_path = os.path.join(project_root, 'data_new', 'data_analysis')
    os.makedirs(result_base_path, exist_ok=True)
    
    all_cities_data = {}
    
    for city in all_cities:
        print(f"\n{'='*50}")
        print(f"Starting analysis for {city}")
        print(f"{'='*50}")
        
        result_path = os.path.join(result_base_path, city)
        os.makedirs(result_path, exist_ok=True)
        
        # Process city and collect features
        process_single_city(city, project_root, result_path)
        print(f"Completed analysis for {city}")

if __name__ == '__main__':
    main()