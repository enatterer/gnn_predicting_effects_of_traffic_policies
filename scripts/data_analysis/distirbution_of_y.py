
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from torch_geometric.data import Data

target_features = ['abs_vol_car', 'abs_vol_car_percentage','vol_car_signed_log','vol_car_percentage_signed_log', 'vol_car_mean_std','vol_car_percentage_mean_std']
city_name = ['nuernberg']

def plot_all_pt_files_distributions(data_dir, save_path, target_features, city_name):
    """
    Iterate over all .pt files in a directory and plot their Y value distributions
    
    Args:
        data_dir: Directory containing .pt files
        save_path: Directory to save plots
        target_features: List of target features to analyze
    """
    
    # Find all .pt files
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    pt_files.sort()  # Sort for reproducibility
    
    print(f"Found {len(pt_files)} .pt files in {data_dir}")
    
    if len(pt_files) == 0:
        print("No .pt files found!")
        return
    
    # Collect all Y values for each target feature
    all_y_values_dict = {target_feature: [] for target_feature in target_features}
    
    for i, pt_file in enumerate(pt_files):
        try:
            # Load the data
            data = torch.load(pt_file)
            
            # Extract Y values for each target feature
            for target_feature in target_features:
                # Check for specific target feature first
                if hasattr(data, f'y_{target_feature}'):
                    y_values = getattr(data, f'y_{target_feature}')
                elif hasattr(data, 'y') and data.y is not None:
                    y_values = data.y
                else:
                    # Debug: Print available attributes for the first few files
                    if i < 5:
                        print(f"Debug: Available attributes in {pt_file}:")
                        for attr in dir(data):
                            if not attr.startswith('_') and hasattr(data, attr):
                                try:
                                    val = getattr(data, attr)
                                    if hasattr(val, 'shape'):
                                        print(f"  {attr}: {type(val)}, shape={val.shape}")
                                    else:
                                        print(f"  {attr}: {type(val)}")
                                except:
                                    print(f"  {attr}: <error reading>")
                    print(f"Warning: No Y values found for {target_feature} in {pt_file}")
                    continue
                
                # Convert to numpy and flatten
                if isinstance(y_values, torch.Tensor):
                    y_values = y_values.cpu().numpy().flatten()
                else:
                    y_values = np.array(y_values).flatten()
                
                # Filter out None values and ensure we have valid numeric data
                if y_values is not None and len(y_values) > 0:
                    try:
                        # Convert to numpy array and ensure numeric types
                        y_values = np.array(y_values, dtype=np.float64)
                        
                        # Filter out None, NaN, and inf values
                        valid_mask = ~(np.isnan(y_values) | np.isinf(y_values))
                        y_values = y_values[valid_mask]
                        
                        if len(y_values) > 0:
                            all_y_values_dict[target_feature].extend(y_values)
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not convert Y values to numeric for {target_feature} in {pt_file}: {e}")
                        continue
                else:
                    print(f"Warning: Invalid Y values for {target_feature} in {pt_file}")
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(pt_files)} files...")
                
        except Exception as e:
            print(f"Error processing {pt_file}: {e}")
            continue
    
    # Convert to numpy arrays
    for target_feature in target_features:
        all_y_values_dict[target_feature] = np.array(all_y_values_dict[target_feature])
        print(f"\nTotal samples collected for {target_feature}: {len(all_y_values_dict[target_feature]):,}")
    
    # Create distribution plots for each target feature
    for target_feature in target_features:
        if len(all_y_values_dict[target_feature]) > 0:
            plot_y_distributions_from_arrays(
                all_y_values_dict[target_feature], 
                city_name, 
                save_path,
                target_feature
            )
        else:
            print(f"No valid data found for {target_feature}")

def plot_y_distributions_from_arrays(y_values_flat, city_name, save_path, target_feature):
    """Plot distributions of Y values from numpy arrays"""
    
    # Ensure we have valid numeric data
    if y_values_flat is None or len(y_values_flat) == 0:
        print("Error: No valid Y values found")
        return None
    
    # Convert to numpy arrays and filter out None, NaN, and inf values
    try:
        y_values_flat = np.array(y_values_flat, dtype=np.float64)
        
        # Filter out None, NaN, and inf values
        valid_mask = ~(np.isnan(y_values_flat) | np.isinf(y_values_flat))
        y_values_flat = y_values_flat[valid_mask]
    except (ValueError, TypeError) as e:
        print(f"Error: Could not convert data to numeric types: {e}")
        return None
    
    if len(y_values_flat) == 0:
        print("Error: No valid Y values after filtering")
        return None
    
    # Filter out zero Y values for histogram
    y_values_nonzero = y_values_flat[y_values_flat != 0]
    
    # Create 1x3 plot: Y values only
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Y Values Distributions - {city_name}', fontsize=16)
    
    # Y VALUES
    # 1. Y Values Histogram (NON-ZERO ONLY)
    if len(y_values_nonzero) > 0:
        # Create bin edges for ranges like -100 to -95, -95 to -90, etc.
        bin_edges = np.arange(-100, 105, 1)  # From -100 to 100 in steps of 5
        # This creates bins: [-100,-95], [-95,-90], [-90,-85], ..., [95,100]

        axes[0].hist(y_values_nonzero, bins=bin_edges, alpha=0.7, edgecolor='black', color='blue')
        axes[0].set_title(f'Y Values Histogram - Non-Zero Only ({target_feature})')
        axes[0].set_xlim(-100,100)
    else:
        axes[0].text(0.5, 0.5, 'No non-zero Y values', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title(f'Y Values Histogram - No Data ({target_feature})')
    axes[0].set_xlabel('Y Value (excluding zeros)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Y Values Box plot (ALL VALUES INCLUDING ZEROS)
    if len(y_values_flat) > 0:
        axes[1].boxplot(y_values_flat)
        axes[1].set_title('Y Values Box Plot (All Values)')
    else:
        axes[1].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Y Values Box Plot - No Data')
    axes[1].set_ylabel('Y Value')
    axes[1].grid(True, alpha=0.3)
    
    if len(y_values_nonzero) > 0:
        axes[2].boxplot(y_values_nonzero)
        axes[2].set_title('Y Values Box Plot (Non-Zero Only)')
    else:
        axes[2].text(0.5, 0.5, 'No non-zero Y values', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Y Values Box Plot - No Data')
    axes[2].set_ylabel('Y Value')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, f'y_{target_feature}_distribution_{city_name}_nonzero.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {plot_path}")
    
    # Print statistics
    print_y_statistics(y_values_flat, y_values_nonzero, city_name, target_feature, save_path)
    
    plt.show()
    
    return y_values_flat

def print_y_statistics(y_values_flat, y_values_nonzero, city_name, target_feature, save_path):
    """Print and save statistics for Y values only"""
    
    with open(os.path.join(save_path, f'y_{target_feature}_distribution_{city_name}_nonzero.txt'), 'w') as f:
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
        
        
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for city in city_name:
        data_dir = os.path.join(project_root, 'data', 'inductive_data', 'training_data', city)
        save_path = os.path.join(project_root, 'data_new', 'data_analysis_only_stadt', city)
        os.makedirs(save_path, exist_ok=True)
        plot_all_pt_files_distributions(data_dir, save_path, target_features, city)