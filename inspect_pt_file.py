#!/usr/bin/env python3
"""
Inspect a .pt file and show all its attributes and features.
Usage: python inspect_pt_file.py path/to/file.pt
"""

import torch
import sys
import os

def inspect_pt_file(file_path):
    """Load and inspect a .pt file."""
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return
    
    print(f"🔍 Inspecting: {file_path}")
    print(f"📁 File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print("=" * 60)
    
    try:
        # Load the data
        data = torch.load(file_path, map_location='cpu')
        print(f"✅ Successfully loaded data")
        print(f"📦 Data type: {type(data)}")
        print()
        
        # Get all attributes
        all_attrs = [attr for attr in dir(data) if not attr.startswith('_')]
        print(f"🔧 All attributes ({len(all_attrs)}):")
        print(f"   {all_attrs}")
        print()
        
        # ✅ ADD COMPREHENSIVE TARGET ANALYSIS
        print(f"🎯 COMPREHENSIVE TARGET ANALYSIS")
        print("=" * 60)
        
        # Define the expected target variations
        target_variations = [
            'y_abs_vol_car',
            'y_abs_vol_car_percentage', 
            'y_vol_car_signed_log',
            'y_vol_car_percentage_signed_log',
            'y_vol_car_mean_std',
            'y_vol_car_percentage_mean_std',
            'y_vol_car_min_max',
            'y_vol_car_percentage_min_max'
        ]
        
        # Find all y-* attributes in the data
        y_attrs = [attr for attr in all_attrs if attr.startswith('y')]
        
        print(f"📊 Expected target variations: {len(target_variations)}")
        print(f"📊 Found y-* attributes: {len(y_attrs)}")
        print(f"📊 Found attributes: {y_attrs}")
        print()
        
        # Check each expected target variation
        print(f"🔍 Detailed Target Analysis:")
        for i, target_name in enumerate(target_variations, 1):
            print(f"\n{i}. {target_name}:")
            
            if hasattr(data, target_name):
                target = getattr(data, target_name)
                print(f"   ✅ EXISTS")
                print(f"   📦 Type: {type(target)}")
                
                if hasattr(target, 'shape'):
                    print(f"   📏 Shape: {target.shape}")
                    print(f"   🔤 Dtype: {target.dtype}")
                    
                    if target.numel() > 0:
                        print(f"   📈 Range: [{target.min().item():.6f}, {target.max().item():.6f}]")
                        print(f"   📊 Mean: {target.mean().item():.6f}")
                        print(f"   📊 Std: {target.std().item():.6f}")
                        print(f"   📊 Median: {target.median().item():.6f}")
                        
                        # Check for special values
                        has_nan = torch.isnan(target).any().item()
                        has_inf = torch.isinf(target).any().item()
                        num_zeros = (target == 0).sum().item()
                        num_positive = (target > 0).sum().item()
                        num_negative = (target < 0).sum().item()
                        
                        print(f"   🔍 Has NaN: {has_nan}")
                        print(f"   🔍 Has Inf: {has_inf}")
                        print(f"   🔍 Zeros: {num_zeros} ({num_zeros/target.numel()*100:.2f}%)")
                        print(f"   🔍 Positive: {num_positive} ({num_positive/target.numel()*100:.2f}%)")
                        print(f"   🔍 Negative: {num_negative} ({num_negative/target.numel()*100:.2f}%)")
                        
                        # Show percentiles
                        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                        perc_values = torch.quantile(target, torch.tensor([p/100 for p in percentiles]))
                        print(f"   📊 Percentiles:")
                        for p, val in zip(percentiles, perc_values):
                            print(f"      {p:2d}%: {val.item():8.6f}")
                        
                        # Show sample values
                        sample_size = min(10, target.numel())
                        sample_indices = torch.randperm(target.numel())[:sample_size]
                        sample_values = target.flatten()[sample_indices]
                        print(f"   🎲 Random sample values:")
                        for j, val in enumerate(sample_values):
                            print(f"      Sample {j+1:2d}: {val.item():8.6f}")
                        
                        # Distribution analysis
                        if target.numel() > 100:  # Only for reasonably sized tensors
                            # Histogram-like analysis
                            sorted_vals, _ = torch.sort(target.flatten())
                            n = len(sorted_vals)
                            bins = 10
                            bin_size = n // bins
                            print(f"   📊 Distribution (10 bins):")
                            for b in range(bins):
                                start_idx = b * bin_size
                                end_idx = (b + 1) * bin_size if b < bins - 1 else n
                                bin_min = sorted_vals[start_idx].item()
                                bin_max = sorted_vals[end_idx - 1].item()
                                bin_count = end_idx - start_idx
                                print(f"      Bin {b+1:2d}: [{bin_min:8.6f}, {bin_max:8.6f}] -> {bin_count:5d} values ({bin_count/n*100:5.2f}%)")
                else:
                    print(f"   ❌ No shape attribute (not a tensor?)")
                    print(f"   📦 Value: {target}")
            else:
                print(f"   ❌ NOT FOUND")
        
        # ✅ ADD TARGET COMPARISON SECTION
        print(f"\n🔄 TARGET COMPARISON ANALYSIS:")
        print("=" * 50)
        
        # Find pairs for comparison
        vol_targets = [attr for attr in y_attrs if 'vol_car' in attr and 'percentage' not in attr]
        percentage_targets = [attr for attr in y_attrs if 'vol_car_percentage' in attr]
        
        print(f"📊 Volume targets: {vol_targets}")
        print(f"📊 Percentage targets: {percentage_targets}")
        
        # Compare raw vs normalized versions
        normalization_types = ['signed_log', 'mean_std', 'min_max']
        
        for norm_type in normalization_types:
            vol_norm = f'y_vol_car_{norm_type}'
            perc_norm = f'y_vol_car_percentage_{norm_type}'
            
            if hasattr(data, vol_norm) and hasattr(data, perc_norm):
                print(f"\n🔍 Comparing {norm_type} normalization:")
                
                vol_target = getattr(data, vol_norm)
                perc_target = getattr(data, perc_norm)
                
                print(f"   Volume ({vol_norm}):")
                print(f"      Range: [{vol_target.min().item():.6f}, {vol_target.max().item():.6f}]")
                print(f"      Mean: {vol_target.mean().item():.6f}, Std: {vol_target.std().item():.6f}")
                
                print(f"   Percentage ({perc_norm}):")
                print(f"      Range: [{perc_target.min().item():.6f}, {perc_target.max().item():.6f}]")
                print(f"      Mean: {perc_target.mean().item():.6f}, Std: {perc_target.std().item():.6f}")
                
                # Correlation between volume and percentage targets
                if vol_target.numel() == perc_target.numel():
                    correlation = torch.corrcoef(torch.stack([vol_target.flatten(), perc_target.flatten()]))[0, 1]
                    print(f"      Correlation: {correlation.item():.6f}")
        
        # ✅ ADD RAW TARGET INSPECTION
        print(f"\n📋 RAW TARGET INSPECTION:")
        print("=" * 40)
        
        if hasattr(data, 'y_abs_vol_car') and hasattr(data, 'y_abs_vol_car_percentage'):
            raw_vol = data.y_abs_vol_car
            raw_perc = data.y_abs_vol_car_percentage
            
            print(f"📊 Raw volume changes (y_abs_vol_car):")
            print(f"   Range: [{raw_vol.min().item():.2f}, {raw_vol.max().item():.2f}]")
            print(f"   Mean: {raw_vol.mean().item():.2f}, Std: {raw_vol.std().item():.2f}")
            
            print(f"📊 Raw percentage changes (y_abs_vol_car_percentage):")
            print(f"   Range: [{raw_perc.min().item():.2f}%, {raw_perc.max().item():.2f}%]")
            print(f"   Mean: {raw_perc.mean().item():.2f}%, Std: {raw_perc.std().item():.2f}%")
            
            # Show some interpretation
            positive_vol_changes = (raw_vol > 0).sum().item()
            negative_vol_changes = (raw_vol < 0).sum().item()
            zero_vol_changes = (raw_vol == 0).sum().item()
            
            print(f"📈 Traffic volume changes:")
            print(f"   Increases: {positive_vol_changes} edges ({positive_vol_changes/raw_vol.numel()*100:.1f}%)")
            print(f"   Decreases: {negative_vol_changes} edges ({negative_vol_changes/raw_vol.numel()*100:.1f}%)")
            print(f"   No change: {zero_vol_changes} edges ({zero_vol_changes/raw_vol.numel()*100:.1f}%)")
        
        # Continue with existing code...
        
        # Focus on target attributes (y-*)
        y_attrs = [attr for attr in all_attrs if attr.startswith('y')]
        print(f"\n🎯 All target attributes ({len(y_attrs)}):")
        if y_attrs:
            for attr in y_attrs:
                value = getattr(data, attr)
                print(f"   {attr}:")
                print(f"      Type: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"      Shape: {value.shape}")
                    print(f"      Dtype: {value.dtype}")
                    if value.numel() > 0:
                        print(f"      Range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                        print(f"      Mean: {value.mean().item():.4f}, Std: {value.std().item():.4f}")
                else:
                    print(f"      Value: {value}")
                print()
        else:
            print("   ❌ No target attributes found!")
        
        # Show main data tensors
        main_tensors = ['x', 'edge_index', 'pos', 'batch']
        print(f"📊 Main data tensors:")
        for attr in main_tensors:
            if hasattr(data, attr):
                value = getattr(data, attr)
                if hasattr(value, 'shape'):
                    print(f"   {attr}: {value.shape} ({value.dtype})")
                else:
                    print(f"   {attr}: {type(value)} = {value}")
            else:
                print(f"   {attr}: ❌ Not found")
        print()
        
        # Show Laplacian PE info if it exists
        if hasattr(data, 'lap_pe'):
            print(f"🌐 Laplacian Positional Encoding info:")
            lap_pe = data.lap_pe
            print(f"   Shape: {lap_pe.shape}")
            print(f"   Dtype: {lap_pe.dtype}")
            if lap_pe.numel() > 0:
                print(f"   Range: [{lap_pe.min().item():.6f}, {lap_pe.max().item():.6f}]")
                print(f"   Mean: {lap_pe.mean().item():.6f}, Std: {lap_pe.std().item():.6f}")
                print(f"   Has NaN: {torch.isnan(lap_pe).any().item()}")
                print(f"   Has Inf: {torch.isinf(lap_pe).any().item()}")
                
                # Check if all zeros
                is_all_zeros = torch.allclose(lap_pe, torch.zeros_like(lap_pe))
                print(f"   All zeros: {is_all_zeros}")
                
                # Non-zero ratio
                non_zero_ratio = (torch.abs(lap_pe) > 1e-8).float().mean().item()
                print(f"   Non-zero ratio: {non_zero_ratio:.4f}")
                
                # Show statistics for each PE dimension
                if lap_pe.shape[1] <= 16:  # Only show if reasonable number of dimensions
                    print(f"   Per-dimension statistics:")
                    for dim in range(lap_pe.shape[1]):
                        dim_data = lap_pe[:, dim]
                        print(f"      Dim {dim}: mean={dim_data.mean().item():.6f}, "
                              f"std={dim_data.std().item():.6f}, "
                              f"range=[{dim_data.min().item():.6f}, {dim_data.max().item():.6f}]")
                
                # Show sample PE values
                print(f"   Sample Laplacian PE values (first 5 nodes, all dimensions):")
                sample_size = min(5, lap_pe.shape[0])
                for i in range(sample_size):
                    pe_vals = lap_pe[i, :]
                    pe_str = ", ".join([f"{val.item():.6f}" for val in pe_vals])
                    print(f"      Node {i}: [{pe_str}]")
                
                # Check eigenvalue structure (for validation)
                print(f"   Eigenvalue structure analysis:")
                # Compute the magnitude of each eigenvector
                eigvec_magnitudes = torch.norm(lap_pe, dim=0)
                print(f"      Eigenvector magnitudes: {[f'{mag.item():.4f}' for mag in eigvec_magnitudes]}")
                
                # Check orthogonality (should be close to orthogonal)
                if lap_pe.shape[1] > 1:
                    correlation_matrix = torch.corrcoef(lap_pe.T)
                    off_diagonal = correlation_matrix[~torch.eye(correlation_matrix.shape[0], dtype=bool)]
                    max_correlation = torch.abs(off_diagonal).max().item()
                    print(f"      Max off-diagonal correlation: {max_correlation:.6f} (should be close to 0)")
            print()
        else:
            print("🌐 Laplacian Positional Encoding: ❌ Not found")
            print()
        
        # Show metadata
        metadata_attrs = ['policy_region', 'scenario', 'city', 'num_nodes']
        print(f"📋 Metadata:")
        for attr in metadata_attrs:
            if hasattr(data, attr):
                value = getattr(data, attr)
                print(f"   {attr}: {value}")
            else:
                print(f"   {attr}: ❌ Not found")
        print()
        
        # Show additional data attributes
        additional_attrs = ['unscaled_vol_base', 'edge_weights']
        print(f"📈 Additional data attributes:")
        for attr in additional_attrs:
            if hasattr(data, attr):
                value = getattr(data, attr)
                if hasattr(value, 'shape'):
                    print(f"   {attr}: {value.shape} ({value.dtype})")
                    if value.numel() > 0:
                        print(f"      Range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                        print(f"      Mean: {value.mean().item():.4f}, Std: {value.std().item():.4f}")
                else:
                    print(f"   {attr}: {type(value)} = {value}")
            else:
                print(f"   {attr}: ❌ Not found")
        print()
        
        # Show features info if x exists
        if hasattr(data, 'x') and hasattr(data.x, 'shape'):
            print(f"🧬 Feature tensor info:")
            print(f"   Number of nodes: {data.x.shape[0]}")
            print(f"   Number of features: {data.x.shape[1]}")
            print(f"   Feature range: [{data.x.min().item():.4f}, {data.x.max().item():.4f}]")
            print(f"   Has NaN: {torch.isnan(data.x).any().item()}")
            print(f"   Has Inf: {torch.isinf(data.x).any().item()}")
            print()
            # Print sample values for first 5 nodes and all features
            sample_nodes = min(5, data.x.shape[0])
            sample_features = min(28, data.x.shape[1])  # Show reasonable number of features
            print(f"   Sample data.x values (first {sample_nodes} nodes, first {sample_features} features):")
            for i in range(sample_nodes):
                vals = data.x[i, :sample_features]
                vals_str = ", ".join([f"{v.item():.4f}" for v in vals])
                print(f"      Node {i}: [{vals_str}]")
            print()
        
        # Show positional info if pos exists
        if hasattr(data, 'pos') and hasattr(data.pos, 'shape'):
            print(f"📍 Positional tensor info:")
            print(f"   Pos shape: {data.pos.shape}")
            print(f"   Pos dtype: {data.pos.dtype}")
            
            if len(data.pos.shape) == 3:
                # Shape is [N, T, 2] or similar
                print(f"   Time steps: {data.pos.shape[1]}")
                print(f"   Coordinate dimensions: {data.pos.shape[2]}")
                
                # Show statistics for each time step and coordinate
                for t in range(min(3, data.pos.shape[1])):  # Show max 3 time steps
                    print(f"   Time step {t}:")
                    for coord in range(data.pos.shape[2]):
                        coord_data = data.pos[:, t, coord]
                        print(f"      Coord {coord}: range=[{coord_data.min().item():.4f}, {coord_data.max().item():.4f}], mean={coord_data.mean().item():.4f}, std_dev={coord_data.std().item():.4f}")

                # Show sample position values
                print(f"   Sample positions (first 5 nodes, time step 2 if available):")
                time_idx = min(2, data.pos.shape[1] - 1)  # Use time step 2 or last available
                sample_size = min(5, data.pos.shape[0])
                for i in range(sample_size):
                    pos_vals = data.pos[i, time_idx, :]
                    print(f"      Node {i}: [{pos_vals[0].item():.6f}, {pos_vals[1].item():.6f}]")
                    
            elif len(data.pos.shape) == 2:
                # Shape is [N, 2] or similar
                print(f"   Coordinate dimensions: {data.pos.shape[1]}")
                
                # Show statistics for each coordinate
                for coord in range(data.pos.shape[1]):
                    coord_data = data.pos[:, coord]
                    print(f"   Coord {coord}: range=[{coord_data.min().item():.4f}, {coord_data.max().item():.4f}], mean={coord_data.mean().item():.4f}")
                
                # Show sample position values
                print(f"   Sample positions (first 10 nodes):")
                sample_size = min(10, data.pos.shape[0])
                for i in range(sample_size):
                    pos_vals = data.pos[i, :]
                    pos_str = ", ".join([f"{val.item():.6f}" for val in pos_vals])
                    print(f"      Node {i}: [{pos_str}]")
            
            print(f"   Has NaN: {torch.isnan(data.pos).any().item()}")
            print(f"   Has Inf: {torch.isinf(data.pos).any().item()}")
            print()
        
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_pt_file.py <path_to_pt_file>")
        print("Example: python inspect_pt_file.py /home/abasu/gnn_predicting_effects_of_traffic_policies/data/inductive_data/training_data/rosenheim/000001.pt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    inspect_pt_file(file_path)

if __name__ == "__main__":
    main()