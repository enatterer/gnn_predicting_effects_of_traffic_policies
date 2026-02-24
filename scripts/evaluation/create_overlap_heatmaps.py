#!/usr/bin/env python3
"""
Create heatmaps from overlap results.

X-axis: # of target-city data available (train+val)
Y-axis: Threshold for edges that may be affected by the policy (%)
"""

import argparse
import json
import os
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Colorblind-friendly colormap with green for higher values and red for lower values
COLORMAP = 'RdYlGn'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Mapping from train_count, val_count to total data available
SPLIT_MAPPING = {
    (10, 3): 13,
    (20, 5): 25,
    (40, 10): 50,
    (80, 20): 100,
    (160, 40): 200
}

# Default thresholds for y-axis (ordered from highest to lowest for display)
THRESHOLDS = [1.25, 1.0, 0.75, 0.5, 0.25]

# Default X-axis values (total data available)
X_AXIS_VALUES = [13, 25, 50, 100, 200]


def find_overlap_files(overlaps_dir, city='regensburg', overlap_normalization=None, k_filter=None, test_count=100, required_thresholds=None):
    """
    Find all overlap JSON files for a given city.
    Returns a dictionary mapping (train_count, val_count) to file path.

    When multiple files exist for the same (train_count, val_count), prefers the newest file
    whose results contain at least one of required_thresholds (if provided). This ensures we
    use multi-threshold runs (e.g. 0.5, 1.0, 1.5, 2.0) instead of single-threshold runs (e.g. 100.0).
    """
    pattern = os.path.join(
        overlaps_dir,
        f'overlap_results_varying_capacity_reduction_{city}_t*_v*_test{test_count}_*.json'
    )
    files = glob.glob(pattern)

    # Group candidates by (train_count, val_count), keeping metadata and result keys for filtering
    key_to_candidates = defaultdict(list)

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        metadata = data.get('metadata', {})
        file_norm = metadata.get('overlap_normalization')
        file_k = metadata.get('k')
        if overlap_normalization and file_norm != overlap_normalization:
            continue
        if k_filter is not None and file_k != k_filter:
            continue
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        train_count = None
        val_count = None
        for part in parts:
            if part.startswith('t') and part[1:].isdigit():
                train_count = int(part[1:])
            elif part.startswith('v') and part[1:].isdigit():
                val_count = int(part[1:])
        if train_count is None or val_count is None:
            continue
        key = (train_count, val_count)
        result_keys = set(float(k) for k in data.get('results', {}).keys())
        key_to_candidates[key].append((filepath, result_keys, os.path.getmtime(filepath)))
    # For each key, pick best file: newest that has at least one required_threshold (if specified)
    split_to_file = {}
    for key, candidates in key_to_candidates.items():
        # Sort by mtime descending (newest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        chosen = None
        for filepath, result_keys, _ in candidates:
            if required_thresholds is None:
                chosen = filepath
                break
            if any(t in result_keys for t in required_thresholds):
                chosen = filepath
                break
        if chosen is None:
            chosen = candidates[0][0] if candidates else None
        if chosen is not None:
            split_to_file[key] = chosen
    return split_to_file


def parse_float_list(arg: str):
    """Parse a comma-separated list of floats."""
    return [float(x.strip()) for x in arg.split(',') if x.strip()]


def load_overlap_data(filepath):
    """Load overlap data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def extract_overlap_matrix(data, model_type='scratch'):
    """
    Extract overlap values as a matrix.
    
    Args:
        data: Loaded JSON data
        model_type: 'scratch' or 'finetune'
    
    Returns:
        Dictionary mapping threshold to list of overlap values across seeds
    """
    results = data.get('results', {})
    threshold_overlaps = {}
    
    for threshold_str, seed_results in results.items():
        threshold = float(threshold_str)
        overlaps = []
        
        for seed_idx in sorted(seed_results.keys(), key=int):
            seed_data = seed_results[seed_idx]
            if seed_data is not None:
                if model_type == 'scratch':
                    overlaps.append(seed_data.get('scratch_overlap', None))
                else:
                    overlaps.append(seed_data.get('finetune_overlap', None))
            else:
                overlaps.append(None)
        
        # Calculate mean across seeds (excluding None values)
        valid_overlaps = [o for o in overlaps if o is not None]
        if valid_overlaps:
            threshold_overlaps[threshold] = np.mean(valid_overlaps)
        else:
            threshold_overlaps[threshold] = np.nan
    
    return threshold_overlaps


def create_heatmap_data(split_to_file, thresholds, x_values, model_type='scratch'):
    """
    Create a 2D matrix for heatmap.
    
    Returns:
        heatmap_matrix: 2D numpy array (n_thresholds x n_splits)
        threshold_labels: List of threshold values for y-axis
        split_labels: List of total data values for x-axis
    """
    # Initialize matrix
    n_thresholds = len(thresholds)
    n_splits = len(x_values)
    heatmap_matrix = np.full((n_thresholds, n_splits), np.nan)
    
    # Map split to column index
    split_to_col = {total: i for i, total in enumerate(x_values)}
    
    # Fill matrix
    for (train_count, val_count), filepath in split_to_file.items():
        total_data = SPLIT_MAPPING.get((train_count, val_count))
        if total_data is None:
            continue
        
        col_idx = split_to_col.get(total_data)
        if col_idx is None:
            continue
        
        # Load data
        data = load_overlap_data(filepath)
        threshold_overlaps = extract_overlap_matrix(data, model_type=model_type)
        
        # Fill row for each threshold
        for row_idx, threshold in enumerate(thresholds):
            if threshold in threshold_overlaps:
                heatmap_matrix[row_idx, col_idx] = threshold_overlaps[threshold]
    
    return heatmap_matrix, thresholds, x_values


def create_heatmap(heatmap_matrix, threshold_labels, split_labels, 
                   model_type='scratch', city='regensburg', output_dir='plots',
                   overlap_normalization=None, k_filter=None,
                   font_size=12, label_size=14, tick_size=12, value_size=11,
                   cbar_labelpad=28,
                   y_axis_label='Threshold for edges that may be affected by the policy (%)'):
    """
    Create and save a heatmap.
    
    Args:
        heatmap_matrix: 2D numpy array
        threshold_labels: List of threshold values for y-axis
        split_labels: List of total data values for x-axis
        model_type: 'scratch' or 'finetune'
        city: City name
        output_dir: Directory to save plots
    """
    plt.rcParams['font.size'] = font_size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with scale from 0.0 to 1.0
    im = ax.imshow(heatmap_matrix, aspect='auto', cmap=COLORMAP,
                   vmin=0.0, vmax=1.0, interpolation='nearest')
    
    # Set ticks and labels (x-axis as integers: 25 instead of 25.0)
    ax.set_xticks(np.arange(len(split_labels)))
    x_tick_labels = [str(int(l)) if isinstance(l, (int, float)) and l == int(l) else str(l) for l in split_labels]
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel('# of target-city data available', fontsize=label_size)
    
    ax.set_yticks(np.arange(len(threshold_labels)))
    ax.set_yticklabels([f'{t:.2f}%' for t in threshold_labels])
    ax.set_ylabel(y_axis_label, fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    
    # Add colorbar with ticks from 0.0 to 1.0
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(0.0, 1.1, 0.1))
    cbar.set_label('Overlap', rotation=270, labelpad=cbar_labelpad, fontsize=label_size)
    
    # Add text annotations for each cell
    for i in range(len(threshold_labels)):
        for j in range(len(split_labels)):
            value = heatmap_matrix[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", color="black",
                             fontsize=value_size)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    norm_suffix_map = {
        'k': '_norm_to_k',
        'eligible': '_norm_to_eligible_roads',
        'test_set': '_norm_to_test_set'
    }
    norm_suffix = norm_suffix_map.get(overlap_normalization, f"_norm{overlap_normalization}") if overlap_normalization else ""
    k_suffix = f"_k{k_filter}" if k_filter is not None else ""
    output_filename = f'overlap_heatmap_{model_type}_{city}{norm_suffix}{k_suffix}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create heatmaps from overlap results')
    parser.add_argument('--city', type=str, default='regensburg',
                       help='City name (default: regensburg)')
    parser.add_argument('--overlaps_dir', type=str, 
                       default='scripts/evaluation/overlaps',
                       help='Directory containing overlap JSON files')
    parser.add_argument('--output_dir', type=str, default='scripts/evaluation/plots',
                       help='Directory to save plots (default: scripts/evaluation/plots)')
    parser.add_argument('--x_values', type=str, default=None,
                        help='Comma-separated list of total data counts for x-axis (e.g., "25,50,100,200")')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='Comma-separated list of thresholds (%) for y-axis (e.g., "1.0,0.75,0.5,0.25")')
    parser.add_argument('--overlap_normalization', type=str, default=None,
                        choices=['k', 'eligible', 'test_set'],
                        help="Filter overlap files by normalization: 'k', 'eligible', or 'test_set'")
    parser.add_argument('--k', type=int, default=None,
                        help='Filter overlap files by top/bottom k (e.g., 10 or 20)')
    parser.add_argument('--test_count', type=int, default=100,
                        help='Test set size encoded in overlap filenames (default: 100)')
    parser.add_argument('--model_types', type=str, default='scratch,finetune',
                        help='Comma-separated list: scratch,finetune (default: both)')
    parser.add_argument('--font_size', type=int, default=12,
                        help='Base font size (default: 12)')
    parser.add_argument('--label_size', type=int, default=14,
                        help='Axis/legend label size (default: 14)')
    parser.add_argument('--tick_size', type=int, default=12,
                        help='Tick label size (default: 12)')
    parser.add_argument('--value_size', type=int, default=11,
                        help='Cell value label size (default: 11)')
    parser.add_argument('--cbar_labelpad', type=int, default=28,
                        help='Colorbar label padding (default: 28)')
    parser.add_argument('--y_axis_label', type=str,
                        default='Threshold for edges that may be affected by the policy (%)',
                        help='Y-axis label text')
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    overlaps_dir = os.path.join(project_root, args.overlaps_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    
    print(f"Looking for overlap files in: {overlaps_dir}")
    print(f"Output directory: {output_dir}")

    # Resolve thresholds and x_values first so we can prefer files that contain these thresholds
    x_values = parse_float_list(args.x_values) if args.x_values else X_AXIS_VALUES
    thresholds = parse_float_list(args.thresholds) if args.thresholds else THRESHOLDS

    # Find overlap files (prefer files that contain the requested thresholds, e.g. 0.5, 1.0, 1.5, 2.0)
    split_to_file = find_overlap_files(
        overlaps_dir,
        city=args.city,
        overlap_normalization=args.overlap_normalization,
        k_filter=args.k,
        test_count=args.test_count,
        required_thresholds=thresholds
    )
    
    if not split_to_file:
        print(f"No overlap files found for city: {args.city}")
        return
    
    print(f"\nFound overlap files for splits:")
    for (train, val), filepath in sorted(split_to_file.items()):
        total = SPLIT_MAPPING.get((train, val), '?')
        print(f"  Train={train}, Val={val} (Total={total}): {os.path.basename(filepath)}")

    # Create heatmaps for requested models
    model_types = [m.strip() for m in args.model_types.split(',') if m.strip()]
    for model_type in model_types:
        print(f"\nCreating heatmap for {model_type} model...")
        heatmap_matrix, threshold_labels, split_labels = create_heatmap_data(
            split_to_file, thresholds=thresholds, x_values=x_values, model_type=model_type
        )

        create_heatmap(heatmap_matrix, threshold_labels, split_labels,
                      model_type=model_type, city=args.city, output_dir=output_dir,
                      overlap_normalization=args.overlap_normalization, k_filter=args.k,
                      font_size=args.font_size, label_size=args.label_size,
                      tick_size=args.tick_size, value_size=args.value_size,
                      cbar_labelpad=args.cbar_labelpad,
                      y_axis_label=args.y_axis_label)
    
    print("\nHeatmap creation complete!")


if __name__ == "__main__":
    main()
