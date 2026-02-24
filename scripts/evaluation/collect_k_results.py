#!/usr/bin/env python3
"""
Collect overlap results for different k values and N values.
"""

import json
import glob
import os
import numpy as np

# Configuration
city = 'regensburg'
test_count = 100
capacity_reduction = 1.0  # 1% of all edges

# Test configurations: (N, train_count, val_count)
configs = {
    25: (20, 5),
    50: (40, 10),
    100: (80, 20),
    200: (160, 40)
}

# k values to test
k_values = [1, 5, 10, 20]

overlaps_dir = 'scripts/evaluation/overlaps'

print("="*80)
print("COLLECTING K VARIATION RESULTS")
print("="*80)
print(f"City: {city}")
print(f"Capacity reduction threshold: {capacity_reduction}% of all edges")
print(f"Test configurations (N): {list(configs.keys())}")
print(f"k values: {k_values}")
print("="*80)

results = {}

for N, (train_count, val_count) in configs.items():
    print(f"\nN={N} (train={train_count}, val={val_count}):")
    results[N] = {}
    
    for k in k_values:
        # Find the most recent file for this configuration
        pattern = os.path.join(overlaps_dir, 
                              f'overlap_results_varying_capacity_reduction_{city}_t{train_count}_v{val_count}_test{test_count}_*.json')
        files = glob.glob(pattern)
        
        if not files:
            print(f"  k={k}: No file found")
            continue
        
        # Filter files that might have k in metadata or check all
        # We'll need to check the metadata to see which k was used
        # For now, let's assume files are named/ordered by time and check metadata
        
        # Get all files and check their metadata
        k_files = []
        for f in files:
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    file_k = data.get('metadata', {}).get('k')
                    file_percentages = data.get('metadata', {}).get('capacity_reduction_percentages', [])
                    # Check if k matches and if percentages include our target (1.0)
                    if file_k == k and (isinstance(file_percentages, list) and capacity_reduction in file_percentages):
                        k_files.append((os.path.getmtime(f), f))
            except Exception as e:
                continue
        
        if not k_files:
            print(f"  k={k}: No file with k={k} and threshold {capacity_reduction}% found")
            continue
        
        # Get the most recent file for this k
        latest_file = max(k_files, key=lambda x: x[0])[1]
        
        # Load and extract results
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Extract overlap values for capacity_reduction = 1.0%
        results_data = data.get('results', {})
        threshold_key = str(capacity_reduction)
        
        if threshold_key in results_data:
            seed_results = results_data[threshold_key]
            scratch_overlaps = []
            finetune_overlaps = []
            
            for seed_idx in sorted(seed_results.keys(), key=int):
                seed_data = seed_results[seed_idx]
                if seed_data is not None:
                    scratch_overlaps.append(seed_data.get('scratch_overlap'))
                    finetune_overlaps.append(seed_data.get('finetune_overlap'))
            
            if scratch_overlaps and finetune_overlaps:
                results[N][k] = {
                    'scratch_mean': float(np.mean(scratch_overlaps)),
                    'scratch_std': float(np.std(scratch_overlaps)),
                    'finetune_mean': float(np.mean(finetune_overlaps)),
                    'finetune_std': float(np.std(finetune_overlaps)),
                    'scratch_values': scratch_overlaps,
                    'finetune_values': finetune_overlaps
                }
                print(f"  k={k}: Scratch={results[N][k]['scratch_mean']:.3f}±{results[N][k]['scratch_std']:.3f}, "
                      f"Finetune={results[N][k]['finetune_mean']:.3f}±{results[N][k]['finetune_std']:.3f}")
            else:
                print(f"  k={k}: No valid results")
        else:
            print(f"  k={k}: No results for threshold {capacity_reduction}%")

# Print summary table
print("\n" + "="*80)
print("SUMMARY RESULTS")
print("="*80)
print(f"\nCapacity reduction threshold: {capacity_reduction}% of all edges")
print(f"\n{'N':<6} {'k':<6} {'Scratch Mean±Std':<20} {'Finetune Mean±Std':<20}")
print("-"*80)

for N in sorted(results.keys()):
    for k in sorted(results[N].keys()):
        r = results[N][k]
        print(f"{N:<6} {k:<6} {r['scratch_mean']:.3f}±{r['scratch_std']:.3f}     {r['finetune_mean']:.3f}±{r['finetune_std']:.3f}")

print("\n" + "="*80)
print("DETAILED RESULTS (all seed values)")
print("="*80)

for N in sorted(results.keys()):
    print(f"\nN={N}:")
    for k in sorted(results[N].keys()):
        r = results[N][k]
        print(f"  k={k}:")
        print(f"    Scratch: {r['scratch_values']} (mean={r['scratch_mean']:.3f}, std={r['scratch_std']:.3f})")
        print(f"    Finetune: {r['finetune_values']} (mean={r['finetune_mean']:.3f}, std={r['finetune_std']:.3f})")
