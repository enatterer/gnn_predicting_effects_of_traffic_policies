#!/usr/bin/env python3
"""
Test how predictions change as k varies for different target-city budgets.
Tests k=1,5,10,20 for N=25,50,100,200 with capacity reduction <= 1% of all edges.
"""

import subprocess
import json
import os
import glob
import numpy as np

# Configuration
city = 'regensburg'
test_count = 100
capacity_reduction = 1.0  # 1% of all edges

# Test configurations: (N, train_count, val_count)
configs = [
    (25, 20, 5),
    (50, 40, 10),
    (100, 80, 20),
    (200, 160, 40)
]

# k values to test
k_values = [1, 5, 10, 20]

script_path = 'scripts/evaluation/compute_overlaps_varying_capacity_reduction.py'
overlaps_dir = 'scripts/evaluation/overlaps'

print("="*80)
print("TESTING K VARIATION FOR DIFFERENT TARGET-CITY BUDGETS")
print("="*80)
print(f"City: {city}")
print(f"Capacity reduction threshold: {capacity_reduction}% of all edges")
print(f"Test configurations: {[N for N, _, _ in configs]}")
print(f"k values: {k_values}")
print("="*80)

results = {}

for N, train_count, val_count in configs:
    print(f"\n{'='*80}")
    print(f"Testing N={N} (train={train_count}, val={val_count})")
    print(f"{'='*80}")
    
    results[N] = {}
    
    for k in k_values:
        print(f"\nRunning with k={k}...")
        
        # Run the script
        cmd = [
            'python', script_path,
            '--city', city,
            '--train_count', str(train_count),
            '--val_count', str(val_count),
            '--test_count', str(test_count),
            '--cap_red_on_all_roads',
            '--percentages', str(capacity_reduction),
            '--k', str(k)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        if result.returncode != 0:
            print(f"ERROR running k={k}: {result.stderr}")
            continue
        
        # Find the most recent output file
        pattern = os.path.join(overlaps_dir, 
                              f'overlap_results_varying_capacity_reduction_{city}_t{train_count}_v{val_count}_test{test_count}_*.json')
        files = glob.glob(pattern)
        
        if not files:
            print(f"WARNING: No output file found for k={k}")
            continue
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getmtime)
        
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
