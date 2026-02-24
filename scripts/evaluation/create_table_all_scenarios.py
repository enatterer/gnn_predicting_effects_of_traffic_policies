#!/usr/bin/env python3
"""
Create LaTeX table from overlap results using ALL scenarios as test set.
"""

import json
import glob
import os
import numpy as np

city = 'regensburg'
capacity_reduction = 1.0

configs = {
    25: (20, 5),
    50: (40, 10),
    100: (80, 20),
    200: (160, 40)
}

k_values = [5, 10, 20, 50, 100]  # Top-5, Top-10, Top-20, Top-50, Top-100

# Find all result files
files_by_config = {}
for file in glob.glob(f'scripts/evaluation/overlaps/overlap_results_all_scenarios_{city}_*.json'):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        meta = data.get('metadata', {})
        train = meta.get('train_count')
        val = meta.get('val_count')
        k = meta.get('k')
        test_count = meta.get('test_count')
        pcts = meta.get('capacity_reduction_percentages', [])
        filter_mode = meta.get('filter_mode', '')
        
        if (train and val and k and isinstance(pcts, list) and 
            capacity_reduction in pcts and filter_mode == 'upper_bound' and
            test_count == 'all'):
            key = (train, val, k)
            if key not in files_by_config or os.path.getmtime(file) > os.path.getmtime(files_by_config[key]):
                files_by_config[key] = file
    except:
        pass

# Extract results
results = {}
for N, (train, val) in configs.items():
    results[N] = {}
    for k in k_values:
        key = (train, val, k)
        if key in files_by_config:
            with open(files_by_config[key], 'r') as f:
                data = json.load(f)
            
            threshold_key = str(capacity_reduction)
            if threshold_key in data.get('results', {}):
                seed_results = data['results'][threshold_key]
                finetune_vals = []
                
                for seed_idx in sorted(seed_results.keys(), key=int):
                    sd = seed_results[seed_idx]
                    if sd:
                        finetune_vals.append(sd.get('finetune_overlap'))
                
                if finetune_vals:
                    finetune_mean = float(np.mean(finetune_vals))
                    finetune_std = float(np.std(finetune_vals))
                    results[N][k] = {
                        'mean': finetune_mean,
                        'std': finetune_std
                    }

# Create LaTeX table
table_content = """\\begin{table}[H]
\\centering
\\caption{Top-$x$ overlap (mean $\\pm$ std over five seeds) between predicted and true simulation-based Top-$x$ intervention sets, evaluated under the feasibility constraint that interventions affect at most 1\\% of edges. Both models are trained with $N$ simulation-labeled scenarios (training budget). Results shown for a representative city (C1), evaluated on all available scenarios.}
\\label{tab:top_x_scenario_overlap_all}
% \\resizebox{\\linewidth}{!}{
\\begin{tabular}{lcccc}
\\toprule
& \\multicolumn{4}{c}{$N$ (training budget)} \\\\
\\cmidrule(lr){2-5}
Metric & $N=25$ & $N=50$ & $N=100$ & $N=200$ \\\\
\\midrule
"""

# Add rows for each k value
for k in k_values:
    row = f"Top-{k}"
    for N in [25, 50, 100, 200]:
        if N in results and k in results[N]:
            mean = results[N][k]['mean']
            std = results[N][k]['std']
            # Format: round to 2 decimal places
            mean_str = f"{mean:.2f}"
            std_str = f"{std:.2f}"
            row += f" & ${mean_str} \\pm {std_str}$"
        else:
            row += " & --"
    row += " \\\\\n"
    table_content += row

table_content += """% \\midrule
% Avg. (Top-5/10/20/50/100)
% & $0.80 \\pm 0.10$ &  & $0.63 \\pm 0.18$ &  & $0.71 \\pm 0.18$ &  \\\\
\\bottomrule
\\end{tabular}
% }
\\end{table}
"""

# Save table
output_file = 'scripts/evaluation/overlap_table_all_scenarios.tex'
with open(output_file, 'w') as f:
    f.write(table_content)

print("="*80)
print("TABLE CREATED")
print("="*80)
print(f"Saved to: {output_file}")
print(f"\nConfigurations found: {sum(1 for N in results for k in results[N])} / {len(configs) * len(k_values)}")
print("\nMissing configurations:")
for N, (train, val) in configs.items():
    for k in k_values:
        key = (train, val, k)
        if key not in files_by_config:
            print(f"  N={N}, k={k}")

if sum(1 for N in results for k in results[N]) == len(configs) * len(k_values):
    print("\n✓ All configurations complete!")
    print("\nTable preview:")
    print(table_content)
