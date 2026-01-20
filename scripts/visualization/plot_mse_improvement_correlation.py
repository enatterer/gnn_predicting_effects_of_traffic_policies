#!/usr/bin/env python3
"""
Plot correlation between Scratch MSE and MSE improvement from fine-tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set up matplotlib style with Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

# Professional colorblind-friendly palette (Tol colorblind-safe scheme)
colorblind_colors = [
    '#4477AA',  # Blue
    '#EE6677',  # Red/Pink
    '#228833',  # Green
    '#CCBB44',  # Yellow
    '#66CCEE',  # Cyan
    '#AA3377',  # Purple
]

# Different marker shapes for additional distinction
markers = ['o', 's', '^', 'D', 'v', 'p']

# Data from the table
cities = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
scratch_mse = np.array([140.98, 124.61, 23.26, 34.45, 67.26, 28.46])
finetune_mse = np.array([51.46, 49.77, 16.13, 18.23, 38.24, 21.24])
mse_improvement = scratch_mse - finetune_mse

# Calculate correlation
pearson_r, pearson_p = stats.pearsonr(scratch_mse, mse_improvement)
r_squared = pearson_r ** 2

# Fit linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(scratch_mse, mse_improvement)
line_x = np.linspace(scratch_mse.min() * 0.9, scratch_mse.max() * 1.1, 100)
line_y = slope * line_x + intercept

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot with city labels using colorblind-friendly colors and different markers
for i, city in enumerate(cities):
    ax.scatter(scratch_mse[i], mse_improvement[i], 
              s=150, c=colorblind_colors[i], marker=markers[i],
              edgecolors='black', linewidths=1.5,
              alpha=0.9, zorder=3, label=city)

# Regression line
ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.7, 
        label='Linear best fit', zorder=2)

# Add text box with correlation statistics
textstr = f'Pearson r = {pearson_r:.4f}\n$R^2$ = {r_squared:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', bbox=props)

# Labels (no title)
ax.set_xlabel('Scratch MSE ((veh/day)$^2$)', fontweight='bold')
ax.set_ylabel('MSE Improvement (Scratch - Finetune)\n((veh/day)$^2$)', fontweight='bold')

# Grid
ax.grid(True, alpha=0.3, linestyle=':', zorder=1)

# Legend
ax.legend(loc='lower right', framealpha=0.9, edgecolor='black')

# Tight layout
plt.tight_layout()

# Save figure
# Save to results folder at the same level as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'results')
os.makedirs(output_dir, exist_ok=True)

output_path_pdf = os.path.join(output_dir, 'mse_improvement_correlation.pdf')
output_path_png = os.path.join(output_dir, 'mse_improvement_correlation.png')

plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')

print(f"Saved figure to:")
print(f"  - {output_path_pdf}")
print(f"  - {output_path_png}")

plt.show()
