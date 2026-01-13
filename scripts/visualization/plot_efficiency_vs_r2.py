"""
Generate plot showing the relationship between training efficiency (train time) 
and R² metric, with visual grouping by city.

Compares Scratch vs Finetune training across 6 cities, where each city has
a distinct color and the two training methods are connected.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set Times New Roman font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Data extracted from tab:result_table_efficiency and tab:result_table_accuracy_new

# Train Time data (minutes) - mean (std)
train_time_data = {
    'C1': {'Scratch': (30.56, 9.37), 'Finetune': (26.14, 8.12)},
    'C2': {'Scratch': (39.21, 1.71), 'Finetune': (13.33, 2.99)},
    'C3': {'Scratch': (56.01, 9.37), 'Finetune': (11.66, 3.02)},
    'C4': {'Scratch': (34.70, 8.26), 'Finetune': (10.77, 3.25)},
    'C5': {'Scratch': (40.58, 5.77), 'Finetune': (15.66, 4.39)},
    'C6': {'Scratch': (44.79, 4.69), 'Finetune': (16.78, 6.28)},
}

# R² data (coefficient) - mean (std)
r2_data = {
    'C1': {'Scratch': (0.21, 0.11), 'Finetune': (0.47, 0.07)},
    'C2': {'Scratch': (0.62, 0.19), 'Finetune': (0.86, 0.05)},
    'C3': {'Scratch': (0.44, 0.06), 'Finetune': (0.51, 0.04)},
    'C4': {'Scratch': (0.59, 0.08), 'Finetune': (0.77, 0.03)},
    'C5': {'Scratch': (0.28, 0.05), 'Finetune': (0.37, 0.02)},
    'C6': {'Scratch': (0.48, 0.07), 'Finetune': (0.67, 0.04)},
}

cities = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

# Colorblind-friendly colors for training methods
# Using orange-blue scheme (safe for deuteranopia/protanopia)
method_colors = {
    'Scratch': '#D55E00',   # Vermillion (reddish-orange)
    'Finetune': '#0072B2',  # Blue
}

# Arrow color (gray to not compete with data points)
arrow_color = '#666666'

# Markers for different methods
markers = {'Scratch': 'o', 'Finetune': 's'}  # Circle for Scratch, Square for Finetune
marker_sizes = {'Scratch': 120, 'Finetune': 100}


def main():
    """Generate the R² vs Train Time plot."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Track if we've added labels for legend
    labels_added = {'Scratch': False, 'Finetune': False}
    connection_line_added = False
    
    # Plot each city
    for city in cities:
        # Extract data for scratch and finetune
        scratch_time, scratch_time_std = train_time_data[city]['Scratch']
        finetune_time, finetune_time_std = train_time_data[city]['Finetune']
        scratch_r2, scratch_r2_std = r2_data[city]['Scratch']
        finetune_r2, finetune_r2_std = r2_data[city]['Finetune']
        
        # Draw line connecting scratch to finetune for this city
        connection_label = 'City connection' if not connection_line_added else None
        ax.plot([scratch_time, finetune_time], 
               [scratch_r2, finetune_r2],
               color='gray',
               linestyle='-',
               linewidth=1.5,
               alpha=0.5,
               label=connection_label,
               zorder=1)
        connection_line_added = True
        
        # Plot scratch point (with label only once for legend)
        scratch_label = 'Scratch' if not labels_added['Scratch'] else None
        ax.errorbar(scratch_time, scratch_r2,
                   xerr=scratch_time_std, yerr=scratch_r2_std,
                   fmt=markers['Scratch'],
                   color=method_colors['Scratch'],
                   markersize=10,
                   markeredgecolor='black',
                   markeredgewidth=1.5,
                   capsize=4,
                   capthick=1.5,
                   elinewidth=1.5,
                   alpha=0.85,
                   label=scratch_label,
                   zorder=3)
        labels_added['Scratch'] = True
        
        # Plot finetune point (with label only once for legend)
        finetune_label = 'Finetune' if not labels_added['Finetune'] else None
        ax.errorbar(finetune_time, finetune_r2,
                   xerr=finetune_time_std, yerr=finetune_r2_std,
                   fmt=markers['Finetune'],
                   color=method_colors['Finetune'],
                   markersize=9,
                   markeredgecolor='black',
                   markeredgewidth=1.5,
                   capsize=4,
                   capthick=1.5,
                   elinewidth=1.5,
                   alpha=0.85,
                   label=finetune_label,
                   zorder=3)
        labels_added['Finetune'] = True
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # Method legend (Scratch vs Finetune)
    method_elements = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=method_colors['Scratch'], markeredgecolor='black',
               markeredgewidth=1.5, markersize=10, label='Scratch'),
        Line2D([0], [0], marker='s', color='w', 
               markerfacecolor=method_colors['Finetune'], markeredgecolor='black',
               markeredgewidth=1.5, markersize=9, label='Finetune'),
        Line2D([0], [0], color='gray', linestyle='-', 
               linewidth=1.5, alpha=0.5, label='City connection'),
        Line2D([0], [0], marker='', color='none', label=''),  # Empty line for spacing
        Line2D([0], [0], marker='', color='gray', linestyle='-', 
               linewidth=1.5, label='Error bars: ±1 std'),
    ]
    
    # Add legend
    legend_title = 'Training Method'
    ax.legend(handles=method_elements, 
             loc='upper right', 
             title=legend_title,
             fontsize=16,
             title_fontsize=16,
             framealpha=0.95,
             edgecolor='black',
             fancybox=True,
             shadow=True)
    
    # Formatting
    ax.set_xlabel('Train Time (min)', fontsize=16, fontweight='bold')
    ax.set_ylabel('R² ↑', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Set axis limits
    ax.set_xlim(5, 65)
    ax.set_ylim(0, 1.0)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'efficiency_vs_r2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_path}")
    
    output_path_pdf = output_dir / 'efficiency_vs_r2.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_path_pdf}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY: R² vs Training Time")
    print("="*60)
    
    for city in cities:
        scratch_time, _ = train_time_data[city]['Scratch']
        finetune_time, _ = train_time_data[city]['Finetune']
        scratch_r2, _ = r2_data[city]['Scratch']
        finetune_r2, _ = r2_data[city]['Finetune']
        
        time_savings = scratch_time - finetune_time
        time_savings_pct = (time_savings / scratch_time) * 100
        r2_improvement = finetune_r2 - scratch_r2
        r2_improvement_pct = (r2_improvement / scratch_r2) * 100
        
        print(f"\n{city}:")
        print(f"  Time: {scratch_time:.1f} → {finetune_time:.1f} min "
              f"(saved {time_savings:.1f} min, {time_savings_pct:.1f}%)")
        print(f"  R²:   {scratch_r2:.2f} → {finetune_r2:.2f} "
              f"(improved {r2_improvement:+.2f}, {r2_improvement_pct:+.1f}%)")
    
    # Overall statistics
    all_scratch_times = [train_time_data[city]['Scratch'][0] for city in cities]
    all_finetune_times = [train_time_data[city]['Finetune'][0] for city in cities]
    all_scratch_r2 = [r2_data[city]['Scratch'][0] for city in cities]
    all_finetune_r2 = [r2_data[city]['Finetune'][0] for city in cities]
    
    avg_time_savings = (np.mean(all_scratch_times) - np.mean(all_finetune_times))
    avg_time_savings_pct = (avg_time_savings / np.mean(all_scratch_times)) * 100
    avg_r2_improvement = np.mean(all_finetune_r2) - np.mean(all_scratch_r2)
    avg_r2_improvement_pct = (avg_r2_improvement / np.mean(all_scratch_r2)) * 100
    
    print("\n" + "="*60)
    print("AVERAGE ACROSS ALL CITIES:")
    print(f"  Time savings: {avg_time_savings:.1f} min ({avg_time_savings_pct:.1f}%)")
    print(f"  R² improvement: {avg_r2_improvement:+.2f} ({avg_r2_improvement_pct:+.1f}%)")
    print("="*60)


if __name__ == '__main__':
    main()

