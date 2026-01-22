#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
Visualize spine simulation results as colored tables.
Shows T12L1 kN values (with peak G in brackets) across drop rates and jerk limits.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


def load_data(json_path):
    """Load JSON and organize into a dict keyed by (drop_rate, jerk_limit)."""
    with open(json_path) as f:
        data = json.load(f)

    result = {}
    for entry in data:
        # Parse filename: e.g., "1.0-10-1000.csv"
        parts = entry['file'].replace('.csv', '').split('-')
        drop_rate = float(parts[0])
        jerk_limit = int(parts[2])

        result[(drop_rate, jerk_limit)] = {
            'kN': entry['peak_T12L1_kN'],
            'peak_g': entry['base_accel_peak_g']
        }

    return result


def main():
    output_dir = Path(__file__).parent / 'output'

    # Load both datasets
    unlimited_data = load_data(output_dir / 'unlimited.json')
    fixed_data = load_data(output_dir / 'fixed.json')

    # Get all unique drop rates and jerk limits
    all_keys = set(unlimited_data.keys()) | set(fixed_data.keys())
    drop_rates = sorted(set(k[0] for k in all_keys))
    jerk_limits = sorted(set(k[1] for k in all_keys))

    # Create matrices for the heatmaps
    def create_matrix(data):
        matrix = np.full((len(drop_rates), len(jerk_limits)), np.nan)
        for i, dr in enumerate(drop_rates):
            for j, jl in enumerate(jerk_limits):
                if (dr, jl) in data:
                    matrix[i, j] = data[(dr, jl)]['kN']
        return matrix

    unlimited_matrix = create_matrix(unlimited_data)
    fixed_matrix = create_matrix(fixed_data)

    # Find global min/max for consistent colormap
    all_values = []
    for m in [unlimited_matrix, fixed_matrix]:
        all_values.extend(m[~np.isnan(m)].flatten())
    vmin, vmax = min(all_values), max(all_values)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Custom colormap: green below 8, transition to red/orange above 8
    # Create a diverging colormap centered around 8 kN (danger threshold)
    danger_threshold = 8.0
    # Normalize so that 8 kN maps to ~0.5 in the colormap
    norm_center = (danger_threshold - vmin) / (vmax - vmin)

    # Create custom colormap: green -> yellow -> orange -> red
    colors = [
        (0.2, 0.7, 0.3),    # green (safe)
        (0.5, 0.8, 0.3),    # yellow-green
        (0.95, 0.9, 0.3),   # yellow (threshold)
        (0.95, 0.6, 0.2),   # orange
        (0.85, 0.2, 0.2),   # red (danger)
    ]
    # Position colors so yellow is at the danger threshold
    positions = [0.0, norm_center * 0.5, norm_center, norm_center + (1 - norm_center) * 0.5, 1.0]
    cmap = mcolors.LinearSegmentedColormap.from_list('danger', list(zip(positions, colors)))

    def plot_table(ax, matrix, data, title):
        # Create heatmap
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Set ticks
        ax.set_xticks(range(len(jerk_limits)))
        ax.set_xticklabels([str(jl) for jl in jerk_limits])
        ax.set_yticks(range(len(drop_rates)))
        ax.set_yticklabels([f'{dr:.0f}' for dr in drop_rates])

        # Labels
        ax.set_xlabel('Jerk Limit (m/s³)', fontsize=12)
        ax.set_ylabel('Drop Rate (m/s)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add text annotations
        for i, dr in enumerate(drop_rates):
            for j, jl in enumerate(jerk_limits):
                if (dr, jl) in data:
                    kn = data[(dr, jl)]['kN']
                    peak_g = data[(dr, jl)]['peak_g']

                    # Determine text color based on background brightness
                    norm_val = (kn - vmin) / (vmax - vmin)
                    text_color = 'white' if norm_val > 0.6 or norm_val < 0.3 else 'black'

                    ax.text(j, i, f'{kn:.2f}kN\n({peak_g:.0f}G)',
                            ha='center', va='center', fontsize=9,
                            color=text_color, fontweight='bold')
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center',
                            fontsize=9, color='gray')

    # Plot both tables
    plot_table(ax1, unlimited_matrix, unlimited_data,
               'Buttock tissue\nno bottoming out limit')
    plot_table(ax2, fixed_matrix, fixed_data,
               'Buttock tissue\nbottoms out at 7kN (~38mm compression)')

    # Save only (no display)
    output_path = output_dir / 'comparison_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to: {output_path}')
    plt.close()


if __name__ == '__main__':
    main()
