#!/usr/bin/env -S uv run

"""
Visualize spine simulation results as colored tables.
Shows T12L1 kN values (with peak G in brackets) across drop rates and jerk limits.

Processes all JSON files in the output/ folder and generates PNG images.
"""

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def load_data(json_path):
    """Load JSON and organize into a dict keyed by (drop_rate, jerk_limit)."""
    with open(json_path) as f:
        data = json.load(f)

    result = {}
    bottom_out_force_kN = None
    bottom_out_compression = None

    for entry in data:
        # Parse filename: e.g., "1.0-10-1000.csv"
        parts = entry['file'].replace('.csv', '').split('-')
        drop_rate = float(parts[0])
        jerk_limit = int(parts[2])

        result[(drop_rate, jerk_limit)] = {
            'kN': entry['peak_T12L1_kN'],
            'peak_g': entry['base_accel_peak_g'],
        }

        # Capture bottom_out values from first entry
        if bottom_out_force_kN is None and 'buttocks' in entry:
            bottom_out_force_kN = entry['buttocks'].get('bottom_out_force_kN')
        if bottom_out_compression is None and 'bottom_out_compression_mm' in entry:
            bottom_out_compression = entry['bottom_out_compression_mm']

    return result, bottom_out_force_kN, bottom_out_compression


def process_universal(output_dir):
    """Process all JSON files and create a universal comparison table.

    Rows: input CSV files (impactrate-maxg-jerk.csv)
    Columns: configs grouped by bottom-out type, with gaps between groups
    """
    # Discover all JSON files and parse their configs
    json_files = list(output_dir.glob('*.json'))
    configs_info = []  # [(config_name, stiffness, bottom_out), ...]

    for jf in json_files:
        name = jf.stem  # e.g., "85-unlimited" or "305-7"
        parts = name.split('-')
        if len(parts) == 2:
            stiffness = int(parts[0])
            bottom_out = parts[1]
            configs_info.append((name, stiffness, bottom_out))

    # Group by bottom_out type: "unlimited" first, then numeric values sorted
    def bottom_out_sort_key(bo):
        if bo == 'unlimited':
            return (0, 0)
        return (1, int(bo))

    # Get unique bottom_out values, sorted
    bottom_out_types = sorted(set(c[2] for c in configs_info), key=bottom_out_sort_key)

    # Build groups: each group is a list of (config_name, stiffness) sorted by stiffness
    groups = []
    for bo in bottom_out_types:
        group_configs = [(c[0], c[1]) for c in configs_info if c[2] == bo]
        group_configs.sort(key=lambda x: x[1])  # sort by stiffness
        groups.append((bo, group_configs))

    # Load all data from all JSON files
    all_data = {}  # config -> {filename -> entry}
    all_files = set()

    for config_name, _, _ in configs_info:
        json_path = output_dir / f'{config_name}.json'
        with open(json_path) as f:
            data = json.load(f)

        all_data[config_name] = {}
        for entry in data:
            filename = entry['file']
            all_files.add(filename)
            all_data[config_name][filename] = entry

    # Sort files by (drop_rate, max_g, jerk)
    def parse_filename(f):
        parts = f.replace('.csv', '').split('-')
        return (float(parts[0]), int(parts[1]), int(parts[2]))

    sorted_files = sorted(all_files, key=parse_filename)
    n_rows = len(sorted_files)

    # Color scale settings
    threshold_caution = 8.0
    threshold_danger = 10.0

    # Compute global vmin across all data
    all_values = []
    for config_name in all_data:
        for filename in all_data[config_name]:
            all_values.append(all_data[config_name][filename]['peak_T12L1_kN'])
    vmin = min(all_values) if all_values else 0
    vmax = 16.0

    # Build colormap
    norm_caution = (threshold_caution - vmin) / (vmax - vmin)
    norm_danger = (threshold_danger - vmin) / (vmax - vmin)
    norm_caution = max(0.0, min(1.0, norm_caution))
    norm_danger = max(0.0, min(1.0, norm_danger))

    colors = []
    positions = []
    colors.extend([
        (0.2, 0.4, 0.8),
        (0.2, 0.7, 0.7),
        (0.3, 0.8, 0.4),
    ])
    positions.extend([0.0, norm_caution * 0.5, norm_caution])
    colors.extend([
        (0.95, 0.9, 0.3),
        (0.95, 0.6, 0.2),
    ])
    positions.extend([norm_caution + 0.001, norm_danger])
    colors.extend([
        (0.9, 0.2, 0.2),
        (0.7, 0.1, 0.3),
        (0.6, 0.1, 0.6),
        (0.5, 0.0, 0.8),
    ])
    positions.extend([
        norm_danger + 0.001,
        norm_danger + (1.0 - norm_danger) * 0.33,
        norm_danger + (1.0 - norm_danger) * 0.66,
        1.0,
    ])
    cmap = mcolors.LinearSegmentedColormap.from_list('danger', list(zip(positions, colors)))

    # Create figure with gridspec for groups with gaps
    n_groups = len(groups)
    group_sizes = [len(g[1]) for g in groups]
    total_cols = sum(group_sizes)

    fig_height = max(10, 2 + n_rows * 0.6)
    fig_width = 4 + total_cols * 1.5

    # Width ratios: columns for each group, with small gaps between groups
    width_ratios = []
    for i, size in enumerate(group_sizes):
        width_ratios.extend([1] * size)
        if i < n_groups - 1:
            width_ratios.append(0.3)  # gap

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(1, len(width_ratios), figure=fig, width_ratios=width_ratios,
                  left=0.15, right=0.95, top=0.95, bottom=0.12, wspace=0.05)

    # Track column index across groups
    col_idx = 0
    axes = []

    for group_idx, (bo, group_configs) in enumerate(groups):
        n_cols = len(group_configs)

        # Create axis for this group
        ax = fig.add_subplot(gs[0, col_idx:col_idx + n_cols])
        axes.append(ax)

        # Build matrix for this group
        matrix = np.full((n_rows, n_cols), np.nan)
        data_values = {}

        for i, filename in enumerate(sorted_files):
            for j, (config_name, stiffness) in enumerate(group_configs):
                if config_name in all_data and filename in all_data[config_name]:
                    entry = all_data[config_name][filename]
                    matrix[i, j] = entry['peak_T12L1_kN']
                    data_values[(i, j)] = {
                        'kN': entry['peak_T12L1_kN'],
                        'peak_g': entry['base_accel_peak_g'],
                    }

        # Create heatmap
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Column labels
        col_labels = [str(stiffness) for _, stiffness in group_configs]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=10)

        # Group title (same format as grid mode)
        if bo == 'unlimited':
            group_title = 'Buttock tissue\nno bottoming out limit'
        else:
            group_title = f'Buttock tissue\nbottoms out at {bo}kN'
        ax.set_xlabel(group_title, fontsize=11, fontweight='bold')

        # Row labels only on first group
        if group_idx == 0:
            row_labels = [f.replace('.csv', '') for f in sorted_files]
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(row_labels, fontsize=8)
        else:
            ax.set_yticks([])

        # Add text annotations
        for i in range(n_rows):
            for j in range(n_cols):
                if (i, j) in data_values:
                    kn = data_values[(i, j)]['kN']

                    if kn >= threshold_danger:
                        text_color = 'white'
                    elif kn >= threshold_caution:
                        text_color = 'black'
                    else:
                        norm_in_zone = (kn - vmin) / (threshold_caution - vmin) if threshold_caution > vmin else 0
                        text_color = 'white' if norm_in_zone < 0.4 else 'black'

                    ax.text(j, i, f'{kn:.2f}', ha='center', va='center',
                            fontsize=8, color=text_color, fontweight='bold')
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='gray')

        # Move to next group (skip gap column)
        col_idx += n_cols
        if group_idx < n_groups - 1:
            col_idx += 1  # skip gap

    # Save
    output_path = output_dir / 'universal.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to: {output_path}')
    plt.close()


def process_json(json_path):
    """Process a single JSON file and generate a PNG."""
    # Output PNG with same base name
    output_path = json_path.with_suffix('.png')

    # Load data
    data, bottom_out_force_kN, bottom_out_compression_mm = load_data(json_path)

    # Get all unique drop rates and jerk limits
    drop_rates = sorted({k[0] for k in data.keys()})
    jerk_limits = sorted({k[1] for k in data.keys()})

    # Create matrix for the heatmap
    matrix = np.full((len(drop_rates), len(jerk_limits)), np.nan)
    for i, dr in enumerate(drop_rates):
        for j, jl in enumerate(jerk_limits):
            if (dr, jl) in data:
                matrix[i, j] = data[(dr, jl)]['kN']

    # Fixed color scale: min from data, max fixed at 17
    vmin = np.nanmin(matrix)
    vmax = 16.0

    # Create figure with fixed size and layout
    fig = plt.figure(figsize=(8, 9))

    # Fixed layout: title area at top, table below
    # Using fixed axes positions to ensure consistent table placement
    title_height = 0.12  # Fixed height for title area
    ax = fig.add_axes([0.12, 0.08, 0.75, 0.75])  # [left, bottom, width, height]

    # Custom colormap with gradients:
    # < 8 kN: blue -> cyan -> green (safe)
    # 8-10 kN: yellow -> orange (caution)
    # > 10 kN: red -> dark red -> crazy purple (danger)
    threshold_caution = 8.0
    threshold_danger = 10.0

    # Normalize threshold positions to [0, 1] range
    norm_caution = (threshold_caution - vmin) / (vmax - vmin)
    norm_danger = (threshold_danger - vmin) / (vmax - vmin)

    # Clamp to valid range
    norm_caution = max(0.0, min(1.0, norm_caution))
    norm_danger = max(0.0, min(1.0, norm_danger))

    # Build gradient with multiple stops in each zone
    colors = []
    positions = []

    # Zone 1: Blue-green gradient (vmin to 8)
    colors.extend(
        [
            (0.2, 0.4, 0.8),  # blue
            (0.2, 0.7, 0.7),  # cyan
            (0.3, 0.8, 0.4),  # green
        ]
    )
    positions.extend(
        [
            0.0,
            norm_caution * 0.5,
            norm_caution,
        ]
    )

    # Zone 2: Yellow-orange gradient (8 to 10)
    colors.extend(
        [
            (0.95, 0.9, 0.3),  # yellow
            (0.95, 0.6, 0.2),  # orange
        ]
    )
    positions.extend(
        [
            norm_caution + 0.001,
            norm_danger,
        ]
    )

    # Zone 3: Red to crazy purple gradient (10 to vmax)
    colors.extend(
        [
            (0.9, 0.2, 0.2),  # red
            (0.7, 0.1, 0.3),  # dark red/crimson
            (0.6, 0.1, 0.6),  # purple
            (0.5, 0.0, 0.8),  # crazy purple/violet
        ]
    )
    positions.extend(
        [
            norm_danger + 0.001,
            norm_danger + (1.0 - norm_danger) * 0.33,
            norm_danger + (1.0 - norm_danger) * 0.66,
            1.0,
        ]
    )

    cmap = mcolors.LinearSegmentedColormap.from_list('danger', list(zip(positions, colors)))

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

    # Build title based on bottom_out_force_kN
    if bottom_out_force_kN is not None and bottom_out_force_kN >= 9999:
        title = 'Buttock tissue\nno bottoming out limit'
    elif bottom_out_force_kN is not None:
        if bottom_out_compression_mm is not None:
            title = f'Buttock tissue\nbottoms out at {bottom_out_force_kN:.0f}kN (~{bottom_out_compression_mm:.0f}mm compression)'
        else:
            title = f'Buttock tissue\nbottoms out at {bottom_out_force_kN:.0f}kN'
    else:
        title = 'Buttock tissue'

    # Title with fixed position
    fig.text(0.5, 0.92, title, ha='center', va='center', fontsize=14, fontweight='bold')

    # Add text annotations
    for i, dr in enumerate(drop_rates):
        for j, jl in enumerate(jerk_limits):
            if (dr, jl) in data:
                kn = data[(dr, jl)]['kN']
                peak_g = data[(dr, jl)]['peak_g']

                # Determine text color based on kN value
                if kn >= threshold_danger:
                    text_color = 'white'
                elif kn >= threshold_caution:
                    text_color = 'black'
                else:
                    # Blue-green zone: white for darker blues, black for lighter greens
                    norm_in_zone = (
                        (kn - vmin) / (threshold_caution - vmin)
                        if threshold_caution > vmin
                        else 0
                    )
                    text_color = 'white' if norm_in_zone < 0.4 else 'black'

                ax.text(
                    j,
                    i,
                    f'{kn:.2f}kN\n({peak_g:.0f}G)',
                    ha='center',
                    va='center',
                    fontsize=9,
                    color=text_color,
                    fontweight='bold',
                )
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=9, color='gray')

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to: {output_path}')
    plt.close()


def main():
    import sys

    output_dir = Path(__file__).parent / 'output'
    json_files = sorted(output_dir.glob('*.json'))

    if not json_files:
        print(f'No JSON files found in {output_dir}')
        return

    if '--grid' in sys.argv:
        # Original mode: generate individual tables per JSON
        for json_path in json_files:
            process_json(json_path)
    else:
        # Default: universal comparison table
        process_universal(output_dir)


if __name__ == '__main__':
    main()
