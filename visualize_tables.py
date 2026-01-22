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
    vmax = 17.0

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
    output_dir = Path(__file__).parent / 'output'
    json_files = sorted(output_dir.glob('*.json'))

    if not json_files:
        print(f'No JSON files found in {output_dir}')
        return

    for json_path in json_files:
        process_json(json_path)


if __name__ == '__main__':
    main()
