#!/usr/bin/env -S uv run

"""
Visualize spine simulation results as colored tables.
Shows T12L1 kN values (with peak G in brackets) across drop rates and jerk limits.

Processes all CSV files in the output/ folder and generates PNG images.
"""

import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Color scale constants
VMIN = 2.0
VMAX = 14.0


def create_colormap():
    """Create smooth rainbow-like colormap for spine force visualization.

    Progression: deep blue → blue → cyan → teal → green → yellow → orange
    Then above 10 kN: red → magenta → purple → dark purple
    """
    def kn_to_norm(kn):
        return (kn - VMIN) / (VMAX - VMIN)

    colors = [
        (0.18, 0.20, 0.58),  # 2 kN: Deep indigo-blue
        (0.20, 0.36, 0.72),  # 3 kN: Rich blue
        (0.16, 0.50, 0.72),  # 4 kN: Blue-cyan
        (0.12, 0.62, 0.67),  # 5 kN: Cyan-teal
        (0.20, 0.68, 0.48),  # 6 kN: Teal-green
        (0.70, 0.80, 0.28),  # 7 kN: Yellow-green
        (0.92, 0.70, 0.20),  # 8 kN: Orange-yellow
        (0.92, 0.45, 0.18),  # 9 kN: Orange
        (0.90, 0.22, 0.18),  # 10 kN: True red
        (0.82, 0.18, 0.45),  # 11 kN: Red-magenta
        (0.65, 0.12, 0.55),  # 12 kN: Magenta-purple
        (0.45, 0.08, 0.50),  # 13 kN: Purple
        (0.30, 0.05, 0.40),  # 14 kN: Dark purple
    ]
    positions = [kn_to_norm(kn) for kn in range(2, 15)]
    return mcolors.LinearSegmentedColormap.from_list('spine_rainbow', list(zip(positions, colors)))


def get_text_color(kn):
    """Return text color (white/black) based on background brightness."""
    if kn <= 4.0 or kn >= 10.0:
        return 'white'
    return 'black'


# Shared colormap instance
CMAP = create_colormap()


def load_data(csv_path):
    """Load CSV and organize into a dict keyed by (drop_rate, jerk_limit)."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    result = {}
    bottom_out_force_kN = None
    bottom_out_compression = None

    for entry in data:
        parts = entry['filename'].replace('.csv', '').split('-')
        drop_rate = float(parts[0])
        jerk_limit = int(parts[2])

        result[(drop_rate, jerk_limit)] = {
            'kN': float(entry['peak_T12L1_kN']),
            'peak_g': float(entry['base_accel_peak_g']),
        }

        if bottom_out_force_kN is None and 'buttocks_bottom_out_force_kN' in entry:
            bottom_out_force_kN = float(entry['buttocks_bottom_out_force_kN'])
        if bottom_out_compression is None and 'buttocks_bottom_out_limit_mm' in entry:
            bottom_out_compression = float(entry['buttocks_bottom_out_limit_mm'])

    return result, bottom_out_force_kN, bottom_out_compression


def process_universal(output_dir):
    """Process all CSV files and create a universal comparison table."""
    from matplotlib.patches import Rectangle

    batch_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    configs_info = []

    for batch_dir in batch_dirs:
        name = batch_dir.name
        parts = name.split('-')
        if len(parts) == 2 and parts[0].isdigit():
            stiffness = int(parts[0])
            bottom_out = parts[1]
            csv_path = batch_dir / f'{name}.csv'
            if csv_path.exists():
                configs_info.append((name, stiffness, bottom_out, csv_path))

    if not configs_info:
        print('  No batch config folders found')
        return

    def bottom_out_sort_key(bo):
        return (0, 0) if bo == 'unlimited' else (1, int(bo))

    bottom_out_types = sorted({c[2] for c in configs_info}, key=bottom_out_sort_key)

    groups = []
    for bo in bottom_out_types:
        group_configs = sorted([(c[0], c[1]) for c in configs_info if c[2] == bo], key=lambda x: x[1])
        groups.append((bo, group_configs))

    all_data = {}
    all_files = set()

    for config_name, _, _, csv_path in configs_info:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)

        all_data[config_name] = {}
        for entry in data:
            filename = entry['filename']
            all_files.add(filename)
            all_data[config_name][filename] = {
                'peak_T12L1_kN': float(entry['peak_T12L1_kN']),
                'base_accel_peak_g': float(entry['base_accel_peak_g']),
            }

    def parse_filename(f):
        parts = f.replace('.csv', '').split('-')
        try:
            if len(parts) == 3:
                return (0, float(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            pass
        return (1, f, 0, 0)

    sorted_files = sorted(all_files, key=parse_filename)
    is_grid_mode = all(parse_filename(f)[0] == 0 for f in all_files)

    # Grid mode requires multiple distinct drop rates (first value) to form a 2D grid
    if is_grid_mode:
        drop_rates = {parse_filename(f)[1] for f in all_files}
        if len(drop_rates) <= 1:
            is_grid_mode = False

    print(f"  Mode: {'Grid' if is_grid_mode else 'List'} ({len(sorted_files)} files)")
    n_rows = len(sorted_files)

    # In list mode with exactly 2 groups, create a single merged table
    if not is_grid_mode and len(groups) == 2:
        # Get stiffness values (columns) - use first group's configs
        stiffness_values = sorted({c[1] for c in configs_info})
        n_cols = len(stiffness_values)

        # Build lookup: (bottom_out, stiffness) -> config_name
        config_lookup = {}
        for name, stiffness, bottom_out, _ in configs_info:
            config_lookup[(bottom_out, stiffness)] = name

        # Fixed dimensions (in inches)
        cell_size = 0.55  # Each half-cell is square
        gap_size = 0.25   # Gap between stiffness groups
        left_margin = 1.5  # Fixed space for row labels
        right_margin = 0.3
        top_margin = 0.8   # Space for two-line title
        bottom_margin = 0.5  # Space for stiffness label

        # Calculate figure size
        table_width = n_cols * (2 * cell_size) + (n_cols - 1) * gap_size
        table_height = n_rows * cell_size
        fig_width = left_margin + table_width + right_margin
        fig_height = top_margin + table_height + bottom_margin

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Set axes position to match fixed margins
        ax_left = left_margin / fig_width
        ax_bottom = bottom_margin / fig_height
        ax_width = table_width / fig_width
        ax_height = table_height / fig_height
        ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

        # Data coordinates: each column group is 2 units wide (1 per half), gaps between
        x_positions = []
        for j in range(n_cols):
            x_positions.append(j * (2 + gap_size / cell_size))

        ax.set_xlim(-1, x_positions[-1] + 1)
        ax.set_ylim(n_rows, 0)

        # Draw split cells with colored backgrounds
        norm = plt.Normalize(vmin=VMIN, vmax=VMAX)
        bo_left, bo_right = bottom_out_types[0], bottom_out_types[1]

        for i, filename in enumerate(sorted_files):
            for j, stiffness in enumerate(stiffness_values):
                x = x_positions[j]

                # Get values for both groups
                left_config = config_lookup.get((bo_left, stiffness))
                right_config = config_lookup.get((bo_right, stiffness))

                left_kn, left_g = None, None
                right_kn, right_g = None, None
                if left_config and left_config in all_data and filename in all_data[left_config]:
                    left_kn = all_data[left_config][filename]['peak_T12L1_kN']
                    left_g = all_data[left_config][filename]['base_accel_peak_g']
                if right_config and right_config in all_data and filename in all_data[right_config]:
                    right_kn = all_data[right_config][filename]['peak_T12L1_kN']
                    right_g = all_data[right_config][filename]['base_accel_peak_g']

                # Draw left half (square)
                left_color = CMAP(norm(left_kn)) if left_kn is not None else (0.9, 0.9, 0.9, 1)
                rect_left = Rectangle((x - 1, i), 1, 1, facecolor=left_color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect_left)

                # Draw right half (square)
                right_color = CMAP(norm(right_kn)) if right_kn is not None else (0.9, 0.9, 0.9, 1)
                rect_right = Rectangle((x, i), 1, 1, facecolor=right_color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect_right)

                # Add text for left half: kN on line 1, (G) on line 2
                if left_kn is not None:
                    text_color = get_text_color(left_kn)
                    ax.text(x - 0.5, i + 0.35, f'{left_kn:.1f} kN', ha='center', va='center',
                            fontsize=7, color=text_color, fontweight='bold')
                    ax.text(x - 0.5, i + 0.65, f'({left_g:.0f} G)', ha='center', va='center',
                            fontsize=6, color=text_color)
                else:
                    ax.text(x - 0.5, i + 0.5, 'N/A', ha='center', va='center', fontsize=7, color='gray')

                # Add text for right half: kN on line 1, (G) on line 2
                if right_kn is not None:
                    text_color = get_text_color(right_kn)
                    ax.text(x + 0.5, i + 0.35, f'{right_kn:.1f} kN', ha='center', va='center',
                            fontsize=7, color=text_color, fontweight='bold')
                    ax.text(x + 0.5, i + 0.65, f'({right_g:.0f} G)', ha='center', va='center',
                            fontsize=6, color=text_color)
                else:
                    ax.text(x + 0.5, i + 0.5, 'N/A', ha='center', va='center', fontsize=7, color='gray')

        # Add stiffness labels below each column group
        for j, stiffness in enumerate(stiffness_values):
            ax.text(x_positions[j], n_rows + 0.15, str(stiffness),
                    ha='center', va='top', fontsize=10)

        # Add row labels to the left
        for i, filename in enumerate(sorted_files):
            ax.text(-1.1, i + 0.5, filename.replace('.csv', ''),
                    ha='right', va='center', fontsize=8)

        # Center of table in figure coordinates
        table_center_x = ax_left + ax_width / 2

        # Title (two lines)
        left_label = 'unlimited' if bo_left == 'unlimited' else f'{bo_left} kN'
        right_label = 'unlimited' if bo_right == 'unlimited' else f'{bo_right} kN'
        title = f'Buttock bottoming out\n{left_label} (left) | {right_label} (right)'
        fig.text(table_center_x, 1 - 0.3 / fig_height, title, ha='center', va='top', fontsize=11, fontweight='bold')

        # Bottom label
        fig.text(table_center_x, 0.0 / fig_height, 'Buttock tissue stiffness (kN/m)', ha='center', fontsize=10)

        # Remove all axes decorations
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        output_path = output_dir / f'{output_dir.name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved to: {output_path}')
        plt.close()
        return

    # Original grid mode or fallback for non-2-group cases
    from matplotlib.gridspec import GridSpec

    n_groups = len(groups)
    group_sizes = [len(g[1]) for g in groups]
    total_cols = sum(group_sizes)

    fig_height = max(10, 2 + n_rows * 0.6)
    fig_width = 4 + total_cols * 1.5

    width_ratios = []
    for i, size in enumerate(group_sizes):
        width_ratios.extend([1] * size)
        if i < n_groups - 1:
            width_ratios.append(0.3)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(1, len(width_ratios), figure=fig, width_ratios=width_ratios,
                  left=0.15, right=0.95, top=0.95, bottom=0.12, wspace=0.05)

    col_idx = 0

    for group_idx, (bo, group_configs) in enumerate(groups):
        n_cols = len(group_configs)
        ax = fig.add_subplot(gs[0, col_idx:col_idx + n_cols])

        matrix = np.full((n_rows, n_cols), np.nan)
        data_values = {}

        for i, filename in enumerate(sorted_files):
            for j, (config_name, _) in enumerate(group_configs):
                if config_name in all_data and filename in all_data[config_name]:
                    entry = all_data[config_name][filename]
                    matrix[i, j] = entry['peak_T12L1_kN']
                    data_values[(i, j)] = entry['peak_T12L1_kN']

        ax.imshow(matrix, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect='auto')

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([str(stiffness) for _, stiffness in group_configs], fontsize=10)

        group_title = 'Buttock tissue\nno bottoming out limit' if bo == 'unlimited' else f'Buttock tissue\nbottoms out at {bo}kN'
        ax.set_title(group_title, fontsize=11, fontweight='bold')

        if group_idx == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([f.replace('.csv', '') for f in sorted_files], fontsize=8)
        else:
            ax.set_yticks([])

        for i in range(n_rows):
            for j in range(n_cols):
                if (i, j) in data_values:
                    kn = data_values[(i, j)]
                    ax.text(j, i, f'{kn:.2f}', ha='center', va='center',
                            fontsize=8, color=get_text_color(kn), fontweight='bold')
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='gray')

        col_idx += n_cols + (1 if group_idx < n_groups - 1 else 0)

    output_path = output_dir / f'{output_dir.name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to: {output_path}')
    plt.close()


def process_csv(csv_path):
    """Process a single CSV file and generate a PNG."""
    output_path = csv_path.with_suffix('.png')
    data, bottom_out_force_kN, bottom_out_compression_mm = load_data(csv_path)

    drop_rates = sorted({k[0] for k in data})
    jerk_limits = sorted({k[1] for k in data})

    matrix = np.full((len(drop_rates), len(jerk_limits)), np.nan)
    for i, dr in enumerate(drop_rates):
        for j, jl in enumerate(jerk_limits):
            if (dr, jl) in data:
                matrix[i, j] = data[(dr, jl)]['kN']

    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_axes([0.12, 0.08, 0.75, 0.75])

    ax.imshow(matrix, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect='auto')

    ax.set_xticks(range(len(jerk_limits)))
    ax.set_xticklabels([str(jl) for jl in jerk_limits])
    ax.set_yticks(range(len(drop_rates)))
    ax.set_yticklabels([f'{dr:.0f}' for dr in drop_rates])
    ax.set_xlabel('Jerk Limit (m/s³)', fontsize=12)
    ax.set_ylabel('Drop Rate (m/s)', fontsize=12)

    if bottom_out_force_kN is not None and bottom_out_force_kN >= 9999:
        title = 'Buttock tissue\nno bottoming out limit'
    elif bottom_out_force_kN is not None:
        if bottom_out_compression_mm is not None:
            title = f'Buttock tissue\nbottoms out at {bottom_out_force_kN:.0f}kN (~{bottom_out_compression_mm:.0f}mm compression)'
        else:
            title = f'Buttock tissue\nbottoms out at {bottom_out_force_kN:.0f}kN'
    else:
        title = 'Buttock tissue'

    fig.text(0.5, 0.92, title, ha='center', va='center', fontsize=14, fontweight='bold')

    for i, dr in enumerate(drop_rates):
        for j, jl in enumerate(jerk_limits):
            if (dr, jl) in data:
                kn = data[(dr, jl)]['kN']
                peak_g = data[(dr, jl)]['peak_g']
                ax.text(j, i, f'{kn:.2f}kN\n({peak_g:.0f}G)', ha='center', va='center',
                        fontsize=9, color=get_text_color(kn), fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=9, color='gray')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to: {output_path}')
    plt.close()


def main():
    import sys

    output_base = Path(__file__).parent / 'output'

    if not output_base.exists():
        print(f'Output directory does not exist: {output_base}')
        return

    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if args:
        subfolder = output_base / args[0]
        if not subfolder.exists():
            print(f'Subfolder does not exist: {subfolder}')
            return
        subfolders = [subfolder]
    else:
        subfolders = sorted([d for d in output_base.iterdir() if d.is_dir()])

    if not subfolders:
        print(f'No subfolders found in {output_base}')
        return

    for subfolder in subfolders:
        print(f'\n{"=" * 60}')
        print(f'Processing: {subfolder.name}')
        print(f'{"=" * 60}')

        batch_dirs = [d for d in subfolder.iterdir() if d.is_dir()]

        if not batch_dirs:
            print(f'  No batch folders found in {subfolder}')
            continue

        csv_files = []
        for batch_dir in sorted(batch_dirs):
            csv_path = batch_dir / f'{batch_dir.name}.csv'
            if csv_path.exists():
                csv_files.append(csv_path)

        if not csv_files:
            print('  No CSV files found in batch folders')
            continue

        if '--grid' in sys.argv:
            for csv_path in csv_files:
                process_csv(csv_path)
        else:
            process_universal(subfolder)


if __name__ == '__main__':
    main()
