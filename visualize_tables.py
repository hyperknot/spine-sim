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
    buttocks_mode = None
    buttocks_profile = None

    for entry in data:
        parts = entry['filename'].replace('.csv', '').split('-')
        drop_rate = float(parts[0])
        jerk_limit = int(parts[2])

        result[(drop_rate, jerk_limit)] = {
            'kN': float(entry['peak_T12L1_kN']),
            'peak_g': float(entry['base_accel_peak_g']),
        }

        if buttocks_mode is None and 'buttocks_mode' in entry:
            buttocks_mode = entry['buttocks_mode']
        if buttocks_profile is None and 'buttocks_profile' in entry:
            buttocks_profile = entry['buttocks_profile']

    return result, buttocks_mode, buttocks_profile


def process_universal(output_dir):
    """Process all CSV files and create a universal comparison table.

    Batch folders are named {profile}-{mode_label} (e.g. sporty-loc, avg-uni).
    Groups by mode (localized/uniform), with profiles as columns.
    """
    from matplotlib.patches import Rectangle

    valid_modes = {'loc', 'uni'}
    batch_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    configs_info = []

    for batch_dir in batch_dirs:
        name = batch_dir.name
        parts = name.rsplit('-', 1)
        if len(parts) == 2 and parts[1] in valid_modes:
            profile = parts[0]
            mode = parts[1]
            csv_path = batch_dir / f'{name}.csv'
            if csv_path.exists():
                configs_info.append((name, profile, mode, csv_path))

    if not configs_info:
        print('  No batch config folders found')
        return

    # Ordered profiles and modes
    preferred_profiles = ['sporty', 'avg', 'soft']
    all_profiles = list(dict.fromkeys(c[1] for c in configs_info))
    if all(p in all_profiles for p in preferred_profiles):
        profiles = preferred_profiles
    else:
        profiles = sorted(all_profiles)

    modes = ['uni', 'loc']
    mode_labels = {'loc': 'localized', 'uni': 'uniform'}

    # Build lookup: (mode, profile) -> config_name
    config_lookup = {}
    for name, profile, mode, _ in configs_info:
        config_lookup[(mode, profile)] = name

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

    if is_grid_mode:
        drop_rates = {parse_filename(f)[1] for f in all_files}
        if len(drop_rates) <= 1:
            is_grid_mode = False

    print(f'  Mode: {"Grid" if is_grid_mode else "List"} ({len(sorted_files)} files)')
    n_rows = len(sorted_files)
    n_cols = len(profiles)

    # Split-cell table: left = localized, right = uniform, columns = profiles
    if not is_grid_mode and len(modes) == 2:
        # Fixed dimensions (in inches)
        cell_size = 0.55
        gap_size = 0.25
        left_margin = 1.5
        right_margin = 0.3
        top_margin = 0.8
        bottom_margin = 0.5

        table_width = n_cols * (2 * cell_size) + (n_cols - 1) * gap_size
        table_height = n_rows * cell_size
        fig_width = left_margin + table_width + right_margin
        fig_height = top_margin + table_height + bottom_margin

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax_left = left_margin / fig_width
        ax_bottom = bottom_margin / fig_height
        ax_width = table_width / fig_width
        ax_height = table_height / fig_height
        ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

        x_positions = []
        for j in range(n_cols):
            x_positions.append(j * (2 + gap_size / cell_size))

        ax.set_xlim(-1, x_positions[-1] + 1)
        ax.set_ylim(n_rows, 0)

        norm = plt.Normalize(vmin=VMIN, vmax=VMAX)
        mode_left, mode_right = modes[0], modes[1]

        for i, filename in enumerate(sorted_files):
            for j, profile in enumerate(profiles):
                x = x_positions[j]

                left_config = config_lookup.get((mode_left, profile))
                right_config = config_lookup.get((mode_right, profile))

                left_kn, left_g = None, None
                right_kn, right_g = None, None
                if left_config and left_config in all_data and filename in all_data[left_config]:
                    left_kn = all_data[left_config][filename]['peak_T12L1_kN']
                    left_g = all_data[left_config][filename]['base_accel_peak_g']
                if right_config and right_config in all_data and filename in all_data[right_config]:
                    right_kn = all_data[right_config][filename]['peak_T12L1_kN']
                    right_g = all_data[right_config][filename]['base_accel_peak_g']

                # Draw left half
                left_color = CMAP(norm(left_kn)) if left_kn is not None else (0.9, 0.9, 0.9, 1)
                rect_left = Rectangle(
                    (x - 1, i), 1, 1, facecolor=left_color, edgecolor='white', linewidth=0.5
                )
                ax.add_patch(rect_left)

                # Draw right half
                right_color = CMAP(norm(right_kn)) if right_kn is not None else (0.9, 0.9, 0.9, 1)
                rect_right = Rectangle(
                    (x, i), 1, 1, facecolor=right_color, edgecolor='white', linewidth=0.5
                )
                ax.add_patch(rect_right)

                if left_kn is not None:
                    text_color = get_text_color(left_kn)
                    ax.text(
                        x - 0.5,
                        i + 0.35,
                        f'{left_kn:.1f} kN',
                        ha='center',
                        va='center',
                        fontsize=7,
                        color=text_color,
                        fontweight='bold',
                    )
                    ax.text(
                        x - 0.5,
                        i + 0.65,
                        f'({left_g:.0f} G)',
                        ha='center',
                        va='center',
                        fontsize=6,
                        color=text_color,
                    )
                else:
                    ax.text(
                        x - 0.5, i + 0.5, 'N/A', ha='center', va='center', fontsize=7, color='gray'
                    )

                if right_kn is not None:
                    text_color = get_text_color(right_kn)
                    ax.text(
                        x + 0.5,
                        i + 0.35,
                        f'{right_kn:.1f} kN',
                        ha='center',
                        va='center',
                        fontsize=7,
                        color=text_color,
                        fontweight='bold',
                    )
                    ax.text(
                        x + 0.5,
                        i + 0.65,
                        f'({right_g:.0f} G)',
                        ha='center',
                        va='center',
                        fontsize=6,
                        color=text_color,
                    )
                else:
                    ax.text(
                        x + 0.5, i + 0.5, 'N/A', ha='center', va='center', fontsize=7, color='gray'
                    )

        # Profile labels below each column
        for j, profile in enumerate(profiles):
            ax.text(
                x_positions[j], n_rows + 0.15, profile, ha='center', va='top', fontsize=10
            )

        # Row labels
        for i, filename in enumerate(sorted_files):
            ax.text(
                -1.1, i + 0.5, filename.replace('.csv', ''), ha='right', va='center', fontsize=8
            )

        table_center_x = ax_left + ax_width / 2

        title = f'Buttock mode comparison\n{mode_labels[mode_left]} (left) | {mode_labels[mode_right]} (right)'
        fig.text(
            table_center_x,
            1 - 0.3 / fig_height,
            title,
            ha='center',
            va='top',
            fontsize=11,
            fontweight='bold',
        )

        fig.text(
            table_center_x,
            0.0 / fig_height,
            'Buttock profile',
            ha='center',
            fontsize=10,
        )

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

    # Grid mode fallback: one subplot per mode
    from matplotlib.gridspec import GridSpec

    groups = []
    for mode in modes:
        group_configs = [(config_lookup[(mode, p)], p) for p in profiles if (mode, p) in config_lookup]
        if group_configs:
            groups.append((mode, group_configs))

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
    gs = GridSpec(
        1,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        left=0.15,
        right=0.95,
        top=0.95,
        bottom=0.12,
        wspace=0.05,
    )

    col_idx = 0

    for group_idx, (mode, group_configs) in enumerate(groups):
        n_group_cols = len(group_configs)
        ax = fig.add_subplot(gs[0, col_idx : col_idx + n_group_cols])

        matrix = np.full((n_rows, n_group_cols), np.nan)
        data_values = {}

        for i, filename in enumerate(sorted_files):
            for j, (config_name, _) in enumerate(group_configs):
                if config_name in all_data and filename in all_data[config_name]:
                    entry = all_data[config_name][filename]
                    matrix[i, j] = entry['peak_T12L1_kN']
                    data_values[(i, j)] = entry['peak_T12L1_kN']

        ax.imshow(matrix, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect='auto')

        ax.set_xticks(range(n_group_cols))
        ax.set_xticklabels([profile for _, profile in group_configs], fontsize=10)

        ax.set_title(f'Buttock mode: {mode_labels.get(mode, mode)}', fontsize=11, fontweight='bold')

        if group_idx == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([f.replace('.csv', '') for f in sorted_files], fontsize=8)
        else:
            ax.set_yticks([])

        for i in range(n_rows):
            for j in range(n_group_cols):
                if (i, j) in data_values:
                    kn = data_values[(i, j)]
                    ax.text(
                        j,
                        i,
                        f'{kn:.2f}',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color=get_text_color(kn),
                        fontweight='bold',
                    )
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='gray')

        col_idx += n_group_cols + (1 if group_idx < n_groups - 1 else 0)

    output_path = output_dir / f'{output_dir.name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to: {output_path}')
    plt.close()


def process_csv(csv_path):
    """Process a single CSV file and generate a PNG."""
    output_path = csv_path.with_suffix('.png')
    data, buttocks_mode, buttocks_profile = load_data(csv_path)

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

    title_parts = ['Buttock tissue']
    if buttocks_mode:
        title_parts.append(f'mode={buttocks_mode}')
    if buttocks_profile:
        title_parts.append(f'profile={buttocks_profile}')
    title = title_parts[0] + '\n' + ', '.join(title_parts[1:]) if len(title_parts) > 1 else title_parts[0]

    fig.text(0.5, 0.92, title, ha='center', va='center', fontsize=14, fontweight='bold')

    for i, dr in enumerate(drop_rates):
        for j, jl in enumerate(jerk_limits):
            if (dr, jl) in data:
                kn = data[(dr, jl)]['kN']
                peak_g = data[(dr, jl)]['peak_g']
                ax.text(
                    j,
                    i,
                    f'{kn:.2f}kN\n({peak_g:.0f}G)',
                    ha='center',
                    va='center',
                    fontsize=9,
                    color=get_text_color(kn),
                    fontweight='bold',
                )
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
