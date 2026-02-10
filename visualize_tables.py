#!/usr/bin/env -S uv run
# /// script
# dependencies = ["matplotlib", "click"]
# ///

"""
Visualize spine simulation results as colored tables.

Processes CSV files in output/<scenario>/<profile>-<mode>/<profile>-<mode>.csv
and generates a PNG per scenario.

Layout options:
  --profile: Group by buttock profile (3 groups × 2 columns) [default]
  --mode: Group by buttock mode (2 groups × 3 columns)
"""

import csv
import re
from pathlib import Path

import click
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Color scale constants
VMIN = 2.0
VMAX = 14.0

# Layout constants
PROFILES = ['sporty', 'avg', 'soft']
MODES = ['uni', 'loc']
MODE_LABELS = {'loc': 'localized', 'uni': 'uniform'}


def natural_sort_key(s):
    """Generate a key for natural sorting (numbers sorted numerically)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', s)]


def create_colormap():
    """Create smooth rainbow-like colormap for spine force visualization."""

    def kn_to_norm(kn):
        return (kn - VMIN) / (VMAX - VMIN)

    colors = [
        (0.18, 0.20, 0.58),  # 2 kN
        (0.20, 0.36, 0.72),  # 3 kN
        (0.16, 0.50, 0.72),  # 4 kN
        (0.12, 0.62, 0.67),  # 5 kN
        (0.20, 0.68, 0.48),  # 6 kN
        (0.70, 0.80, 0.28),  # 7 kN
        (0.92, 0.70, 0.20),  # 8 kN
        (0.92, 0.45, 0.18),  # 9 kN
        (0.90, 0.22, 0.18),  # 10 kN
        (0.82, 0.18, 0.45),  # 11 kN
        (0.65, 0.12, 0.55),  # 12 kN
        (0.45, 0.08, 0.50),  # 13 kN
        (0.30, 0.05, 0.40),  # 14 kN
    ]
    positions = [kn_to_norm(kn) for kn in range(2, 15)]
    return mcolors.LinearSegmentedColormap.from_list('spine_rainbow', list(zip(positions, colors)))


def get_text_color(kn):
    """Return text color (white/black) based on background brightness."""
    return 'white' if kn <= 4.0 or kn >= 10.0 else 'black'


CMAP = create_colormap()


def collect_scenario_data(scenario_dir: Path):
    """Collect all CSV data from a scenario folder.

    Returns:
        all_data: dict (profile, mode) -> filename -> {peak_T12L1_kN, base_accel_peak_g}
        filenames: sorted list of all filenames
        available_combos: set of (profile, mode) tuples that exist
    """
    all_data = {}
    all_filenames = set()
    available_combos = set()

    for profile in PROFILES:
        for mode in MODES:
            batch_name = f'{profile}-{mode}'
            csv_path = scenario_dir / batch_name / f'{batch_name}.csv'

            if not csv_path.exists():
                continue

            available_combos.add((profile, mode))

            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)

            all_data[(profile, mode)] = {}
            for entry in data:
                filename = entry['filename']
                all_filenames.add(filename)
                all_data[(profile, mode)][filename] = {
                    'peak_T12L1_kN': float(entry['peak_T12L1_kN']),
                    'base_accel_peak_g': float(entry['base_accel_peak_g']),
                }

    return all_data, sorted(all_filenames, key=natural_sort_key), available_combos


def render_table(scenario_dir: Path, group_by: str = 'mode'):
    """Render the comparison table for a scenario."""
    all_data, sorted_files, available_combos = collect_scenario_data(scenario_dir)

    if not sorted_files:
        print('  No data found, skipping')
        return

    # Check we have all required profile/mode combos
    required_combos = {(p, m) for p in PROFILES for m in MODES}
    if available_combos != required_combos:
        missing = required_combos - available_combos
        print(f'  Missing combos: {missing}, skipping')
        return

    n_rows = len(sorted_files)

    # Define groups based on grouping mode
    if group_by == 'profile':
        # 3 groups (sporty, avg, soft), 2 columns each (uni, loc)
        groups = [(p, [(p, m) for m in MODES]) for p in PROFILES]
        group_labels = PROFILES
        col_labels = {(p, m): MODE_LABELS[m] for p in PROFILES for m in MODES}
    else:  # group_by == "mode"
        # 2 groups (uni, loc), 3 columns each (sporty, avg, soft)
        groups = [(m, [(p, m) for p in PROFILES]) for m in MODES]
        group_labels = [MODE_LABELS[m] for m in MODES]
        col_labels = {(p, m): p for p in PROFILES for m in MODES}

    n_groups = len(groups)
    cols_per_group = len(groups[0][1])

    # Dimensions (matching original list mode)
    cell_size = 0.55
    gap_size = 0.25
    left_margin = 1.5
    right_margin = 0.3
    top_margin = 0.8
    bottom_margin = 0.5

    # Calculate table dimensions
    n_cells = n_groups * cols_per_group
    n_gaps = n_groups - 1
    gap_units = gap_size / cell_size
    total_units = n_cells + n_gaps * gap_units

    table_width = n_cells * cell_size + n_gaps * gap_size
    table_height = n_rows * cell_size

    fig_width = left_margin + table_width + right_margin
    fig_height = top_margin + table_height + bottom_margin

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Position axis in figure
    ax_left = left_margin / fig_width
    ax_bottom = bottom_margin / fig_height
    ax_width = table_width / fig_width
    ax_height = table_height / fig_height
    ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

    # Axis limits
    ax.set_xlim(0, total_units)
    ax.set_ylim(n_rows, 0)

    # Calculate x positions (cell centers)
    x_positions = []
    combos_in_order = []
    for g_idx, (_, combos) in enumerate(groups):
        group_start = g_idx * (cols_per_group + gap_units)
        for c_idx, combo in enumerate(combos):
            x_positions.append(group_start + c_idx + 0.5)
            combos_in_order.append(combo)

    norm = plt.Normalize(vmin=VMIN, vmax=VMAX)

    # Draw cells
    for col_idx, (profile, mode) in enumerate(combos_in_order):
        x = x_positions[col_idx]

        for row_idx, filename in enumerate(sorted_files):
            entry = all_data[(profile, mode)].get(filename)

            if entry:
                kn = entry['peak_T12L1_kN']
                g_val = entry['base_accel_peak_g']
                color = CMAP(norm(kn))
                text_color = get_text_color(kn)
            else:
                kn, g_val = None, None
                color = (0.9, 0.9, 0.9, 1)
                text_color = 'gray'

            rect = Rectangle(
                (x - 0.5, row_idx),
                1,
                1,
                facecolor=color,
                edgecolor='white',
                linewidth=0.5,
            )
            ax.add_patch(rect)

            if kn is not None:
                ax.text(
                    x,
                    row_idx + 0.35,
                    f'{kn:.1f} kN',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color=text_color,
                    fontweight='bold',
                )
                ax.text(
                    x,
                    row_idx + 0.65,
                    f'({g_val:.0f} G)',
                    ha='center',
                    va='center',
                    fontsize=6,
                    color=text_color,
                )
            else:
                ax.text(
                    x,
                    row_idx + 0.5,
                    'N/A',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='gray',
                )

    # Column labels (below columns)
    for col_idx, combo in enumerate(combos_in_order):
        ax.text(
            x_positions[col_idx],
            n_rows + 0.15,
            col_labels[combo],
            ha='center',
            va='top',
            fontsize=9,
        )

    # Group labels (above each group, centered)
    for g_idx, (_, combos) in enumerate(groups):
        group_start = g_idx * (cols_per_group + gap_units)
        group_center = group_start + cols_per_group / 2
        label = group_labels[g_idx]
        ax.text(
            group_center,
            -0.3,
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    # Row labels
    for i, filename in enumerate(sorted_files):
        ax.text(
            -0.1,
            i + 0.5,
            filename.replace('.csv', ''),
            ha='right',
            va='center',
            fontsize=8,
        )

    # Title
    table_center_x = ax_left + ax_width / 2

    fig.text(
        table_center_x,
        1 - 0.15 / fig_height,
        'Spine force comparison',
        ha='center',
        va='top',
        fontsize=11,
        fontweight='bold',
    )

    # Hide spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Save
    suffix = 'by-profile' if group_by == 'profile' else 'by-mode'
    output_path = scenario_dir / f'{scenario_dir.name}-{suffix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {output_path}')
    plt.close()


@click.command()
@click.argument('scenario', required=False)
@click.option('--profile', 'group_by', flag_value='profile', help='Group by buttock profile')
@click.option('--mode', 'group_by', flag_value='mode', help='Group by buttock mode')
def main(scenario, group_by):
    """Generate spine force comparison tables.

    SCENARIO: Optional scenario name. If omitted, processes all scenarios.
    """
    output_base = Path(__file__).parent / 'output'

    if not output_base.exists():
        print(f'Output directory does not exist: {output_base}')
        return

    if scenario:
        scenario_dir = output_base / scenario
        if not scenario_dir.exists():
            print(f'Scenario does not exist: {scenario_dir}')
            return
        scenarios = [scenario_dir]
    else:
        scenarios = sorted([d for d in output_base.iterdir() if d.is_dir()])

    if not scenarios:
        print(f'No scenario folders found in {output_base}')
        return

    for scenario_dir in scenarios:
        print(f'\n{"=" * 60}')
        print(f'Processing: {scenario_dir.name}')
        print(f'{"=" * 60}')
        render_table(scenario_dir, group_by)


if __name__ == '__main__':
    main()
