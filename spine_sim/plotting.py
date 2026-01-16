from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# Fixed plot duration
PLOT_DURATION_MS = 200.0

# Default buttocks tissue height (pelvis above seat surface when sitting)
# This is added to all OpenSim-derived heights since OpenSim gives heights
# relative to pelvis, but we want heights above seat surface
DEFAULT_BUTTOCKS_HEIGHT_MM = 100.0


def _build_height_map(
    node_names: list[str],
    heights_from_model: dict[str, float] | None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
) -> np.ndarray:
    """Build array of initial heights for each node.

    Args:
        node_names: List of node names in the spine chain
        heights_from_model: Dict mapping OpenSim body names to heights relative to pelvis (mm)
        buttocks_height_mm: Height of pelvis above seat surface (buttocks tissue thickness)

    Returns:
        Array of heights in mm for each node
    """
    # Mapping from our node names to OpenSim body names
    name_map = {
        'pelvis': 'pelvis',
        'L5': 'lumbar5',
        'L4': 'lumbar4',
        'L3': 'lumbar3',
        'L2': 'lumbar2',
        'L1': 'lumbar1',
        'T12': 'thoracic12',
        'T11': 'thoracic11',
        'T10': 'thoracic10',
        'T9': 'thoracic9',
        'T8': 'thoracic8',
        'T7': 'thoracic7',
        'T6': 'thoracic6',
        'T5': 'thoracic5',
        'T4': 'thoracic4',
        'T3': 'thoracic3',
        'T2': 'thoracic2',
        'T1': 'thoracic1',
        'HEAD': 'head_neck',
    }

    heights = []
    for node in node_names:
        if heights_from_model is None:
            # Fallback: linear spacing
            heights.append(buttocks_height_mm + len(heights) * 35.0)
        else:
            opensim_name = name_map.get(node.upper(), name_map.get(node, None))
            if opensim_name and opensim_name in heights_from_model:
                # Height from OpenSim (relative to pelvis) + buttocks offset
                rel_height = heights_from_model[opensim_name]
                heights.append(buttocks_height_mm + rel_height)
            else:
                # Fallback for unknown bodies
                heights.append(buttocks_height_mm + len(heights) * 35.0)

    return np.array(heights, dtype=float)


def plot_displacements(
    time_s: np.ndarray,
    y: np.ndarray,
    node_names: list[str],
    out_path: Path,
    heights_from_model: dict[str, float] | None = None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
) -> None:
    """Plot spine side view: height above seat vs time.

    Shows anatomical positions during impact. Each line represents a body
    segment positioned at its height from the OpenSim model. Displacements
    show compression during impact.
    """
    initial_heights_mm = _build_height_map(node_names, heights_from_model, buttocks_height_mm)

    # Position = initial height + displacement (y in meters -> mm)
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0

    plt.figure(figsize=(14, 8))

    # Base plate at y=0
    plt.axhline(y=0, color='black', linewidth=2.5, label='Base plate')

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(node_names)))

    for i, name in enumerate(node_names):
        plt.plot(
            time_s * 1000,
            positions_mm[:, i],
            label=name,
            linewidth=1.4,
            color=colors[i],
        )

    plt.xlabel('Time (ms)')
    plt.ylabel('Height above seat (mm)')
    plt.title('Spine Side View - Segment Heights During Impact')
    plt.xlim(0, PLOT_DURATION_MS)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def plot_forces(
    time_s: np.ndarray,
    forces_n: np.ndarray,
    elem_names: list[str],
    out_path: Path,
    highlight: str,
) -> None:
    """Plot junction forces vs time."""
    plt.figure(figsize=(12, 7))
    for i, name in enumerate(elem_names):
        lw = 2.0 if name == highlight else 1.0
        plt.plot(time_s * 1000, forces_n[:, i] / 1000.0, label=name, linewidth=lw)
    plt.xlabel('Time (ms)')
    plt.ylabel('Force (kN)')
    plt.title('Junction Forces vs Time (Compression Positive)')
    plt.xlim(0, PLOT_DURATION_MS)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_displacement_colored_by_force(
    time_s: np.ndarray,
    y: np.ndarray,
    forces_n: np.ndarray,
    node_names: list[str],
    elem_names: list[str],
    out_path: Path,
    heights_from_model: dict[str, float] | None = None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
) -> None:
    """Plot spine side view with segments colored by force below."""
    fig, ax = plt.subplots(figsize=(14, 8))

    initial_heights_mm = _build_height_map(node_names, heights_from_model, buttocks_height_mm)
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0

    force_ref = forces_n.copy()
    fmin = min(np.min(force_ref), 0.0)
    fmax = max(np.max(force_ref), 1.0)

    norm = plt.Normalize(vmin=fmin / 1000.0, vmax=fmax / 1000.0)
    cmap = plt.get_cmap('plasma')

    ax.axhline(y=0, color='black', linewidth=2.5, label='Base plate')

    for i, name in enumerate(node_names):
        f = force_ref[:, i] / 1000.0
        x = time_s * 1000
        y_i = positions_mm[:, i]

        points = np.column_stack([x, y_i])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = cmap(norm(f[:-1]))

        lc = LineCollection(segments, colors=colors, linewidths=1.6)
        ax.add_collection(lc)
        ax.plot(x, y_i, alpha=0.1, color='gray', linewidth=0.5)

    ax.set_xlim(0, PLOT_DURATION_MS)
    ymin = min(0, np.min(positions_mm) - 10)
    ymax = np.max(positions_mm) + 20
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Height above seat (mm)')
    ax.set_title('Spine Side View - Colored by Junction Force Below')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Force (kN)')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
