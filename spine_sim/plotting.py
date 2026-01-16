from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_displacements(
    time_s: np.ndarray, y: np.ndarray, node_names: list[str], out_path: Path
) -> None:
    """Plot spine side view: height above base plate vs time.

    Origin is at the base plate. Each body segment is stacked at a constant
    level above the base. Displacements show how these levels move and compress
    during impact - like viewing the pilot's spine from the side with a camera.

    If nothing happened (no impact), all lines would be parallel horizontal lines.
    During compression, the lines converge as segments are pushed together.
    """
    # Anatomical spacing between segments (mm)
    # Using approximate disc/segment heights for realistic visualization
    level_spacing_mm = 25.0  # ~25mm between vertebral levels

    n_nodes = len(node_names)

    # Initial heights: base at 0, pelvis at level 1, L5 at level 2, etc.
    # Each node is stacked above the previous one
    initial_heights_mm = np.array([(i + 1) * level_spacing_mm for i in range(n_nodes)])

    # Position at each time = initial height + displacement
    # y[:, i] is displacement in meters (positive = moved up relative to base)
    # Convert to mm
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0

    plt.figure(figsize=(14, 8))

    # Plot base plate at y=0
    plt.axhline(y=0, color='black', linewidth=2.5, label='Base plate')

    # Color palette for clear differentiation
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_nodes))

    for i, name in enumerate(node_names):
        plt.plot(
            time_s * 1000,
            positions_mm[:, i],
            label=name,
            linewidth=1.4,
            color=colors[i]
        )

    plt.xlabel('Time (ms)')
    plt.ylabel('Height above base plate (mm)')
    plt.title('Spine Side View - Segment Heights During Impact')
    plt.grid(True, alpha=0.3)

    # Legend outside plot for clarity
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def plot_forces(
    time_s: np.ndarray, forces_n: np.ndarray, elem_names: list[str], out_path: Path, highlight: str
) -> None:
    plt.figure(figsize=(12, 7))
    for i, name in enumerate(elem_names):
        lw = 2.0 if name == highlight else 1.0
        plt.plot(time_s * 1000, forces_n[:, i] / 1000.0, label=name, linewidth=lw)
    plt.xlabel('Time (ms)')
    plt.ylabel('Force (kN)')
    plt.title('Junction Forces vs Time (Compression Positive)')
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
) -> None:
    """Plot spine side view with segments colored by force.

    Same as plot_displacements but each segment line is colored by the
    compressive force in the element below it. High forces show as bright colors.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Anatomical spacing (same as plot_displacements)
    level_spacing_mm = 25.0
    n_nodes = len(node_names)
    initial_heights_mm = np.array([(i + 1) * level_spacing_mm for i in range(n_nodes)])

    # Position = initial height + displacement
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0

    # Color by force in the element below each node
    force_ref = forces_n.copy()
    fmin = np.min(force_ref)
    fmax = np.max(force_ref)
    fmin = min(fmin, 0.0)
    fmax = max(fmax, 1.0)

    norm = plt.Normalize(vmin=fmin / 1000.0, vmax=fmax / 1000.0)  # Convert to kN
    cmap = plt.get_cmap('plasma')

    # Plot base plate
    ax.axhline(y=0, color='black', linewidth=2.5, label='Base plate')

    for i, name in enumerate(node_names):
        f = force_ref[:, i] / 1000.0  # Convert to kN
        x = time_s * 1000
        y_i = positions_mm[:, i]

        points = np.column_stack([x, y_i])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = cmap(norm(f[:-1]))

        lc = LineCollection(segments, colors=colors, linewidths=1.6)
        ax.add_collection(lc)
        # Light background line for visibility
        ax.plot(x, y_i, alpha=0.1, color='gray', linewidth=0.5)

    ax.set_xlim(time_s[0] * 1000, time_s[-1] * 1000)
    ymin = min(0, np.min(positions_mm) - 10)
    ymax = np.max(positions_mm) + 20
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Height above base plate (mm)')
    ax.set_title('Spine Side View - Colored by Junction Force Below')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Force (kN)')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
