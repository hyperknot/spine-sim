from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_displacements(
    time_s: np.ndarray, y: np.ndarray, node_names: list[str], out_path: Path
) -> None:
    plt.figure(figsize=(12, 7))
    for i, name in enumerate(node_names):
        plt.plot(time_s * 1000, y[:, i] * 1000, label=name, linewidth=1.2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Displacement (mm)')
    plt.title('Node Displacements vs Time')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
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
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color by force in the element below each node
    # For pelvis (node 0), use buttocks element (index 0)
    # For node i>0, use element i (between node i-1 and i)
    force_ref = forces_n.copy()
    fmin = np.min(force_ref)
    fmax = np.max(force_ref)
    fmin = min(fmin, 0.0)
    fmax = max(fmax, 1.0)

    norm = plt.Normalize(vmin=fmin, vmax=fmax)
    cmap = plt.get_cmap('viridis')

    for i, name in enumerate(node_names):
        f = force_ref[:, i]
        x = time_s * 1000
        y_i = y[:, i] * 1000

        points = np.column_stack([x, y_i])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = cmap(norm(f[:-1]))

        lc = LineCollection(segments, colors=colors, linewidths=1.4)
        ax.add_collection(lc)
        ax.plot(x, y_i, alpha=0.15)

    ax.set_xlim(time_s[0] * 1000, time_s[-1] * 1000)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Displacement (mm)')
    ax.set_title('Displacements Colored by Junction Force Below')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Force (N)')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
