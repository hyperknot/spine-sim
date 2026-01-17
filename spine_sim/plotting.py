from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


PLOT_DURATION_MS = 200.0
DEFAULT_BUTTOCKS_HEIGHT_MM = 100.0


def _build_height_map(
    node_names: list[str],
    heights_from_model: dict[str, float] | None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
) -> np.ndarray:
    name_map = {
        "pelvis": "pelvis",
        "L5": "lumbar5",
        "L4": "lumbar4",
        "L3": "lumbar3",
        "L2": "lumbar2",
        "L1": "lumbar1",
        "T12": "thoracic12",
        "T11": "thoracic11",
        "T10": "thoracic10",
        "T9": "thoracic9",
        "T8": "thoracic8",
        "T7": "thoracic7",
        "T6": "thoracic6",
        "T5": "thoracic5",
        "T4": "thoracic4",
        "T3": "thoracic3",
        "T2": "thoracic2",
        "T1": "thoracic1",
        "HEAD": "head_neck",
    }

    heights = []
    for node in node_names:
        if heights_from_model is None:
            heights.append(buttocks_height_mm + len(heights) * 35.0)
        else:
            opensim_name = name_map.get(node.upper(), name_map.get(node, None))
            if opensim_name and opensim_name in heights_from_model:
                rel_height = heights_from_model[opensim_name]
                heights.append(buttocks_height_mm + rel_height)
            else:
                heights.append(buttocks_height_mm + len(heights) * 35.0)

    return np.array(heights, dtype=float)


def _apply_reference_frame(
    positions_mm: np.ndarray,
    node_names: list[str],
    reference_frame: str,
) -> tuple[np.ndarray, str, float]:
    """
    reference_frame:
      - "base": absolute heights above base plate (old behavior)
      - "pelvis": subtract pelvis motion so pelvis stays fixed (recommended for your request)
    """
    if reference_frame == "pelvis":
        pelvis_idx = node_names.index("pelvis")
        pelvis_pos = positions_mm[:, pelvis_idx]
        # anchor pelvis to 0 in plot frame
        positions_mm = positions_mm - pelvis_pos[:, None]
        return positions_mm, "Height relative to pelvis (mm)", 0.0

    return positions_mm, "Height above seat (mm)", 0.0


def plot_displacements(
    time_s: np.ndarray,
    y: np.ndarray,
    node_names: list[str],
    out_path: Path,
    *,
    heights_from_model: dict[str, float] | None = None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
    reference_frame: str = "pelvis",
) -> None:
    initial_heights_mm = _build_height_map(node_names, heights_from_model, buttocks_height_mm)
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0
    positions_mm, ylabel, ref_line = _apply_reference_frame(positions_mm, node_names, reference_frame)

    plt.figure(figsize=(14, 8))
    plt.axhline(y=ref_line, color="black", linewidth=2.0, label=f"{reference_frame} reference")

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(node_names)))
    for i, name in enumerate(node_names):
        plt.plot(time_s * 1000, positions_mm[:, i], label=name, linewidth=1.4, color=colors[i])

    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.title("Spine Side View - Segment Heights During Impact")
    plt.xlim(0, PLOT_DURATION_MS)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
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
    *,
    heights_from_model: dict[str, float] | None = None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
    reference_frame: str = "pelvis",
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    initial_heights_mm = _build_height_map(node_names, heights_from_model, buttocks_height_mm)
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0
    positions_mm, ylabel, ref_line = _apply_reference_frame(positions_mm, node_names, reference_frame)

    fmin = min(float(np.min(forces_n)), 0.0) / 1000.0
    fmax = max(float(np.max(forces_n)), 1.0) / 1000.0
    norm = plt.Normalize(vmin=fmin, vmax=fmax)
    cmap = plt.get_cmap("plasma")

    ax.axhline(y=ref_line, color="black", linewidth=2.0, label=f"{reference_frame} reference")

    for i, _name in enumerate(node_names):
        f = forces_n[:, i] / 1000.0
        x = time_s * 1000
        y_i = positions_mm[:, i]

        points = np.column_stack([x, y_i])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = cmap(norm(f[:-1]))

        lc = LineCollection(segments, colors=colors, linewidths=1.6)
        ax.add_collection(lc)

    ax.set_xlim(0, PLOT_DURATION_MS)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title("Spine Side View - Colored by Junction Force Below")
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Force (kN)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
