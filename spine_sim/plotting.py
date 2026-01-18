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
    if reference_frame == "pelvis":
        pelvis_idx = node_names.index("pelvis")
        pelvis_pos = positions_mm[:, pelvis_idx]
        positions_mm = positions_mm - pelvis_pos[:, None]
        return positions_mm, "Height relative to pelvis (mm)", 0.0

    return positions_mm, "Height above seat (mm)", 0.0


def plot_displacements(
    time_s: np.ndarray,
    y: np.ndarray,
    accel_g: np.ndarray,
    node_names: list[str],
    out_path: Path,
    *,
    heights_from_model: dict[str, float] | None = None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
    reference_frame: str = "pelvis",
) -> None:
    """Plot displacements with acceleration subplot below."""
    initial_heights_mm = _build_height_map(node_names, heights_from_model, buttocks_height_mm)
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0
    positions_mm, ylabel, ref_line = _apply_reference_frame(positions_mm, node_names, reference_frame)

    time_ms = time_s * 1000

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Displacement plot
    ax1.axhline(y=ref_line, color="black", linewidth=2.0, label=f"{reference_frame} reference")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(node_names)))
    for i, name in enumerate(node_names):
        ax1.plot(time_ms, positions_mm[:, i], label=name, linewidth=1.4, color=colors[i])

    ax1.set_ylabel(ylabel)
    ax1.set_title("Spine Side View - Segment Heights During Impact")
    ax1.set_xlim(0, PLOT_DURATION_MS)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # Acceleration subplot
    ax2.plot(time_ms, accel_g, color="tab:red", linewidth=1.2)
    ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Base Accel (g)")
    ax2.set_xlim(0, PLOT_DURATION_MS)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_forces(
    time_s: np.ndarray,
    forces_n: np.ndarray,
    accel_g: np.ndarray,
    elem_names: list[str],
    out_path: Path,
    highlight: str,
) -> None:
    """Plot junction forces with acceleration subplot below."""
    time_ms = time_s * 1000

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Forces plot
    for i, name in enumerate(elem_names):
        lw = 2.0 if name == highlight else 1.0
        ax1.plot(time_ms, forces_n[:, i] / 1000.0, label=name, linewidth=lw)

    ax1.set_ylabel("Force (kN)")
    ax1.set_title("Junction Forces vs Time (Compression Positive)")
    ax1.set_xlim(0, PLOT_DURATION_MS)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2, fontsize=8)

    # Acceleration subplot
    ax2.plot(time_ms, accel_g, color="tab:red", linewidth=1.2)
    ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Base Accel (g)")
    ax2.set_xlim(0, PLOT_DURATION_MS)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_displacement_colored_by_force(
    time_s: np.ndarray,
    y: np.ndarray,
    forces_n: np.ndarray,
    accel_g: np.ndarray,
    node_names: list[str],
    elem_names: list[str],
    out_path: Path,
    *,
    heights_from_model: dict[str, float] | None = None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
    reference_frame: str = "pelvis",
) -> None:
    """Plot displacement colored by force with acceleration subplot below."""
    initial_heights_mm = _build_height_map(node_names, heights_from_model, buttocks_height_mm)
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0
    positions_mm, ylabel, ref_line = _apply_reference_frame(positions_mm, node_names, reference_frame)

    time_ms = time_s * 1000

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    fmin = min(float(np.min(forces_n)), 0.0) / 1000.0
    fmax = max(float(np.max(forces_n)), 1.0) / 1000.0
    norm = plt.Normalize(vmin=fmin, vmax=fmax)
    cmap = plt.get_cmap("plasma")

    ax1.axhline(y=ref_line, color="black", linewidth=2.0, label=f"{reference_frame} reference")

    for i, _name in enumerate(node_names):
        f = forces_n[:, i] / 1000.0
        x = time_ms
        y_i = positions_mm[:, i]

        points = np.column_stack([x, y_i])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = cmap(norm(f[:-1]))

        lc = LineCollection(segments, colors=colors, linewidths=1.6)
        ax1.add_collection(lc)

    ax1.set_xlim(0, PLOT_DURATION_MS)
    ymin = float(np.min(positions_mm))
    ymax = float(np.max(positions_mm))
    ax1.set_ylim(ymin - 20.0, ymax + 20.0)

    ax1.set_ylabel(ylabel)
    ax1.set_title("Spine Side View - Colored by Junction Force Below")
    ax1.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1)
    cbar.set_label("Force (kN)")

    # Acceleration subplot
    ax2.plot(time_ms, accel_g, color="tab:red", linewidth=1.2)
    ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Base Accel (g)")
    ax2.set_xlim(0, PLOT_DURATION_MS)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_gravity_settling(
    time_s: np.ndarray,
    y: np.ndarray,
    node_names: list[str],
    out_path: Path,
    *,
    heights_from_model: dict[str, float] | None = None,
) -> None:
    """
    Gravity-settling phase plot.

    Reference frame is the excitor plate (base), NOT the pelvis.
    No acceleration subplot for this plot.
    """
    initial_heights_mm = _build_height_map(
        node_names, heights_from_model, buttocks_height_mm=0.0
    )
    positions_mm = initial_heights_mm[np.newaxis, :] + y * 1000.0
    positions_mm, ylabel, ref_line = _apply_reference_frame(positions_mm, node_names, reference_frame="base")

    plt.figure(figsize=(14, 8))
    plt.axhline(y=0.0, color="black", linewidth=2.0, label="excitor plate (base)")

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(node_names)))
    for i, name in enumerate(node_names):
        plt.plot(time_s * 1000, positions_mm[:, i], label=name, linewidth=1.4, color=colors[i])

    plt.xlabel("Time (ms)")
    plt.ylabel("Height above excitor plate (mm)")
    plt.title("Gravity Settling Phase (base-referenced)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
