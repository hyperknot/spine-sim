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


def _build_element_rest_lengths_mm(
    node_heights_mm: np.ndarray,
    buttocks_height_mm: float,
) -> np.ndarray:
    rest = np.zeros_like(node_heights_mm, dtype=float)
    rest[0] = buttocks_height_mm
    if node_heights_mm.size > 1:
        rest[1:] = node_heights_mm[1:] - node_heights_mm[:-1]
    return rest


def _build_node_heights_from_rest(rest_lengths_mm: np.ndarray) -> np.ndarray:
    node_heights_mm = np.zeros_like(rest_lengths_mm, dtype=float)
    node_heights_mm[0] = rest_lengths_mm[0]
    for i in range(1, rest_lengths_mm.size):
        node_heights_mm[i] = node_heights_mm[i - 1] + rest_lengths_mm[i]
    return node_heights_mm


def _compute_element_geometry_from_rest(
    y_mm: np.ndarray,
    rest_lengths_mm: np.ndarray,
    *,
    buttocks_clamp_to_height: bool = True,
    stack_elements: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_steps, n_elems = y_mm.shape

    ext = np.zeros((n_steps, n_elems), dtype=float)
    ext[:, 0] = y_mm[:, 0]
    if n_elems > 1:
        ext[:, 1:] = y_mm[:, 1:] - y_mm[:, :-1]

    thickness = rest_lengths_mm[np.newaxis, :] + ext

    if buttocks_clamp_to_height:
        butt_thickness = rest_lengths_mm[0] + np.minimum(ext[:, 0], 0.0)
        thickness[:, 0] = np.clip(butt_thickness, 0.0, rest_lengths_mm[0])

    if stack_elements:
        lower = np.zeros_like(thickness)
        upper = np.zeros_like(thickness)
        lower[:, 0] = 0.0
        upper[:, 0] = thickness[:, 0]
        for e in range(1, n_elems):
            lower[:, e] = upper[:, e - 1]
            upper[:, e] = lower[:, e] + thickness[:, e]
    else:
        node_heights_mm = _build_node_heights_from_rest(rest_lengths_mm)
        positions_mm = node_heights_mm[np.newaxis, :] + y_mm

        lower = np.zeros_like(thickness)
        upper = np.zeros_like(thickness)
        lower[:, 0] = 0.0
        upper[:, 0] = positions_mm[:, 0]
        for e in range(1, n_elems):
            lower[:, e] = positions_mm[:, e - 1]
            upper[:, e] = positions_mm[:, e]

    centers = 0.5 * (lower + upper)
    return centers, lower, upper, thickness


def _apply_reference_frame_to_elements(
    centers_mm: np.ndarray,
    lower_mm: np.ndarray,
    upper_mm: np.ndarray,
    node_names: list[str],
    y_mm: np.ndarray,
    rest_lengths_mm: np.ndarray,
    reference_frame: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float, str]:
    if reference_frame == "pelvis":
        pelvis_idx = node_names.index("pelvis")
        node_heights_mm = _build_node_heights_from_rest(rest_lengths_mm)
        pelvis_pos = node_heights_mm[pelvis_idx] + y_mm[:, pelvis_idx]

        centers_mm = centers_mm - pelvis_pos[:, None]
        lower_mm = lower_mm - pelvis_pos[:, None]
        upper_mm = upper_mm - pelvis_pos[:, None]
        return centers_mm, lower_mm, upper_mm, "Height relative to pelvis (mm)", 0.0, "pelvis reference"

    return centers_mm, lower_mm, upper_mm, "Height above excitor plate (mm)", 0.0, "excitor plate (base)"


def plot_displacements(
    time_s: np.ndarray,
    y: np.ndarray,
    accel_g: np.ndarray,
    node_names: list[str],
    elem_names: list[str],
    out_path: Path,
    *,
    heights_from_model: dict[str, float] | None = None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
    reference_frame: str = "base",
    show_element_thickness: bool = False,
    stack_elements: bool = True,
    buttocks_clamp_to_height: bool = True,
) -> None:
    """Plot element centers with buttocks shading and acceleration subplot below."""
    initial_heights_mm = _build_height_map(node_names, heights_from_model, buttocks_height_mm)
    rest_lengths_mm = _build_element_rest_lengths_mm(initial_heights_mm, buttocks_height_mm)
    y_mm = y * 1000.0

    centers_mm, lower_mm, upper_mm, _thickness = _compute_element_geometry_from_rest(
        y_mm,
        rest_lengths_mm,
        buttocks_clamp_to_height=buttocks_clamp_to_height,
        stack_elements=stack_elements,
    )

    centers_mm, lower_mm, upper_mm, ylabel, ref_line, ref_label = _apply_reference_frame_to_elements(
        centers_mm,
        lower_mm,
        upper_mm,
        node_names,
        y_mm,
        rest_lengths_mm,
        reference_frame,
    )

    time_ms = time_s * 1000

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.axhline(y=ref_line, color="black", linewidth=2.0, label=ref_label)
    n_elems = min(len(elem_names), centers_mm.shape[1])
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_elems))

    for i in range(n_elems):
        if show_element_thickness or i == 0:
            ax1.fill_between(
                time_ms,
                lower_mm[:, i],
                upper_mm[:, i],
                color=colors[i],
                alpha=0.18,
                linewidth=0.0,
            )
        ax1.plot(time_ms, centers_mm[:, i], label=elem_names[i], linewidth=1.4, color=colors[i])

    ymin = float(np.min(lower_mm[:, :n_elems]))
    ymax = float(np.max(upper_mm[:, :n_elems]))

    ax1.set_ylabel(ylabel)
    ax1.set_title("Compressible Element Centers During Impact")
    ax1.set_xlim(0, PLOT_DURATION_MS)
    ax1.set_ylim(ymin - 20.0, ymax + 20.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

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

    for i, name in enumerate(elem_names):
        lw = 2.0 if name == highlight else 1.0
        ax1.plot(time_ms, forces_n[:, i] / 1000.0, label=name, linewidth=lw)

    ax1.set_ylabel("Force (kN)")
    ax1.set_title("Junction Forces vs Time (Compression Positive)")
    ax1.set_xlim(0, PLOT_DURATION_MS)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2, fontsize=8)

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
    reference_frame: str = "base",
    show_element_thickness: bool = False,
    stack_elements: bool = True,
    buttocks_clamp_to_height: bool = True,
) -> None:
    """Plot element centers colored by element force with acceleration subplot below."""
    initial_heights_mm = _build_height_map(node_names, heights_from_model, buttocks_height_mm)
    rest_lengths_mm = _build_element_rest_lengths_mm(initial_heights_mm, buttocks_height_mm)
    y_mm = y * 1000.0

    centers_mm, lower_mm, upper_mm, _thickness = _compute_element_geometry_from_rest(
        y_mm,
        rest_lengths_mm,
        buttocks_clamp_to_height=buttocks_clamp_to_height,
        stack_elements=stack_elements,
    )

    centers_mm, lower_mm, upper_mm, ylabel, ref_line, ref_label = _apply_reference_frame_to_elements(
        centers_mm,
        lower_mm,
        upper_mm,
        node_names,
        y_mm,
        rest_lengths_mm,
        reference_frame,
    )

    time_ms = time_s * 1000

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    n_elems = min(len(elem_names), centers_mm.shape[1])
    forces_kN = forces_n[:, :n_elems] / 1000.0

    fmin = min(float(np.min(forces_kN)), 0.0)
    fmax = max(float(np.max(forces_kN)), 1.0)
    norm = plt.Normalize(vmin=fmin, vmax=fmax)
    cmap = plt.get_cmap("plasma")

    ax1.axhline(y=ref_line, color="black", linewidth=2.0, label=ref_label)

    if show_element_thickness or n_elems > 0:
        ax1.fill_between(
            time_ms,
            lower_mm[:, 0],
            upper_mm[:, 0],
            color="lightgray",
            alpha=0.3,
            linewidth=0.0,
        )

    for i in range(n_elems):
        f = forces_kN[:, i]
        y_i = centers_mm[:, i]

        points = np.column_stack([time_ms, y_i])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = cmap(norm(f[:-1]))

        lc = LineCollection(segments, colors=colors, linewidths=1.6)
        ax1.add_collection(lc)

    ax1.set_xlim(0, PLOT_DURATION_MS)
    ymin = float(np.min(lower_mm[:, :n_elems]))
    ymax = float(np.max(upper_mm[:, :n_elems]))
    ax1.set_ylim(ymin - 20.0, ymax + 20.0)

    ax1.set_ylabel(ylabel)
    ax1.set_title("Element Centers Colored by Element Force")
    ax1.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1)
    cbar.set_label("Force (kN)")

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
    elem_names: list[str],
    out_path: Path,
    *,
    heights_from_model: dict[str, float] | None = None,
    buttocks_height_mm: float = DEFAULT_BUTTOCKS_HEIGHT_MM,
    show_element_thickness: bool = False,
    stack_elements: bool = True,
    buttocks_clamp_to_height: bool = True,
) -> None:
    """
    Gravity-settling phase plot.

    Reference frame is the excitor plate (base), NOT the pelvis.
    No acceleration subplot for this plot.
    """
    initial_heights_mm = _build_height_map(
        node_names, heights_from_model, buttocks_height_mm=buttocks_height_mm
    )
    rest_lengths_mm = _build_element_rest_lengths_mm(initial_heights_mm, buttocks_height_mm)
    y_mm = y * 1000.0

    centers_mm, lower_mm, upper_mm, _thickness = _compute_element_geometry_from_rest(
        y_mm,
        rest_lengths_mm,
        buttocks_clamp_to_height=buttocks_clamp_to_height,
        stack_elements=stack_elements,
    )

    centers_mm, lower_mm, upper_mm, ylabel, ref_line, ref_label = _apply_reference_frame_to_elements(
        centers_mm,
        lower_mm,
        upper_mm,
        node_names,
        y_mm,
        rest_lengths_mm,
        reference_frame="base",
    )

    plt.figure(figsize=(14, 8))
    plt.axhline(y=ref_line, color="black", linewidth=2.0, label=ref_label)

    n_elems = min(len(elem_names), centers_mm.shape[1])
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_elems))
    for i in range(n_elems):
        if show_element_thickness or i == 0:
            plt.fill_between(
                time_s * 1000,
                lower_mm[:, i],
                upper_mm[:, i],
                color=colors[i],
                alpha=0.18,
                linewidth=0.0,
            )
        plt.plot(time_s * 1000, centers_mm[:, i], label=elem_names[i], linewidth=1.4, color=colors[i])

    plt.xlabel("Time (ms)")
    plt.ylabel(ylabel)
    plt.title("Gravity Settling Phase (base-referenced)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
