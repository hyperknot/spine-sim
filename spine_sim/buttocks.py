"""Toen buttocks model utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spine_sim.model import SpineModel
from spine_sim.plotting import plot_toen_buttocks_force_compression
from spine_sim.toen_drop import (
    TOEN_IMPACT_V_MPS,
    TOEN_SOLVER_DT_S,
    TOEN_SOLVER_DURATION_S,
    TOEN_SOLVER_MAX_NEWTON_ITER,
    simulate_toen_drop_trace,
)
from spine_sim.toen_store import require_toen_drop_calibration
from spine_sim.toen_subjects import TOEN_SUBJECTS, toen_torso_mass_scaled_kg
from spine_sim.toen_targets import TOEN_FLOOR_STIFFNESS_N_PER_M

# ----------------------------
# Hard-coded run settings
# ----------------------------
TOEN_SUBJECT_ID = "avg"
MALE50_MASS_KG = 75.4


def get_toen_buttocks_params() -> dict:
    """Load Toen calibration. Fails if missing (calibration is required)."""
    doc, path = require_toen_drop_calibration()

    return {
        "k": doc.get("buttocks_k_n_per_m"),
        "c": doc.get("buttocks_c_ns_per_m"),
        "limit_mm": doc.get("buttocks_limit_mm"),
        "stop_k": doc.get("buttocks_stop_k_n_per_m"),
        "smoothing_mm": doc.get("buttocks_stop_smoothing_mm"),
        "_path": str(path),
    }


def apply_toen_buttocks_to_model(model: SpineModel, toen_params: dict) -> SpineModel:
    """Apply Toen-calibrated buttocks parameters to spine model.

    Updates k, c, and densification parameters for the buttocks element (index 0).
    Also scales maxwell branch stiffnesses proportionally to the new k.
    """
    if toen_params is None:
        return model

    k_elem = model.k_elem.copy()
    c_elem = model.c_elem.copy()

    # Scale factor for maxwell branches (ratio of new k to old k)
    k_scale = 1.0
    if toen_params.get("k") is not None:
        new_k = float(toen_params["k"])
        if k_elem[0] > 0:
            k_scale = new_k / k_elem[0]
        k_elem[0] = new_k
    if toen_params.get("c") is not None:
        c_elem[0] = float(toen_params["c"])

    # Scale maxwell branches for buttocks element
    maxwell_k = model.maxwell_k
    if maxwell_k is not None and maxwell_k.size > 0 and k_scale != 1.0:
        maxwell_k = maxwell_k.copy()
        maxwell_k[0, :] *= k_scale

    limit_m = model.compression_limit_m
    stop_k = model.compression_stop_k
    smoothing_m = model.compression_stop_smoothing_m

    if toen_params.get("limit_mm") is not None:
        if limit_m is None:
            limit_m = np.zeros(model.n_elems(), dtype=float)
        else:
            limit_m = limit_m.copy()
        limit_m[0] = float(toen_params["limit_mm"]) / 1000.0

    if toen_params.get("stop_k") is not None:
        if stop_k is None:
            stop_k = np.zeros(model.n_elems(), dtype=float)
        else:
            stop_k = stop_k.copy()
        stop_k[0] = float(toen_params["stop_k"])

    if toen_params.get("smoothing_mm") is not None:
        if smoothing_m is None:
            smoothing_m = np.zeros(model.n_elems(), dtype=float)
        else:
            smoothing_m = smoothing_m.copy()
        smoothing_m[0] = float(toen_params["smoothing_mm"]) / 1000.0

    return SpineModel(
        node_names=model.node_names,
        masses_kg=model.masses_kg,
        element_names=model.element_names,
        k_elem=k_elem,
        c_elem=c_elem,
        compression_ref_m=model.compression_ref_m,
        compression_k_mult=model.compression_k_mult,
        tension_k_mult=model.tension_k_mult,
        compression_only=model.compression_only,
        damping_compression_only=model.damping_compression_only,
        gap_m=model.gap_m,
        maxwell_k=maxwell_k,
        maxwell_tau_s=model.maxwell_tau_s,
        maxwell_compression_only=model.maxwell_compression_only,
        poly_k2=model.poly_k2,
        poly_k3=model.poly_k3,
        compression_limit_m=limit_m,
        compression_stop_k=stop_k,
        compression_stop_smoothing_m=smoothing_m,
    )


def compute_free_buttocks_height_mm(toen_params: dict | None) -> float:
    """Compute the free (uncompressed) buttocks height from Toen parameters."""
    if toen_params is None or toen_params.get("limit_mm") is None:
        return 100.0
    limit_mm = float(toen_params["limit_mm"])
    return limit_mm / 0.6


def generate_buttocks_plot(
    out_dir: Path,
    v_plot: float,
    buttocks_params: dict,
    dt_s: float = TOEN_SOLVER_DT_S,
    duration_s: float = TOEN_SOLVER_DURATION_S,
    max_newton_iter: int = TOEN_SOLVER_MAX_NEWTON_ITER,
) -> Path:
    """Generate buttocks force/compression plot for a velocity."""
    subj = TOEN_SUBJECTS[TOEN_SUBJECT_ID]
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=MALE50_MASS_KG)

    k_butt = float(buttocks_params["k"])
    c_butt = float(buttocks_params["c"])
    limit_mm = float(buttocks_params["limit_mm"])
    stop_k = float(buttocks_params["stop_k"])
    smoothing_mm = float(buttocks_params["smoothing_mm"])

    compression_by_floor: dict[str, np.ndarray] = {}
    force_by_floor: dict[str, np.ndarray] = {}
    time_s = None

    for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
        _res, trace = simulate_toen_drop_trace(
            floor_name=floor_name,
            body_mass_kg=torso_mass,
            buttocks_k_n_per_m=k_butt,
            buttocks_c_ns_per_m=c_butt,
            floor_k_n_per_m=float(k_floor),
            impact_velocity_mps=v_plot,
            buttocks_limit_mm=limit_mm,
            buttocks_stop_k_n_per_m=stop_k,
            buttocks_stop_smoothing_mm=smoothing_mm,
            dt_s=dt_s,
            duration_s=duration_s,
            max_newton_iter=max_newton_iter,
        )
        if time_s is None:
            time_s = trace.time_s
        compression_by_floor[floor_name] = trace.buttocks_compression_m * 1000.0
        force_by_floor[floor_name] = trace.buttocks_force_n / 1000.0

    out_path = out_dir / f"buttocks_force_compression_v{v_plot:.2f}.png"
    plot_toen_buttocks_force_compression(
        time_s,
        compression_by_floor_mm=compression_by_floor,
        force_by_floor_kN=force_by_floor,
        out_path=out_path,
        title=f"Toen buttocks response (subject=avg, v={v_plot:.2f} m/s)",
    )
    return out_path
