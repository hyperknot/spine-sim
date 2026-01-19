"""Model path registry - builds spine models with different configurations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spine_sim.calibration import (
    CalibrationCase,
    CalibrationResult,
    PeakCalibrationCase,
    calibrate_model,
    calibrate_model_peaks,
)
from spine_sim.model import SpineModel
from spine_sim.model_components import build_spine_elements


DEFAULT_SCALES = {
    's_k_spine': 1.0,
    's_c_spine': 1.0,
    's_k_butt': 1.0,
    's_c_butt': 1.0,
}


def _build_spine_model(mass_map: dict, config: dict, model_key: str) -> SpineModel:
    """
    Build a spine model with nonlinear springs and Maxwell branches.

    Shared implementation for both 'maxwell' and 'zwt' model types.
    The only difference is which config section is read.

    Note: Buttocks parameters (k, c, densification) are set by Toen calibration
    via apply_toen_buttocks_to_model(). This function only sets structural flags.
    """
    cfg = config.get(model_key, config.get('maxwell', config.get('nonlinear', {})))

    c_base = float(cfg.get('c_base_ns_per_m', 1200.0))
    node_names, masses, element_names, k_elem, c_elem = build_spine_elements(mass_map, c_base)

    # Polynomial coefficients for spine (ZWT style) - buttocks uses Toen, not polynomials
    disc_k2 = float(cfg.get('disc_poly_k2_n_per_m2', 0.0))
    disc_k3 = float(cfg.get('disc_poly_k3_n_per_m3', 0.0))

    # Reference-compression multiplier nonlinearity for spine elements
    disc_ref_mm = float(cfg.get('disc_ref_compression_mm', 2.0))
    disc_kmult = float(cfg.get('disc_k_mult_at_ref', 8.0))

    n_elem = len(k_elem)
    compression_ref_m = np.zeros(n_elem, dtype=float)
    compression_k_mult = np.ones(n_elem, dtype=float)
    tension_k_mult = np.ones(n_elem, dtype=float)
    compression_only = np.zeros(n_elem, dtype=bool)
    damping_compression_only = np.zeros(n_elem, dtype=bool)
    gap_m = np.zeros(n_elem, dtype=float)

    # Buttocks element (index 0) - structural flags only, k/c/densification set by Toen
    compression_only[0] = True
    damping_compression_only[0] = True
    # gap_m[0], compression_ref_m[0], compression_k_mult[0] left at defaults (0, 0, 1)
    # since Toen uses densification (limit/stop_k) instead of multiplier nonlinearity

    # Spine elements (index 1+)
    compression_ref_m[1:] = disc_ref_mm / 1000.0
    compression_k_mult[1:] = disc_kmult

    # Polynomial arrays - only for spine elements if non-zero
    poly_k2 = None
    poly_k3 = None
    if any(abs(x) > 0.0 for x in [disc_k2, disc_k3]):
        poly_k2 = np.zeros(n_elem, dtype=float)
        poly_k3 = np.zeros(n_elem, dtype=float)
        # poly_k2[0], poly_k3[0] = 0 (buttocks uses Toen densification, not polynomial)
        poly_k2[1:], poly_k3[1:] = disc_k2, disc_k3

    # Maxwell branches
    mx_k_ratios = [float(x) for x in cfg.get('maxwell_k_ratios', [1.0, 0.5])]
    mx_tau_ms = [float(x) for x in cfg.get('maxwell_tau_ms', [10.0, 120.0])]
    B = max(len(mx_k_ratios), len(mx_tau_ms))
    mx_k_ratios = (mx_k_ratios + [0.0] * B)[:B]
    mx_tau_ms = (mx_tau_ms + [0.0] * B)[:B]

    maxwell_k = np.zeros((n_elem, B), dtype=float)
    maxwell_tau_s = np.zeros((n_elem, B), dtype=float)
    for e in range(n_elem):
        for b in range(B):
            maxwell_k[e, b] = k_elem[e] * mx_k_ratios[b]
            maxwell_tau_s[e, b] = mx_tau_ms[b] / 1000.0

    maxwell_compression_only = np.ones(n_elem, dtype=bool)

    return SpineModel(
        node_names=node_names,
        masses_kg=masses,
        element_names=element_names,
        k_elem=k_elem,
        c_elem=c_elem,
        compression_ref_m=compression_ref_m,
        compression_k_mult=compression_k_mult,
        tension_k_mult=tension_k_mult,
        compression_only=compression_only,
        damping_compression_only=damping_compression_only,
        gap_m=gap_m,
        maxwell_k=maxwell_k,
        maxwell_tau_s=maxwell_tau_s,
        maxwell_compression_only=maxwell_compression_only,
        poly_k2=poly_k2,
        poly_k3=poly_k3,
    )


def _apply_scales(
    base_model: SpineModel,
    s_k_spine: float,
    s_c_spine: float,
    s_k_butt: float,
    s_c_butt: float,
) -> SpineModel:
    """Apply calibration scale factors to model."""
    k = base_model.k_elem.copy()
    c = base_model.c_elem.copy()

    k[0] *= s_k_butt
    c[0] *= s_c_butt
    k[1:] *= s_k_spine
    c[1:] *= s_c_spine

    mx_k = base_model.maxwell_k.copy()
    if mx_k.size:
        mx_k[0, :] *= s_k_butt
        mx_k[1:, :] *= s_k_spine

    poly_k2 = None if base_model.poly_k2 is None else base_model.poly_k2.copy()
    poly_k3 = None if base_model.poly_k3 is None else base_model.poly_k3.copy()

    if poly_k2 is not None:
        poly_k2[0] *= s_k_butt
        poly_k2[1:] *= s_k_spine

    if poly_k3 is not None:
        poly_k3[0] *= s_k_butt
        poly_k3[1:] *= s_k_spine

    return SpineModel(
        node_names=base_model.node_names,
        masses_kg=base_model.masses_kg,
        element_names=base_model.element_names,
        k_elem=k,
        c_elem=c,
        compression_ref_m=base_model.compression_ref_m,
        compression_k_mult=base_model.compression_k_mult,
        tension_k_mult=base_model.tension_k_mult,
        compression_only=base_model.compression_only,
        damping_compression_only=base_model.damping_compression_only,
        gap_m=base_model.gap_m,
        maxwell_k=mx_k,
        maxwell_tau_s=base_model.maxwell_tau_s,
        maxwell_compression_only=base_model.maxwell_compression_only,
        poly_k2=poly_k2,
        poly_k3=poly_k3,
        compression_limit_m=base_model.compression_limit_m,
        compression_stop_k=base_model.compression_stop_k,
        compression_stop_smoothing_m=base_model.compression_stop_smoothing_m,
    )


def apply_calibration(base_model: SpineModel, scales: dict) -> SpineModel:
    """Apply calibration scales to a model."""
    return _apply_scales(
        base_model,
        scales['s_k_spine'],
        scales['s_c_spine'],
        scales['s_k_butt'],
        scales['s_c_butt'],
    )


def calibrate_peaks(
    base_model: SpineModel,
    cases: list[PeakCalibrationCase],
    t12_element_index: int,
    *,
    init_scales: dict | None = None,
    calibrate_damping: bool = False,
) -> CalibrationResult:
    """Peak-based calibration."""
    return calibrate_model_peaks(
        base_model,
        cases,
        t12_element_index,
        init_scales=init_scales,
        calibrate_damping=calibrate_damping,
        apply_scales=_apply_scales,
    )


def calibrate_curves(
    base_model: SpineModel,
    cases: list[CalibrationCase],
    t12_element_index: int,
    *,
    init_scales: dict | None = None,
) -> CalibrationResult:
    """Waveform-based calibration."""
    return calibrate_model(
        base_model,
        cases,
        t12_element_index,
        init_scales=init_scales,
        apply_scales=_apply_scales,
    )


@dataclass
class ModelPath:
    name: str
    build_model: Callable[[dict, dict], SpineModel]
    apply_calibration: Callable[[SpineModel, dict], SpineModel]
    calibrate_peaks: Callable[..., CalibrationResult]
    calibrate_curves: Callable[..., CalibrationResult]
    default_scales: dict


# Create model paths for each model type
MODEL_PATHS: dict[str, ModelPath] = {
    'maxwell': ModelPath(
        name='maxwell',
        build_model=lambda m, c: _build_spine_model(m, c, 'maxwell'),
        apply_calibration=apply_calibration,
        calibrate_peaks=calibrate_peaks,
        calibrate_curves=calibrate_curves,
        default_scales=DEFAULT_SCALES,
    ),
    'zwt': ModelPath(
        name='zwt',
        build_model=lambda m, c: _build_spine_model(m, c, 'zwt'),
        apply_calibration=apply_calibration,
        calibrate_peaks=calibrate_peaks,
        calibrate_curves=calibrate_curves,
        default_scales=DEFAULT_SCALES,
    ),
}


def get_model_path(name: str) -> ModelPath:
    """Get a model path by name."""
    key = name.strip().lower()
    if key not in MODEL_PATHS:
        valid = ', '.join(MODEL_PATHS.keys())
        raise ValueError(f"Unknown model type '{name}'. Available: {valid}")
    return MODEL_PATHS[key]
