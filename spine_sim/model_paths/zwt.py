from __future__ import annotations

import numpy as np

from spine_sim.calibration import (
    CalibrationCase,
    PeakCalibrationCase,
    CalibrationResult,
    calibrate_model,
    calibrate_model_peaks,
)
from spine_sim.model import SpineModel
from spine_sim.model_components import build_spine_elements


DEFAULT_SCALES = {
    "s_k_spine": 1.0,
    "s_c_spine": 1.0,
    "s_k_butt": 1.0,
    "s_c_butt": 1.0,
}


def build_model(mass_map: dict, config: dict) -> SpineModel:
    """
    ZWT path:
    - nonlinear polynomial spring (k + k2*x^2 + k3*x^3)
    - Kelvin-Voigt damping
    - two Maxwell branches (ratios + taus)

    Note: If k2/k3 are all zero, we fall back to the "multiplier at reference compression"
    cubic defined by (compression_ref_m, compression_k_mult), matching the Maxwell path style.
    """
    zwt_cfg = config.get("zwt", {})

    # Fallback: allow reuse of the Maxwell nonlinearity keys if the ZWT block doesn't define them.
    fallback_nl = config.get("maxwell", config.get("nonlinear", {}))

    c_base = float(zwt_cfg.get("c_base_ns_per_m", 1200.0))
    node_names, masses, element_names, k_elem, c_elem = build_spine_elements(mass_map, c_base)

    butt_gap_mm = float(zwt_cfg.get("buttocks_gap_mm", 0.0))

    # Optional polynomial coefficients
    disc_k2 = float(zwt_cfg.get("disc_poly_k2_n_per_m2", 0.0))
    disc_k3 = float(zwt_cfg.get("disc_poly_k3_n_per_m3", 0.0))
    butt_k2 = float(zwt_cfg.get("buttocks_poly_k2_n_per_m2", 0.0))
    butt_k3 = float(zwt_cfg.get("buttocks_poly_k3_n_per_m3", 0.0))

    # Reference-compression multiplier nonlinearity (used when poly_k3 is None)
    disc_ref_mm = float(zwt_cfg.get("disc_ref_compression_mm", fallback_nl.get("disc_ref_compression_mm", 2.0)))
    disc_kmult = float(zwt_cfg.get("disc_k_mult_at_ref", fallback_nl.get("disc_k_mult_at_ref", 8.0)))
    butt_ref_mm = float(zwt_cfg.get("buttocks_ref_compression_mm", fallback_nl.get("buttocks_ref_compression_mm", 25.0)))
    butt_kmult = float(zwt_cfg.get("buttocks_k_mult_at_ref", fallback_nl.get("buttocks_k_mult_at_ref", 20.0)))

    compression_ref_m = np.zeros_like(k_elem, dtype=float)
    compression_k_mult = np.ones_like(k_elem, dtype=float)
    tension_k_mult = np.ones_like(k_elem, dtype=float)
    compression_only = np.zeros_like(k_elem, dtype=bool)
    damping_compression_only = np.zeros_like(k_elem, dtype=bool)
    gap_m = np.zeros_like(k_elem, dtype=float)

    # Buttocks element (0)
    compression_only[0] = True
    damping_compression_only[0] = True
    gap_m[0] = butt_gap_mm / 1000.0
    compression_ref_m[0] = butt_ref_mm / 1000.0
    compression_k_mult[0] = butt_kmult

    # Spine elements (1:)
    compression_ref_m[1:] = disc_ref_mm / 1000.0
    compression_k_mult[1:] = disc_kmult

    # If all polynomial terms are zero, do NOT provide poly arrays, so model.py will use multiplier cubic.
    use_poly = any(abs(x) > 0.0 for x in [disc_k2, disc_k3, butt_k2, butt_k3])

    poly_k2 = None
    poly_k3 = None
    if use_poly:
        poly_k2 = np.zeros_like(k_elem, dtype=float)
        poly_k3 = np.zeros_like(k_elem, dtype=float)
        poly_k2[0] = butt_k2
        poly_k3[0] = butt_k3
        poly_k2[1:] = disc_k2
        poly_k3[1:] = disc_k3

    # Maxwell branches
    mx_k_ratios = zwt_cfg.get("maxwell_k_ratios", [1.0, 0.5])
    mx_tau_ms = zwt_cfg.get("maxwell_tau_ms", [10.0, 120.0])

    mx_k_ratios = [float(x) for x in mx_k_ratios]
    mx_tau_ms = [float(x) for x in mx_tau_ms]
    B = max(len(mx_k_ratios), len(mx_tau_ms))
    mx_k_ratios = (mx_k_ratios + [0.0] * B)[:B]
    mx_tau_ms = (mx_tau_ms + [0.0] * B)[:B]

    maxwell_k = np.zeros((len(k_elem), B), dtype=float)
    maxwell_tau_s = np.zeros((len(k_elem), B), dtype=float)

    for e in range(len(k_elem)):
        for b in range(B):
            maxwell_k[e, b] = k_elem[e] * mx_k_ratios[b]
            maxwell_tau_s[e, b] = mx_tau_ms[b] / 1000.0

    maxwell_compression_only = np.ones(len(k_elem), dtype=bool)

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
    return _apply_scales(
        base_model,
        scales["s_k_spine"],
        scales["s_c_spine"],
        scales["s_k_butt"],
        scales["s_c_butt"],
    )


def calibrate_peaks(
    base_model: SpineModel,
    cases: list[PeakCalibrationCase],
    t12_element_index: int,
    *,
    init_scales: dict | None = None,
    calibrate_damping: bool = False,
) -> CalibrationResult:
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
    return calibrate_model(
        base_model,
        cases,
        t12_element_index,
        init_scales=init_scales,
        apply_scales=_apply_scales,
    )
