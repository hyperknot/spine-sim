from __future__ import annotations

import numpy as np
from spine_sim.calibration import (
    CalibrationCase,
    CalibrationResult,
    PeakCalibrationCase,
    calibrate_model,
    calibrate_model_peaks,
)
from spine_sim.calibration import (
    apply_calibration as apply_maxwell_calibration,
)
from spine_sim.model import SpineModel
from spine_sim.model_components import build_spine_elements


DEFAULT_SCALES = {
    's_k_spine': 1.0,
    's_c_spine': 1.0,
    's_k_butt': 1.0,
    's_c_butt': 1.0,
}


def build_model(mass_map: dict, config: dict) -> SpineModel:
    """
    Maxwell path:
    - nonlinear compression (linear + cubic via ref multiplier)
    - Kelvin-Voigt damping
    - Maxwell branches (ratios + taus)
    """
    maxwell_cfg = config.get('maxwell')
    if maxwell_cfg is None:
        maxwell_cfg = config.get('nonlinear', {})

    c_base = float(maxwell_cfg.get('c_base_ns_per_m', 1200.0))
    node_names, masses, element_names, k_elem, c_elem = build_spine_elements(mass_map, c_base)

    # Nonlinear parameters
    disc_ref_mm = float(maxwell_cfg.get('disc_ref_compression_mm', 2.0))
    disc_kmult = float(maxwell_cfg.get('disc_k_mult_at_ref', 8.0))
    butt_ref_mm = float(maxwell_cfg.get('buttocks_ref_compression_mm', 25.0))
    butt_kmult = float(maxwell_cfg.get('buttocks_k_mult_at_ref', 20.0))
    butt_gap_mm = float(maxwell_cfg.get('buttocks_gap_mm', 0.0))

    compression_ref_m = np.zeros_like(k_elem, dtype=float)
    compression_k_mult = np.ones_like(k_elem, dtype=float)
    tension_k_mult = np.ones_like(k_elem, dtype=float)
    compression_only = np.zeros_like(k_elem, dtype=bool)
    damping_compression_only = np.zeros_like(k_elem, dtype=bool)
    gap_m = np.zeros_like(k_elem, dtype=float)

    compression_ref_m[0] = butt_ref_mm / 1000.0
    compression_k_mult[0] = butt_kmult
    compression_only[0] = True
    damping_compression_only[0] = True
    gap_m[0] = butt_gap_mm / 1000.0

    compression_ref_m[1:] = disc_ref_mm / 1000.0
    compression_k_mult[1:] = disc_kmult
    tension_k_mult[1:] = 1.0

    # Maxwell branches
    mx_k_ratios = maxwell_cfg.get('maxwell_k_ratios', [1.0, 0.5])
    mx_tau_ms = maxwell_cfg.get('maxwell_tau_ms', [10.0, 120.0])

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
    )


def apply_calibration(base_model: SpineModel, scales: dict) -> SpineModel:
    return apply_maxwell_calibration(base_model, scales)


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
    )
