from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .model import SpineModel, initial_state_static, newmark_nonlinear


@dataclass
class CalibrationCase:
    """A single calibration case with input acceleration and target force."""
    name: str
    time_s: np.ndarray
    accel_g: np.ndarray
    force_time_s: np.ndarray
    force_n: np.ndarray


@dataclass
class PeakCalibrationCase:
    """
    Peak-only calibration case.

    We calibrate to the peak of a specific element force (e.g., T12-L1).
    """
    name: str
    time_s: np.ndarray
    accel_g: np.ndarray
    target_peak_force_n: float
    settle_ms: float = 0.0


@dataclass
class CalibrationResult:
    scales: dict
    success: bool
    cost: float
    residual_norm: float


def _apply_scales(
    base_model: SpineModel,
    s_k_spine: float,
    s_c_spine: float,
    s_k_butt: float,
    s_c_butt: float,
) -> SpineModel:
    k = base_model.k_elem.copy()
    c = base_model.c_elem.copy()

    # Element 0 is buttocks
    k[0] *= s_k_butt
    c[0] *= s_c_butt

    # Spine elements
    k[1:] *= s_k_spine
    c[1:] *= s_c_spine

    # Maxwell branches scale with the same stiffness scalings
    mx_k = base_model.maxwell_k.copy()
    if mx_k.size:
        mx_k[0, :] *= s_k_butt
        mx_k[1:, :] *= s_k_spine

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
        poly_k2=base_model.poly_k2,
        poly_k3=base_model.poly_k3,
        compression_limit_m=base_model.compression_limit_m,
        compression_stop_k=base_model.compression_stop_k,
        compression_stop_smoothing_m=base_model.compression_stop_smoothing_m,
    )


def calibrate_model(
    base_model: SpineModel,
    cases: list[CalibrationCase],
    t12_element_index: int,
    init_scales: dict | None = None,
    *,
    apply_scales: Callable[[SpineModel, float, float, float, float], SpineModel] | None = None,
) -> CalibrationResult:
    """
    Waveform-based calibration using digitized force time histories.
    """
    if init_scales is None:
        init_scales = {
            "s_k_spine": 1.0,
            "s_c_spine": 1.0,
            "s_k_butt": 1.0,
            "s_c_butt": 1.0,
        }

    if apply_scales is None:
        apply_scales = _apply_scales

    x0 = np.log(
        np.array(
            [
                init_scales["s_k_spine"],
                init_scales["s_c_spine"],
                init_scales["s_k_butt"],
                init_scales["s_c_butt"],
            ],
            dtype=float,
        )
    )

    bounds = (np.log(0.05), np.log(20.0))

    def residuals(log_scales: np.ndarray) -> np.ndarray:
        s = np.exp(log_scales)
        model = apply_scales(base_model, s[0], s[1], s[2], s[3])

        res_all = []
        for case in cases:
            y0, v0, s0 = initial_state_static(model, base_accel_g0=0.0)
            sim = newmark_nonlinear(model, case.time_s, case.accel_g, y0, v0, s0)

            pred_force = sim.element_forces_n[:, t12_element_index]
            pred_force_interp = np.interp(case.force_time_s, sim.time_s, pred_force)

            scale = max(float(np.max(np.abs(case.force_n))), 1.0)
            res_all.append((pred_force_interp - case.force_n) / scale)

        return np.concatenate(res_all)

    result = least_squares(residuals, x0, bounds=bounds, max_nfev=200)

    s = np.exp(result.x)
    scales = {
        "s_k_spine": float(s[0]),
        "s_c_spine": float(s[1]),
        "s_k_butt": float(s[2]),
        "s_c_butt": float(s[3]),
    }

    return CalibrationResult(
        scales=scales,
        success=result.success,
        cost=float(result.cost),
        residual_norm=float(np.linalg.norm(result.fun)),
    )


def calibrate_model_peaks(
    base_model: SpineModel,
    cases: list[PeakCalibrationCase],
    t12_element_index: int,
    *,
    init_scales: dict | None = None,
    calibrate_damping: bool = False,
    apply_scales: Callable[[SpineModel, float, float, float, float], SpineModel] | None = None,
) -> CalibrationResult:
    """
    Peak-only calibration.

    With only peak targets, damping is usually weakly identifiable, so by default
    we calibrate only stiffness scales (s_k_spine, s_k_butt) and keep damping at 1.
    """
    if init_scales is None:
        init_scales = {
            "s_k_spine": 1.0,
            "s_c_spine": 1.0,
            "s_k_butt": 1.0,
            "s_c_butt": 1.0,
        }

    if apply_scales is None:
        apply_scales = _apply_scales

    if calibrate_damping:
        x0 = np.log(
            np.array(
                [
                    init_scales["s_k_spine"],
                    init_scales["s_c_spine"],
                    init_scales["s_k_butt"],
                    init_scales["s_c_butt"],
                ],
                dtype=float,
            )
        )
        bounds = (np.log(0.05), np.log(20.0))

        def unpack(s: np.ndarray) -> tuple[float, float, float, float]:
            return float(s[0]), float(s[1]), float(s[2]), float(s[3])

    else:
        x0 = np.log(
            np.array(
                [
                    init_scales["s_k_spine"],
                    init_scales["s_k_butt"],
                ],
                dtype=float,
            )
        )
        bounds = (np.log(0.05), np.log(20.0))

        def unpack(s: np.ndarray) -> tuple[float, float, float, float]:
            # keep damping fixed at 1.0
            return float(s[0]), 1.0, float(s[1]), 1.0

    def residuals(log_scales: np.ndarray) -> np.ndarray:
        s = np.exp(log_scales)
        s_k_spine, s_c_spine, s_k_butt, s_c_butt = unpack(s)
        model = apply_scales(base_model, s_k_spine, s_c_spine, s_k_butt, s_c_butt)

        res = []
        for case in cases:
            # Replicate your "flat-style" workflow: optional gravity settling then run pulse.
            y0 = np.zeros(model.size(), dtype=float)
            v0 = np.zeros(model.size(), dtype=float)
            s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

            if case.settle_ms > 0.0:
                dt = float(np.median(np.diff(case.time_s)))
                n_settle = int(round((case.settle_ms / 1000.0) / dt)) + 1
                t_settle = dt * np.arange(n_settle)
                a_settle = np.zeros_like(t_settle)
                sim_settle = newmark_nonlinear(model, t_settle, a_settle, y0, v0, s0)
                y0 = sim_settle.y[-1].copy()
                v0 = sim_settle.v[-1].copy()
                s0 = sim_settle.maxwell_state_n[-1].copy()

            sim = newmark_nonlinear(model, case.time_s, case.accel_g, y0, v0, s0)
            pred_peak = float(np.max(sim.element_forces_n[:, t12_element_index]))

            # Normalize residual by target so cases weigh similarly
            scale = max(float(abs(case.target_peak_force_n)), 1.0)
            res.append((pred_peak - case.target_peak_force_n) / scale)

        return np.asarray(res, dtype=float)

    result = least_squares(residuals, x0, bounds=bounds, max_nfev=200)

    s = np.exp(result.x)
    s_k_spine, s_c_spine, s_k_butt, s_c_butt = unpack(s)
    scales = {
        "s_k_spine": float(s_k_spine),
        "s_c_spine": float(s_c_spine),
        "s_k_butt": float(s_k_butt),
        "s_c_butt": float(s_c_butt),
    }

    return CalibrationResult(
        scales=scales,
        success=result.success,
        cost=float(result.cost),
        residual_norm=float(np.linalg.norm(result.fun)),
    )


def apply_calibration(base_model: SpineModel, scales: dict) -> SpineModel:
    return _apply_scales(
        base_model,
        scales["s_k_spine"],
        scales["s_c_spine"],
        scales["s_k_butt"],
        scales["s_c_butt"],
    )
