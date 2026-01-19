from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .model import SimulationResult, SpineModel, initial_state_static, newmark_nonlinear


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
    params: dict
    success: bool
    cost: float
    residual_norm: float


def _simulate_peak_case(model: SpineModel, case: PeakCalibrationCase) -> SimulationResult:
    """
    Simulate a single peak calibration case, optionally performing a gravity settling phase
    (replicates the drop pipeline behavior for "flat"-style inputs).
    """
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

    return newmark_nonlinear(model, case.time_s, case.accel_g, y0, v0, s0)


def calibrate_model_peaks_joint(
    base_model: SpineModel,
    cases: list[PeakCalibrationCase],
    t12_element_index: int,
    *,
    init_params: dict,
    bounds: dict[str, tuple[float, float]],
    apply_params: Callable[[SpineModel, dict], SpineModel],
    max_nfev: int = 200,
) -> CalibrationResult:
    """
    Joint peak-based calibration for:
      - spine scales: s_k_spine, s_c_spine
      - buttocks absolute params: buttocks_k_n_per_m, buttocks_c_ns_per_m, buttocks_limit_mm

    All variables are optimized in log-space to enforce positivity.
    """
    keys = [
        's_k_spine',
        's_c_spine',
        'buttocks_k_n_per_m',
        'buttocks_c_ns_per_m',
        'buttocks_limit_mm',
    ]

    for k in keys:
        if k not in init_params:
            raise ValueError(f"Missing init param '{k}' for joint calibration.")
        if k not in bounds:
            raise ValueError(f"Missing bounds for param '{k}' for joint calibration.")

    x0 = np.log(np.array([float(init_params[k]) for k in keys], dtype=float))
    lb = np.log(np.array([float(bounds[k][0]) for k in keys], dtype=float))
    ub = np.log(np.array([float(bounds[k][1]) for k in keys], dtype=float))

    def residuals(logx: np.ndarray) -> np.ndarray:
        x = np.exp(logx)
        p = {keys[i]: float(x[i]) for i in range(len(keys))}
        model = apply_params(base_model, p)

        res = []
        for case in cases:
            sim = _simulate_peak_case(model, case)
            pred_peak = float(np.max(sim.element_forces_n[:, t12_element_index]))

            scale = max(float(abs(case.target_peak_force_n)), 1.0)
            res.append((pred_peak - case.target_peak_force_n) / scale)

        return np.asarray(res, dtype=float)

    out = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=max_nfev)

    x = np.exp(out.x)
    params = {keys[i]: float(x[i]) for i in range(len(keys))}

    return CalibrationResult(
        params=params,
        success=bool(out.success),
        cost=float(out.cost),
        residual_norm=float(np.linalg.norm(out.fun)),
    )


def calibrate_model_curves_joint(
    base_model: SpineModel,
    cases: list[CalibrationCase],
    t12_element_index: int,
    *,
    init_params: dict,
    bounds: dict[str, tuple[float, float]],
    apply_params: Callable[[SpineModel, dict], SpineModel],
    max_nfev: int = 200,
) -> CalibrationResult:
    """
    Joint curve-based calibration (waveform residuals) with same parameter set as peaks.
    """
    keys = [
        's_k_spine',
        's_c_spine',
        'buttocks_k_n_per_m',
        'buttocks_c_ns_per_m',
        'buttocks_limit_mm',
    ]

    for k in keys:
        if k not in init_params:
            raise ValueError(f"Missing init param '{k}' for joint calibration.")
        if k not in bounds:
            raise ValueError(f"Missing bounds for param '{k}' for joint calibration.")

    x0 = np.log(np.array([float(init_params[k]) for k in keys], dtype=float))
    lb = np.log(np.array([float(bounds[k][0]) for k in keys], dtype=float))
    ub = np.log(np.array([float(bounds[k][1]) for k in keys], dtype=float))

    def residuals(logx: np.ndarray) -> np.ndarray:
        x = np.exp(logx)
        p = {keys[i]: float(x[i]) for i in range(len(keys))}
        model = apply_params(base_model, p)

        res_all = []
        for case in cases:
            y0, v0, s0 = initial_state_static(model, base_accel_g0=0.0)
            sim = newmark_nonlinear(model, case.time_s, case.accel_g, y0, v0, s0)

            pred_force = sim.element_forces_n[:, t12_element_index]
            pred_force_interp = np.interp(case.force_time_s, sim.time_s, pred_force)

            scale = max(float(np.max(np.abs(case.force_n))), 1.0)
            res_all.append((pred_force_interp - case.force_n) / scale)

        return np.concatenate(res_all)

    out = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=max_nfev)

    x = np.exp(out.x)
    params = {keys[i]: float(x[i]) for i in range(len(keys))}

    return CalibrationResult(
        params=params,
        success=bool(out.success),
        cost=float(out.cost),
        residual_norm=float(np.linalg.norm(out.fun)),
    )
