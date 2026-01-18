from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from spine_sim.model import SpineModel, newmark_nonlinear


# Toen 2012 across-subject averages
TOEN_K_N_PER_M = 180_500.0
TOEN_C_NS_PER_M = 3_130.0

# Toen 2012: rigid floor peak ground force ~ 7.8 kN (their Fig. 4 average)
TOEN_RIGID_PEAK_FORCE_KN = 7.8


@dataclass(frozen=True)
class ButtocksOnlyCase:
    name: str
    time_s: np.ndarray
    accel_g: np.ndarray
    settle_ms: float


def get_buttocks_only_config(config: dict) -> dict:
    butt_cfg = config.get('buttocks_only', {})
    return {
        'k_n_per_m': float(butt_cfg.get('k_n_per_m', TOEN_K_N_PER_M)),
        'c_ns_per_m': float(butt_cfg.get('c_ns_per_m', TOEN_C_NS_PER_M)),
        'ref_compression_mm': float(butt_cfg.get('ref_compression_mm', 40.0)),
        'k_mult_at_ref': float(butt_cfg.get('k_mult_at_ref', 1.0)),
        'compression_limit_mm': float(butt_cfg.get('compression_limit_mm', 0.0)),
        'compression_stop_k_n_per_m': float(butt_cfg.get('compression_stop_k_n_per_m', 0.0)),
        'compression_stop_smoothing_mm': float(butt_cfg.get('compression_stop_smoothing_mm', 5.0)),
        'gap_mm': float(butt_cfg.get('gap_mm', 0.0)),
        'poly_k2_n_per_m2': float(butt_cfg.get('poly_k2_n_per_m2', 0.0)),
        'poly_k3_n_per_m3': float(butt_cfg.get('poly_k3_n_per_m3', 0.0)),
    }


def build_buttocks_only_model(torso_mass_kg: float, cfg: dict) -> SpineModel:
    node_names = ['pelvis']
    masses = np.array([torso_mass_kg], dtype=float)
    element_names = ['buttocks']

    k_elem = np.array([cfg['k_n_per_m']], dtype=float)
    c_elem = np.array([cfg['c_ns_per_m']], dtype=float)

    compression_ref_m = np.array([cfg['ref_compression_mm'] / 1000.0], dtype=float)
    compression_k_mult = np.array([cfg['k_mult_at_ref']], dtype=float)
    tension_k_mult = np.ones(1, dtype=float)
    compression_only = np.array([True], dtype=bool)
    damping_compression_only = np.array([True], dtype=bool)
    gap_m = np.array([cfg['gap_mm'] / 1000.0], dtype=float)

    poly_k2 = None
    poly_k3 = None
    if abs(cfg['poly_k2_n_per_m2']) > 0.0 or abs(cfg['poly_k3_n_per_m3']) > 0.0:
        poly_k2 = np.array([cfg['poly_k2_n_per_m2']], dtype=float)
        poly_k3 = np.array([cfg['poly_k3_n_per_m3']], dtype=float)

    compression_limit_m = None
    compression_stop_k = None
    compression_stop_smoothing_m = None
    if cfg['compression_limit_mm'] > 0.0 and cfg['compression_stop_k_n_per_m'] > 0.0:
        compression_limit_m = np.array([cfg['compression_limit_mm'] / 1000.0], dtype=float)
        compression_stop_k = np.array([cfg['compression_stop_k_n_per_m']], dtype=float)
        compression_stop_smoothing_m = np.array(
            [cfg['compression_stop_smoothing_mm'] / 1000.0], dtype=float
        )

    maxwell_k = np.zeros((1, 0), dtype=float)
    maxwell_tau_s = np.zeros((1, 0), dtype=float)
    maxwell_compression_only = np.array([True], dtype=bool)

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
        compression_limit_m=compression_limit_m,
        compression_stop_k=compression_stop_k,
        compression_stop_smoothing_m=compression_stop_smoothing_m,
    )


def pulse_metrics(time_s: np.ndarray, accel_g: np.ndarray) -> dict:
    # Use median dt for robustness
    if time_s.size < 2:
        return {
            'duration_ms': 0.0,
            'dt_ms': 0.0,
            'peak_g': float(np.max(accel_g)) if accel_g.size else 0.0,
        }

    dt = float(np.median(np.diff(time_s)))
    duration = float(time_s[-1] - time_s[0])

    g0 = 9.80665

    # NumPy 2.x: trapezoid is the supported name (trapz removed)
    if hasattr(np, 'trapezoid'):
        integral_gs = float(np.trapezoid(accel_g, time_s))
    else:
        integral_gs = float(np.trapz(accel_g, time_s))  # pragma: no cover

    dv = float(g0 * integral_gs)  # m/s
    h = float((dv * dv) / (2.0 * g0)) if dv > 0 else 0.0

    return {
        'duration_ms': duration * 1000.0,
        'dt_ms': dt * 1000.0,
        'peak_g': float(np.max(accel_g)),
        'min_g': float(np.min(accel_g)),
        'delta_v_mps': dv,
        'equiv_height_m': h,
    }


def buttocks_force_components_from_sim(
    *,
    model: SpineModel,
    sim,
) -> dict:
    """
    Decompose buttocks-only force into:
      - linear/poly spring
      - Kelvin damper (closing-only)
      - bottom-out stop (softplus)
    Only valid/used for the 1-DOF buttocks-only model.
    """
    assert model.n_elems() == 1
    assert model.size() == 1

    k = float(model.k_elem[0])
    c = float(model.c_elem[0])
    gap = float(model.gap_m[0])

    # poly terms, if present
    k2 = float(model.poly_k2[0]) if model.poly_k2 is not None else 0.0
    k3 = float(model.poly_k3[0]) if model.poly_k3 is not None else 0.0

    limit = float(model.compression_limit_m[0]) if model.compression_limit_m is not None else None
    stop_k = float(model.compression_stop_k[0]) if model.compression_stop_k is not None else 0.0
    smooth = (
        float(model.compression_stop_smoothing_m[0])
        if model.compression_stop_smoothing_m is not None
        else 0.0
    )

    y = sim.y[:, 0]  # m
    v = sim.v[:, 0]  # m/s

    ext_eff = y + gap
    x = np.maximum(-ext_eff, 0.0)  # compression (m)
    xdot_closing = np.maximum(-v, 0.0)  # m/s, closing only

    f_s = k * x + k2 * x * x + k3 * x * x * x
    f_d = c * xdot_closing

    f_stop = np.zeros_like(f_s)
    if limit is not None and limit > 0.0 and stop_k > 0.0:
        if smooth <= 0.0:
            smooth = 1e-6
        z = (x - limit) / smooth
        # stable softplus
        softplus = np.where(z > 50.0, z, np.where(z < -50.0, np.exp(z), np.log1p(np.exp(z))))
        smooth_excess = smooth * softplus
        f_stop = stop_k * smooth_excess

    f_total = f_s + f_d + f_stop

    return {
        'max_spring_kN': float(np.max(f_s) / 1000.0),
        'max_damper_kN': float(np.max(f_d) / 1000.0),
        'max_stop_kN': float(np.max(f_stop) / 1000.0),
        'max_total_kN': float(np.max(f_total) / 1000.0),
        'max_comp_mm': float(np.max(x) * 1000.0),
        'limit_mm': (float(limit * 1000.0) if limit is not None else None),
        'overshoot_mm': (float((np.max(x) - limit) * 1000.0) if limit is not None else None),
        'peak_total_mismatch_kN': float(
            (np.max(sim.element_forces_n[:, 0]) - np.max(f_total)) / 1000.0
        ),
    }


def recommend_toen_buttocks_params(
    *, smoothing_mm: float = 5.0, stiffness_multiplier_at_limit: float = 20.0
) -> dict:
    """
    Build a 'Toen-consistent' buttocks config:
      - k,c from Toen 2012 averages
      - compression_limit inferred from rigid-floor peak force / k
      - stop_k chosen to raise tangent stiffness near limit by stiffness_multiplier_at_limit

    Note:
      With the softplus stop, tangent stiffness at x=limit adds ~0.5*stop_k
      (because sigmoid(0) = 0.5). So:
        k_eff(limit) ≈ k + 0.5*stop_k
      Choose stop_k so k_eff ≈ stiffness_multiplier_at_limit * k.
    """
    k = TOEN_K_N_PER_M
    c = TOEN_C_NS_PER_M
    limit_m = (TOEN_RIGID_PEAK_FORCE_KN * 1000.0) / k
    limit_mm = limit_m * 1000.0

    stop_k = max(0.0, (stiffness_multiplier_at_limit * k - k) * 2.0)

    return {
        'k_n_per_m': float(k),
        'c_ns_per_m': float(c),
        'ref_compression_mm': 40.0,
        'k_mult_at_ref': 1.0,
        'compression_limit_mm': float(limit_mm),
        'compression_stop_k_n_per_m': float(stop_k),
        'compression_stop_smoothing_mm': float(smoothing_mm),
        'gap_mm': 0.0,
        'poly_k2_n_per_m2': 0.0,
        'poly_k3_n_per_m3': 0.0,
    }


def calibrate_buttocks_bottom_out(
    *,
    torso_mass_kg: float,
    cases: list[ButtocksOnlyCase],
    init_cfg: dict,
    bottom_out_case_name: str | None = None,
    bottom_out_fraction: float = 0.98,
    overshoot_soft_mm: float = 0.5,
) -> dict:
    """
    Tune bottom-out behavior while keeping Toe(n) k,c fixed.

    Variables we tune (log-space where positive):
      - compression_stop_k_n_per_m
      - compression_stop_smoothing_mm
      - compression_limit_mm (weakly, near Toe(n)-inferred)

    Objective:
      - For one chosen case: max compression ≈ bottom_out_fraction * limit.
      - For all cases: strongly discourage compression beyond limit (smooth hinge).
    """
    if not cases:
        raise ValueError('No cases provided for buttocks calibration.')

    cfg0 = dict(init_cfg)

    if bottom_out_case_name is None:
        # pick the case with the highest peak g
        peaks = [(c.name, float(np.max(c.accel_g))) for c in cases]
        bottom_out_case_name = sorted(peaks, key=lambda x: x[1], reverse=True)[0][0]

    limit0 = float(cfg0['compression_limit_mm'])
    stop_k0 = max(float(cfg0['compression_stop_k_n_per_m']), 1.0)
    smooth0 = max(float(cfg0['compression_stop_smoothing_mm']), 0.5)

    x0 = np.log(np.array([stop_k0, smooth0, limit0], dtype=float))

    # Bounds:
    #  - stop_k: 1e5 .. 1e8 N/m
    #  - smoothing: 0.5 .. 20 mm
    #  - limit: 30 .. 70 mm
    lb = np.log(np.array([1e5, 0.5, 30.0], dtype=float))
    ub = np.log(np.array([1e8, 20.0, 70.0], dtype=float))

    def soft_hinge_mm(x_mm: float, scale_mm: float) -> float:
        # softplus(x/scale) * scale
        z = x_mm / max(scale_mm, 1e-6)
        if z > 50.0:
            return z * scale_mm
        if z < -50.0:
            return float(np.exp(z) * scale_mm)
        return float(np.log1p(np.exp(z)) * scale_mm)

    def residuals(logx: np.ndarray) -> np.ndarray:
        stop_k, smooth_mm, limit_mm = np.exp(logx)

        cfg = dict(cfg0)
        cfg['compression_stop_k_n_per_m'] = float(stop_k)
        cfg['compression_stop_smoothing_mm'] = float(smooth_mm)
        cfg['compression_limit_mm'] = float(limit_mm)

        model = build_buttocks_only_model(torso_mass_kg, cfg)

        res: list[float] = []

        # Weak prior: keep limit near Toe(n)-inferred limit0
        res.append((limit_mm - limit0) / 5.0)  # 5 mm scale

        # Evaluate cases
        for case in cases:
            y0 = np.zeros(model.size(), dtype=float)
            v0 = np.zeros(model.size(), dtype=float)
            s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

            # optional settle
            if case.settle_ms > 0.0 and case.time_s.size >= 2:
                dt = float(np.median(np.diff(case.time_s)))
                n_settle = int(round((case.settle_ms / 1000.0) / dt)) + 1
                t_settle = dt * np.arange(n_settle)
                a_settle = np.zeros_like(t_settle)
                sim_settle = newmark_nonlinear(model, t_settle, a_settle, y0, v0, s0)
                y0 = sim_settle.y[-1].copy()
                v0 = sim_settle.v[-1].copy()
                s0 = sim_settle.maxwell_state_n[-1].copy()

            sim = newmark_nonlinear(model, case.time_s, case.accel_g, y0, v0, s0)
            comp_mm = float(np.max(np.maximum(-(sim.y[:, 0] + model.gap_m[0]), 0.0)) * 1000.0)

            # Overshoot penalty for everyone (smooth hinge)
            overshoot_mm = comp_mm - limit_mm
            res.append(
                soft_hinge_mm(overshoot_mm, overshoot_soft_mm) / 2.0
            )  # 2 mm scale after hinge

            # Force near-bottom-out on the chosen case
            if case.name == bottom_out_case_name:
                target_mm = bottom_out_fraction * limit_mm
                res.append((comp_mm - target_mm) / 2.0)  # 2 mm scale

        return np.asarray(res, dtype=float)

    result = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=150)

    stop_k, smooth_mm, limit_mm = np.exp(result.x)

    cfg_final = dict(cfg0)
    cfg_final['compression_stop_k_n_per_m'] = float(stop_k)
    cfg_final['compression_stop_smoothing_mm'] = float(smooth_mm)
    cfg_final['compression_limit_mm'] = float(limit_mm)
    cfg_final['_calibration'] = {
        'bottom_out_case_name': bottom_out_case_name,
        'bottom_out_fraction': float(bottom_out_fraction),
        'success': bool(result.success),
        'cost': float(result.cost),
        'residual_norm': float(np.linalg.norm(result.fun)),
    }
    return cfg_final
