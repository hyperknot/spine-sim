from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from spine_sim.model import SpineModel, initial_state_static, newmark_nonlinear
from spine_sim.toen_targets import (
    TOEN_FLOOR_STIFFNESS_N_PER_M,
    TOEN_GROUND_PEAKS_KN_AVG_PAPER,
)


# ----------------------------
# Toen 2012 Table 1 masses (50th male)
# Pelvis + L4 + L5 + Sacrum + Upper body + Skin over ITs
# ----------------------------
TOEN_TABLE1_MASSES_50TH_KG = {
    'pelvis': 16.0,
    'l4': 2.5,
    'l5': 1.8,
    'sacrum': 0.7,
    'upper_body': 33.0,
    'skin': 0.01,
}

TOEN_TABLE1_TORSO_TOTAL_MASS_KG = float(sum(TOEN_TABLE1_MASSES_50TH_KG.values()))
TOEN_TABLE1_SKIN_MASS_KG = float(TOEN_TABLE1_MASSES_50TH_KG['skin'])
TOEN_TABLE1_BODY_MASS_KG = float(TOEN_TABLE1_TORSO_TOTAL_MASS_KG - TOEN_TABLE1_SKIN_MASS_KG)

# ----------------------------
# Hard-coded Toen run settings
# ----------------------------
TOEN_IMPACT_V_MPS = 3.5
TOEN_TARGETS = TOEN_GROUND_PEAKS_KN_AVG_PAPER

# Hard-coded solver settings (used by both calibration and simulation unless overridden by caller).
TOEN_SOLVER_DT_S = 0.0005
TOEN_SOLVER_DURATION_S = 0.15
TOEN_SOLVER_MAX_NEWTON_ITER = 10

# Hard-coded calibration behavior.
TOEN_CALIB_FLOORS = ['firm_95', 'rigid_400']

TOEN_FLOOR_THICKNESS_MM: dict[str, float | None] = {
    'soft_59': 105.0,
    'medium_67': 75.0,
    'firm_95': 45.0,
    'rigid_400': None,
}


@dataclass(frozen=True)
class ToenDropResult:
    floor_name: str
    floor_k_n_per_m: float
    impact_velocity_mps: float

    peak_ground_kN: float
    t_peak_ms: float
    peak_buttocks_kN: float

    max_buttocks_comp_mm: float
    max_floor_comp_mm: float

    # Sitting/static reference (gravity-settled)
    static_buttocks_comp_mm: float
    delta_buttocks_comp_mm: float

    buttocks_overshoot_mm: float | None
    floor_overshoot_mm: float | None


@dataclass(frozen=True)
class ToenDropTrace:
    """
    Full time history for a single Toen drop simulation (2-DOF surrogate).
    """

    time_s: np.ndarray
    y_skin_m: np.ndarray
    y_body_m: np.ndarray
    floor_force_n: np.ndarray
    buttocks_force_n: np.ndarray
    floor_compression_m: np.ndarray
    buttocks_compression_m: np.ndarray


def build_toen_drop_model(
    *,
    body_mass_kg: float,
    skin_mass_kg: float,
    buttocks_k_n_per_m: float,
    buttocks_c_ns_per_m: float,
    floor_k_n_per_m: float,
    buttocks_limit_mm: float | None = None,
    buttocks_stop_k_n_per_m: float = 0.0,
    buttocks_stop_smoothing_mm: float = 1.0,
) -> SpineModel:
    node_names = ['skin', 'body']
    masses = np.array([skin_mass_kg, body_mass_kg], dtype=float)

    element_names = ['floor', 'buttocks']
    k_elem = np.array([floor_k_n_per_m, buttocks_k_n_per_m], dtype=float)
    c_elem = np.array([0.0, buttocks_c_ns_per_m], dtype=float)

    compression_ref_m = np.array([0.01, 0.04], dtype=float)
    compression_k_mult = np.array([1.0, 1.0], dtype=float)
    tension_k_mult = np.ones(2, dtype=float)

    compression_only = np.array([True, True], dtype=bool)
    damping_compression_only = np.array([False, True], dtype=bool)
    gap_m = np.array([0.0, 0.0], dtype=float)

    maxwell_k = np.zeros((2, 0), dtype=float)
    maxwell_tau_s = np.zeros((2, 0), dtype=float)
    maxwell_compression_only = np.ones(2, dtype=bool)

    compression_limit_m = None
    compression_stop_k = None
    compression_stop_smoothing_m = None

    if buttocks_limit_mm is not None and buttocks_stop_k_n_per_m > 0.0:
        compression_limit_m = np.zeros(2, dtype=float)
        compression_stop_k = np.zeros(2, dtype=float)
        compression_stop_smoothing_m = np.zeros(2, dtype=float)

        compression_limit_m[1] = float(buttocks_limit_mm) / 1000.0
        compression_stop_k[1] = float(buttocks_stop_k_n_per_m)
        compression_stop_smoothing_m[1] = float(buttocks_stop_smoothing_mm) / 1000.0

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
        compression_limit_m=compression_limit_m,
        compression_stop_k=compression_stop_k,
        compression_stop_smoothing_m=compression_stop_smoothing_m,
    )


def _buttocks_compression_mm(y_skin: np.ndarray, y_body: np.ndarray) -> float:
    x_butt = np.maximum(-(y_body - y_skin), 0.0)
    return float(np.max(x_butt) * 1000.0)


def _floor_compression_mm(y_skin: np.ndarray) -> float:
    x_floor = np.maximum(-y_skin, 0.0)
    return float(np.max(x_floor) * 1000.0)


def _static_buttocks_comp_mm(model: SpineModel) -> float:
    y_stat, _v, _s = initial_state_static(model, base_accel_g0=0.0)
    y_skin = np.asarray([y_stat[0]], dtype=float)
    y_body = np.asarray([y_stat[1]], dtype=float)
    x_stat = np.maximum(-(y_body - y_skin), 0.0)
    return float(x_stat[0] * 1000.0)


def simulate_toen_drop_trace(
    *,
    floor_name: str,
    body_mass_kg: float,
    skin_mass_kg: float,
    buttocks_k_n_per_m: float,
    buttocks_c_ns_per_m: float,
    floor_k_n_per_m: float,
    impact_velocity_mps: float,
    buttocks_limit_mm: float | None = None,
    buttocks_stop_k_n_per_m: float = 0.0,
    buttocks_stop_smoothing_mm: float = 1.0,
    dt_s: float = TOEN_SOLVER_DT_S,
    duration_s: float = TOEN_SOLVER_DURATION_S,
    max_newton_iter: int = TOEN_SOLVER_MAX_NEWTON_ITER,
) -> tuple[ToenDropResult, ToenDropTrace]:
    model = build_toen_drop_model(
        body_mass_kg=body_mass_kg,
        skin_mass_kg=skin_mass_kg,
        buttocks_k_n_per_m=buttocks_k_n_per_m,
        buttocks_c_ns_per_m=buttocks_c_ns_per_m,
        floor_k_n_per_m=floor_k_n_per_m,
        buttocks_limit_mm=buttocks_limit_mm,
        buttocks_stop_k_n_per_m=buttocks_stop_k_n_per_m,
        buttocks_stop_smoothing_mm=buttocks_stop_smoothing_mm,
    )

    static_butt_mm = _static_buttocks_comp_mm(model)

    t = np.arange(0.0, duration_s + dt_s, dt_s, dtype=float)
    base_accel_g = np.zeros_like(t)

    y0 = np.zeros(model.size(), dtype=float)
    v0 = np.zeros(model.size(), dtype=float)
    v0[0] = -float(impact_velocity_mps)
    v0[1] = -float(impact_velocity_mps)
    s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

    sim = newmark_nonlinear(
        model,
        t,
        base_accel_g,
        y0,
        v0,
        s0,
        max_newton_iter=max_newton_iter,
    )

    f_ground = sim.element_forces_n[:, 0]
    idx = int(np.argmax(f_ground))
    peak_ground_kN = float(f_ground[idx] / 1000.0)
    t_peak_ms = float(sim.time_s[idx] * 1000.0)

    f_butt = sim.element_forces_n[:, 1]
    peak_butt_kN = float(np.max(f_butt) / 1000.0)

    y_skin = sim.y[:, 0]
    y_body = sim.y[:, 1]

    max_floor_mm = _floor_compression_mm(y_skin)
    max_butt_mm = _buttocks_compression_mm(y_skin, y_body)
    delta_butt_mm = float(max_butt_mm - static_butt_mm)

    butt_overshoot_mm = None
    if buttocks_limit_mm is not None:
        butt_overshoot_mm = max(0.0, max_butt_mm - float(buttocks_limit_mm))

    result = ToenDropResult(
        floor_name=floor_name,
        floor_k_n_per_m=float(floor_k_n_per_m),
        impact_velocity_mps=float(impact_velocity_mps),
        peak_ground_kN=peak_ground_kN,
        t_peak_ms=t_peak_ms,
        peak_buttocks_kN=float(peak_butt_kN),
        max_buttocks_comp_mm=max_butt_mm,
        max_floor_comp_mm=max_floor_mm,
        static_buttocks_comp_mm=float(static_butt_mm),
        delta_buttocks_comp_mm=float(delta_butt_mm),
        buttocks_overshoot_mm=butt_overshoot_mm,
        floor_overshoot_mm=None,
    )

    floor_comp = np.maximum(-y_skin, 0.0)
    butt_comp = np.maximum(-(y_body - y_skin), 0.0)

    trace = ToenDropTrace(
        time_s=np.asarray(sim.time_s, dtype=float),
        y_skin_m=np.asarray(y_skin, dtype=float),
        y_body_m=np.asarray(y_body, dtype=float),
        floor_force_n=np.asarray(sim.element_forces_n[:, 0], dtype=float),
        buttocks_force_n=np.asarray(sim.element_forces_n[:, 1], dtype=float),
        floor_compression_m=np.asarray(floor_comp, dtype=float),
        buttocks_compression_m=np.asarray(butt_comp, dtype=float),
    )

    return result, trace


def simulate_toen_drop(**kwargs) -> ToenDropResult:
    result, _trace = simulate_toen_drop_trace(**kwargs)
    return result


def run_toen_suite(
    *,
    impact_velocities_mps: list[float],
    dt_s: float = TOEN_SOLVER_DT_S,
    duration_s: float = TOEN_SOLVER_DURATION_S,
    max_newton_iter: int = TOEN_SOLVER_MAX_NEWTON_ITER,
    buttocks_k_n_per_m: float,
    buttocks_c_ns_per_m: float,
    buttocks_limit_mm: float | None,
    buttocks_stop_k_n_per_m: float,
    buttocks_stop_smoothing_mm: float,
    warn_buttocks_comp_mm: float = 60.0,
) -> list[ToenDropResult]:
    targets = TOEN_TARGETS

    skin_mass = TOEN_TABLE1_SKIN_MASS_KG
    body_mass = TOEN_TABLE1_BODY_MASS_KG
    torso_total = TOEN_TABLE1_TORSO_TOTAL_MASS_KG

    print('\n=== TOEN DROP SUITE (2-DOF surrogate) ===')
    print(f'Table1 masses (kg): {json.dumps(TOEN_TABLE1_MASSES_50TH_KG)}')
    print(f'Torso total (Table1 sum): {torso_total:.2f} kg')
    print(f'Body mass: {body_mass:.2f} kg, Skin mass: {skin_mass:.3f} kg')
    print(f'Buttocks: k={buttocks_k_n_per_m:.1f} N/m, c={buttocks_c_ns_per_m:.1f} Ns/m')
    print(
        f'Buttocks densification: limit={buttocks_limit_mm}, stop_k={buttocks_stop_k_n_per_m:.3g}, smoothing={buttocks_stop_smoothing_mm}'
    )
    print(f'Targets (avg paper): {json.dumps(targets)}')
    print(f'Velocities requested (m/s): {impact_velocities_mps}')
    print(
        f'Solver: dt={dt_s * 1000.0:.3f} ms, duration={duration_s * 1000.0:.1f} ms, max_newton_iter={max_newton_iter}'
    )

    results: list[ToenDropResult] = []

    for v in impact_velocities_mps:
        v = float(v)
        energy_j = 0.5 * torso_total * v * v
        print(f'\n--- Impact velocity: {v:.2f} m/s, KE≈{energy_j:.1f} J ---')

        show_targets = abs(v - TOEN_IMPACT_V_MPS) < 1e-6

        for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
            r = simulate_toen_drop(
                floor_name=floor_name,
                body_mass_kg=body_mass,
                skin_mass_kg=skin_mass,
                buttocks_k_n_per_m=buttocks_k_n_per_m,
                buttocks_c_ns_per_m=buttocks_c_ns_per_m,
                floor_k_n_per_m=k_floor,
                impact_velocity_mps=v,
                buttocks_limit_mm=buttocks_limit_mm,
                buttocks_stop_k_n_per_m=buttocks_stop_k_n_per_m,
                buttocks_stop_smoothing_mm=buttocks_stop_smoothing_mm,
                dt_s=dt_s,
                duration_s=duration_s,
                max_newton_iter=max_newton_iter,
            )

            warn = ''
            if r.max_buttocks_comp_mm >= warn_buttocks_comp_mm:
                warn += (
                    f'  WARNING: buttocks_comp={r.max_buttocks_comp_mm:.1f} mm >= '
                    f'{warn_buttocks_comp_mm:.1f} mm'
                )
            if r.buttocks_overshoot_mm is not None and r.buttocks_overshoot_mm > 0.1:
                warn += f'  WARNING: buttocks_overshoot={r.buttocks_overshoot_mm:.1f} mm'

            static_txt = (
                f' (static={r.static_buttocks_comp_mm:.2f} mm, Δ={r.delta_buttocks_comp_mm:.1f} mm)'
            )

            if show_targets:
                tgt = float(targets[floor_name])
                err = (r.peak_ground_kN - tgt) / tgt * 100.0
                target_txt = f' (target={tgt:.3f} kN, {err:+.1f}%)'
            else:
                target_txt = ''

            print(
                f'  {floor_name}: peak_ground={r.peak_ground_kN:.3f} kN'
                f'{target_txt}, t_peak={r.t_peak_ms:.1f} ms, '
                f'butt_comp={r.max_buttocks_comp_mm:.1f} mm{static_txt}, '
                f'floor_comp={r.max_floor_comp_mm:.1f} mm' + warn
            )

            results.append(r)

    return results


def calibrate_toen_buttocks_model(
    *,
    buttocks_stop_k_n_per_m: float,
    buttocks_stop_smoothing_mm: float,
    init_k_n_per_m: float,
    init_c_ns_per_m: float,
    init_limit_mm: float,
    bounds_k_n_per_m: tuple[float, float],
    bounds_c_ns_per_m: tuple[float, float],
    bounds_limit_mm: tuple[float, float],
    debug_every: int = 5,
) -> dict:
    """
    Toen surrogate calibration (2-DOF):
      - fixed Table1 torso mass (no scaling)
      - targets: avg paper
      - v=3.5 m/s
      - calib floors: firm_95 + rigid_400

    Bounds/initial values are supplied by caller (config.json).
    """
    v0_mps = TOEN_IMPACT_V_MPS
    calib_floors = list(TOEN_CALIB_FLOORS)
    targets = TOEN_TARGETS

    skin_mass = TOEN_TABLE1_SKIN_MASS_KG
    body_mass = TOEN_TABLE1_BODY_MASS_KG

    x0 = np.log(np.array([init_k_n_per_m, init_c_ns_per_m, init_limit_mm], dtype=float))

    lb = np.log(
        np.array([bounds_k_n_per_m[0], bounds_c_ns_per_m[0], bounds_limit_mm[0]], dtype=float)
    )
    ub = np.log(
        np.array([bounds_k_n_per_m[1], bounds_c_ns_per_m[1], bounds_limit_mm[1]], dtype=float)
    )

    eval_counter = {'n': 0}

    def residuals(logx: np.ndarray) -> np.ndarray:
        eval_counter['n'] += 1
        k_butt = float(np.exp(logx[0]))
        c_butt = float(np.exp(logx[1]))
        limit_mm = float(np.exp(logx[2]))

        res = []
        for floor_name in calib_floors:
            k_floor = float(TOEN_FLOOR_STIFFNESS_N_PER_M[floor_name])
            tgt = float(targets[floor_name])

            r = simulate_toen_drop(
                floor_name=floor_name,
                body_mass_kg=body_mass,
                skin_mass_kg=skin_mass,
                buttocks_k_n_per_m=k_butt,
                buttocks_c_ns_per_m=c_butt,
                floor_k_n_per_m=k_floor,
                impact_velocity_mps=v0_mps,
                buttocks_limit_mm=limit_mm,
                buttocks_stop_k_n_per_m=buttocks_stop_k_n_per_m,
                buttocks_stop_smoothing_mm=buttocks_stop_smoothing_mm,
                dt_s=TOEN_SOLVER_DT_S,
                duration_s=TOEN_SOLVER_DURATION_S,
                max_newton_iter=TOEN_SOLVER_MAX_NEWTON_ITER,
            )
            res.append((r.peak_ground_kN - tgt) / max(tgt, 1e-6))

        # Extra constraint: rigid_400 should reach the limit at 3.5 m/s.
        rigid_floor = float(TOEN_FLOOR_STIFFNESS_N_PER_M['rigid_400'])
        r_rigid = simulate_toen_drop(
            floor_name='rigid_400',
            body_mass_kg=body_mass,
            skin_mass_kg=skin_mass,
            buttocks_k_n_per_m=k_butt,
            buttocks_c_ns_per_m=c_butt,
            floor_k_n_per_m=rigid_floor,
            impact_velocity_mps=v0_mps,
            buttocks_limit_mm=limit_mm,
            buttocks_stop_k_n_per_m=buttocks_stop_k_n_per_m,
            buttocks_stop_smoothing_mm=buttocks_stop_smoothing_mm,
            dt_s=TOEN_SOLVER_DT_S,
            duration_s=TOEN_SOLVER_DURATION_S,
            max_newton_iter=TOEN_SOLVER_MAX_NEWTON_ITER,
        )
        res.append((r_rigid.max_buttocks_comp_mm - limit_mm) / 1.0)

        if (eval_counter['n'] % max(debug_every, 1)) == 0:
            rms = float(np.sqrt(np.mean(np.square(res))))
            print(
                '  DEBUG toen buttocks calib '
                f'eval={eval_counter["n"]}: '
                f'k={k_butt:.1f}, c={c_butt:.1f}, limit={limit_mm:.2f} mm, rms={rms:.6f}'
            )

        return np.asarray(res, dtype=float)

    out = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=60, verbose=2)

    k_butt = float(np.exp(out.x[0]))
    c_butt = float(np.exp(out.x[1]))
    limit_mm = float(np.exp(out.x[2]))

    print('\n=== CALIBRATION RESULT (TOEN surrogate) ===')
    print('  targets: avg paper')
    print(f'  body_mass_kg: {body_mass:.2f}, skin_mass_kg: {skin_mass:.3f}')
    print(f'  impact_velocity_mps: {v0_mps}')
    print(f'  calib_floors: {calib_floors}')
    print(f'  limit_bounds_mm: [{bounds_limit_mm[0]}, {bounds_limit_mm[1]}]')
    print(
        f'\n  Optimizer: success={out.success}, cost={out.cost:.6f}, '
        f'residual_norm={np.linalg.norm(out.fun):.6f}, nfev={out.nfev}'
    )

    return {
        'buttocks_k_n_per_m': float(k_butt),
        'buttocks_c_ns_per_m': float(c_butt),
        'buttocks_limit_mm': float(limit_mm),
        'buttocks_stop_k_n_per_m': float(buttocks_stop_k_n_per_m),
        'buttocks_stop_smoothing_mm': float(buttocks_stop_smoothing_mm),
    }
