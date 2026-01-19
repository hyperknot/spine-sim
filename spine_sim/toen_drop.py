from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from scipy.optimize import least_squares

from spine_sim.model import SpineModel, initial_state_static, newmark_nonlinear
from spine_sim.toen_targets import (
    TOEN_FLOOR_STIFFNESS_N_PER_M,
    TOEN_GROUND_PEAKS_KN_AVG_PAPER,
)
from spine_sim.toen_subjects import (
    DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    TOEN_TABLE1_MASSES_50TH_KG,
    TOEN_SUBJECTS,
    subject_buttocks_kc,
    toen_torso_mass_50th_kg,
    toen_torso_mass_scaled_kg,
)

# ----------------------------
# Hard-coded Toen run settings
# ----------------------------
TOEN_IMPACT_V_MPS = 3.5
TOEN_SUBJECT_ID = "avg"
TOEN_TARGETS = TOEN_GROUND_PEAKS_KN_AVG_PAPER

# Hard-coded solver settings (used by both calibration and simulation unless overridden by caller).
TOEN_SOLVER_DT_S = 0.0005
TOEN_SOLVER_DURATION_S = 0.15
TOEN_SOLVER_MAX_NEWTON_ITER = 10

# Hard-coded calibration behavior.
TOEN_CALIB_FLOORS = ["firm_95", "rigid_400"]
TOEN_CALIB_LIMIT_BOUNDS_MM = (30.0, 60.0)

TOEN_FLOOR_THICKNESS_MM: dict[str, float | None] = {
    "soft_59": 105.0,
    "medium_67": 75.0,
    "firm_95": 45.0,
    "rigid_400": None,
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
    node_names = ["skin", "body"]
    masses = np.array([skin_mass_kg, body_mass_kg], dtype=float)

    element_names = ["floor", "buttocks"]
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
    buttocks_k_n_per_m: float,
    buttocks_c_ns_per_m: float,
    floor_k_n_per_m: float,
    impact_velocity_mps: float,
    buttocks_limit_mm: float | None = None,
    buttocks_stop_k_n_per_m: float = 0.0,
    buttocks_stop_smoothing_mm: float = 1.0,
    skin_mass_kg: float = TOEN_TABLE1_MASSES_50TH_KG["skin"],
    dt_s: float = TOEN_SOLVER_DT_S,
    duration_s: float = TOEN_SOLVER_DURATION_S,
    max_newton_iter: int = TOEN_SOLVER_MAX_NEWTON_ITER,
) -> tuple[ToenDropResult, ToenDropTrace]:
    """
    Full time-history simulation for plotting (force/compression vs time).
    """
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

    # Time-history signals for plotting
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
    """
    Summary-only simulation (kept for backwards compatibility).
    """
    result, _trace = simulate_toen_drop_trace(**kwargs)
    return result


def run_toen_suite(
    *,
    impact_velocities_mps: list[float],
    male50_mass_kg: float = DEFAULT_50TH_MALE_TOTAL_MASS_KG,
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
    subj = TOEN_SUBJECTS[TOEN_SUBJECT_ID]
    targets = TOEN_TARGETS

    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50_mass_kg)

    print("\n=== TOEN DROP SUITE ===")
    print(
        f"Subject: {TOEN_SUBJECT_ID}, subject_total_mass={subj.total_mass_kg:.2f} kg, height={subj.height_cm:.1f} cm"
    )
    print(f"Male50 total mass assumption: {male50_mass_kg:.2f} kg")
    print(
        f"Torso mass (Table1 sum scaled): {torso_mass:.2f} kg (Table1 sum={toen_torso_mass_50th_kg():.2f} kg)"
    )
    print(f"Buttocks: k={buttocks_k_n_per_m:.1f} N/m, c={buttocks_c_ns_per_m:.1f} Ns/m")
    print(
        f"Buttocks densification: limit={buttocks_limit_mm}, stop_k={buttocks_stop_k_n_per_m:.3g}, smoothing={buttocks_stop_smoothing_mm}"
    )
    print(f"Targets (avg paper): {json.dumps(targets)}")
    print(f"Velocities requested (m/s): {impact_velocities_mps}")
    print(
        f"Solver: dt={dt_s*1000.0:.3f} ms, duration={duration_s*1000.0:.1f} ms, max_newton_iter={max_newton_iter}"
    )

    results: list[ToenDropResult] = []

    for v in impact_velocities_mps:
        v = float(v)
        energy_j = 0.5 * torso_mass * v * v
        print(f"\n--- Impact velocity: {v:.2f} m/s, KE≈{energy_j:.1f} J ---")

        # Only show target errors for the canonical Toen velocity 3.5 m/s.
        show_targets = abs(v - TOEN_IMPACT_V_MPS) < 1e-6

        for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
            r = simulate_toen_drop(
                floor_name=floor_name,
                body_mass_kg=torso_mass,
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

            warn = ""
            if r.max_buttocks_comp_mm >= warn_buttocks_comp_mm:
                warn += (
                    f"  WARNING: buttocks_comp={r.max_buttocks_comp_mm:.1f} mm >= "
                    f"{warn_buttocks_comp_mm:.1f} mm"
                )
            if r.buttocks_overshoot_mm is not None and r.buttocks_overshoot_mm > 0.1:
                warn += f"  WARNING: buttocks_overshoot={r.buttocks_overshoot_mm:.1f} mm"

            static_txt = f" (static={r.static_buttocks_comp_mm:.2f} mm, Δ={r.delta_buttocks_comp_mm:.1f} mm)"

            if show_targets:
                tgt = float(targets[floor_name])
                err = (r.peak_ground_kN - tgt) / tgt * 100.0
                target_txt = f" (target={tgt:.3f} kN, {err:+.1f}%)"
            else:
                target_txt = ""

            print(
                f"  {floor_name}: peak_ground={r.peak_ground_kN:.3f} kN"
                f"{target_txt}, t_peak={r.t_peak_ms:.1f} ms, "
                f"butt_comp={r.max_buttocks_comp_mm:.1f} mm{static_txt}, "
                f"floor_comp={r.max_floor_comp_mm:.1f} mm"
                + warn
            )

            results.append(r)

    return results


def calibrate_toen_buttocks_model(
    *,
    male50_mass_kg: float = DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    buttocks_stop_k_n_per_m: float,
    buttocks_stop_smoothing_mm: float,
    k0_n_per_m: float = 180_500.0,
    c0_ns_per_m: float = 3_130.0,
    limit0_mm: float = 39.0,
    debug_every: int = 5,
) -> dict:
    """
    Simplified calibration:
      - always subject avg
      - always target avg (paper)
      - always velocity 3.5 m/s
      - always calib floors firm_95 + rigid_400
    """
    v0_mps = TOEN_IMPACT_V_MPS
    calib_floors = list(TOEN_CALIB_FLOORS)
    targets = TOEN_TARGETS

    subj = TOEN_SUBJECTS[TOEN_SUBJECT_ID]
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50_mass_kg)

    x0 = np.log(np.array([k0_n_per_m, c0_ns_per_m, limit0_mm], dtype=float))

    limit_bounds = TOEN_CALIB_LIMIT_BOUNDS_MM

    # Bounds:
    # k: 50k–800k N/m, c: 200–20k Ns/m, limit: bounded.
    lb = np.log(np.array([5.0e4, 2.0e2, float(limit_bounds[0])], dtype=float))
    ub = np.log(np.array([8.0e5, 2.0e4, float(limit_bounds[1])], dtype=float))

    eval_counter = {"n": 0}

    def residuals(logx: np.ndarray) -> np.ndarray:
        eval_counter["n"] += 1
        k_butt = float(np.exp(logx[0]))
        c_butt = float(np.exp(logx[1]))
        limit_mm = float(np.exp(logx[2]))

        res = []

        for floor_name in calib_floors:
            k_floor = float(TOEN_FLOOR_STIFFNESS_N_PER_M[floor_name])
            tgt = float(targets[floor_name])

            r = simulate_toen_drop(
                floor_name=floor_name,
                body_mass_kg=torso_mass,
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

        # Enforce: rigid_400 reaches the limit at 3.5 m/s
        rigid_floor = float(TOEN_FLOOR_STIFFNESS_N_PER_M["rigid_400"])
        r_rigid = simulate_toen_drop(
            floor_name="rigid_400",
            body_mass_kg=torso_mass,
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

        if (eval_counter["n"] % max(debug_every, 1)) == 0:
            rms = float(np.sqrt(np.mean(np.square(res))))
            print(
                "  DEBUG toen buttocks calib "
                f"eval={eval_counter['n']}: "
                f"k={k_butt:.1f}, c={c_butt:.1f}, limit={limit_mm:.2f} mm, rms={rms:.6f}"
            )

        return np.asarray(res, dtype=float)

    out = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=60, verbose=2)

    k_butt = float(np.exp(out.x[0]))
    c_butt = float(np.exp(out.x[1]))
    limit_mm = float(np.exp(out.x[2]))

    # Print calibration debug info to console
    print("\n=== CALIBRATION RESULT ===")
    print(f"  subject_id: {TOEN_SUBJECT_ID}")
    print("  targets: avg paper")
    print(f"  male50_mass_kg: {male50_mass_kg:.2f}")
    print(f"  subject_total_mass_kg: {subj.total_mass_kg:.2f}")
    print(f"  torso_mass_kg: {torso_mass:.2f}")
    print(f"  impact_velocity_mps: {v0_mps}")
    print(f"  calib_floors: {calib_floors}")
    print(f"  limit_bounds_mm: [{limit_bounds[0]}, {limit_bounds[1]}]")
    print(f"\n  Optimizer: success={out.success}, cost={out.cost:.6f}, "
          f"residual_norm={np.linalg.norm(out.fun):.6f}, nfev={out.nfev}")

    # Return only essential calibration parameters
    return {
        "buttocks_k_n_per_m": float(k_butt),
        "buttocks_c_ns_per_m": float(c_butt),
        "buttocks_limit_mm": float(limit_mm),
        "buttocks_stop_k_n_per_m": float(buttocks_stop_k_n_per_m),
        "buttocks_stop_smoothing_mm": float(buttocks_stop_smoothing_mm),
    }
