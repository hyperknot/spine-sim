from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from scipy.optimize import least_squares

from spine_sim.env import env_float, env_float_list, env_int, env_str
from spine_sim.model import SpineModel, initial_state_static, newmark_nonlinear
from spine_sim.toen_targets import (
    TOEN_FLOOR_STIFFNESS_N_PER_M,
    TOEN_GROUND_PEAKS_KN_AVG_PAPER,
    TOEN_GROUND_PEAKS_KN_AVG_MEASURED_FIG4,
    TOEN_GROUND_PEAKS_KN_SUBJECT3_FIG3_APPROX,
)
from spine_sim.toen_subjects import (
    DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    TOEN_TABLE1_MASSES_50TH_KG,
    TOEN_SUBJECTS,
    subject_buttocks_kc,
    toen_torso_mass_50th_kg,
    toen_torso_mass_scaled_kg,
)

TOEN_IMPACT_V_MPS = 3.5

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


def _pick_targets(target_set: str) -> dict[str, float]:
    key = target_set.strip().lower()
    if key in {"avg", "paper"}:
        return TOEN_GROUND_PEAKS_KN_AVG_PAPER
    if key in {"avg_measured", "fig4_measured"}:
        return TOEN_GROUND_PEAKS_KN_AVG_MEASURED_FIG4
    if key in {"subj3", "subject3", "fig3_subj3", "fig3"}:
        return TOEN_GROUND_PEAKS_KN_SUBJECT3_FIG3_APPROX
    raise ValueError("target_set must be one of: avg, avg_measured, subj3 (or fig3 alias).")


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


def simulate_toen_drop(
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
    dt_s: float = 0.0005,
    duration_s: float = 0.15,
    max_newton_iter: int = 10,
) -> ToenDropResult:
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

    return ToenDropResult(
        floor_name=floor_name,
        floor_k_n_per_m=float(floor_k_n_per_m),
        impact_velocity_mps=float(impact_velocity_mps),
        peak_ground_kN=peak_ground_kN,
        t_peak_ms=t_peak_ms,
        peak_buttocks_kN=peak_butt_kN,
        max_buttocks_comp_mm=max_butt_mm,
        max_floor_comp_mm=max_floor_mm,
        static_buttocks_comp_mm=float(static_butt_mm),
        delta_buttocks_comp_mm=float(delta_butt_mm),
        buttocks_overshoot_mm=butt_overshoot_mm,
        floor_overshoot_mm=None,
    )



def run_toen_suite(
    *,
    subject_id: str,
    target_set: str,
    male50_mass_kg: float = DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    impact_velocities_mps: list[float] | None = None,
    velocity_scale: float = 1.0,
    dt_s: float = 0.0005,
    duration_s: float = 0.15,
    max_newton_iter: int = 10,
    buttocks_k_n_per_m: float | None = None,
    buttocks_c_ns_per_m: float | None = None,
    buttocks_limit_mm: float | None = None,
    buttocks_stop_k_n_per_m: float = 0.0,
    buttocks_stop_smoothing_mm: float = 1.0,
    warn_buttocks_comp_mm: float = 60.0,
) -> list[ToenDropResult]:
    """
    ENV overrides supported (inside this library function):

      SPINE_SIM_TOEN_SUBJECT_ID
      SPINE_SIM_TOEN_TARGET_SET
      SPINE_SIM_TOEN_VELOCITIES_MPS
      SPINE_SIM_TOEN_VELOCITY_SCALE
      SPINE_SIM_TOEN_DT_S
      SPINE_SIM_TOEN_DURATION_S
      SPINE_SIM_TOEN_MAX_NEWTON_ITER

      SPINE_SIM_TOEN_BUTTOCKS_K_N_PER_M
      SPINE_SIM_TOEN_BUTTOCKS_C_NS_PER_M
      SPINE_SIM_TOEN_BUTTOCKS_LIMIT_MM
      SPINE_SIM_TOEN_BUTTOCKS_STOP_K_N_PER_M
      SPINE_SIM_TOEN_BUTTOCKS_STOP_SMOOTHING_MM
    """
    subject_id = env_str("SPINE_SIM_TOEN_SUBJECT_ID") or subject_id
    target_set = env_str("SPINE_SIM_TOEN_TARGET_SET") or target_set

    v_env = env_float_list("SPINE_SIM_TOEN_VELOCITIES_MPS")
    if v_env is not None:
        impact_velocities_mps = v_env

    velocity_scale = env_float("SPINE_SIM_TOEN_VELOCITY_SCALE") or velocity_scale
    dt_s = env_float("SPINE_SIM_TOEN_DT_S") or dt_s
    duration_s = env_float("SPINE_SIM_TOEN_DURATION_S") or duration_s
    max_newton_iter = env_int("SPINE_SIM_TOEN_MAX_NEWTON_ITER") or max_newton_iter

    buttocks_k_n_per_m = env_float("SPINE_SIM_TOEN_BUTTOCKS_K_N_PER_M") or buttocks_k_n_per_m
    buttocks_c_ns_per_m = env_float("SPINE_SIM_TOEN_BUTTOCKS_C_NS_PER_M") or buttocks_c_ns_per_m
    buttocks_limit_mm = env_float("SPINE_SIM_TOEN_BUTTOCKS_LIMIT_MM") or buttocks_limit_mm
    buttocks_stop_k_n_per_m = (
        env_float("SPINE_SIM_TOEN_BUTTOCKS_STOP_K_N_PER_M") or buttocks_stop_k_n_per_m
    )
    buttocks_stop_smoothing_mm = (
        env_float("SPINE_SIM_TOEN_BUTTOCKS_STOP_SMOOTHING_MM") or buttocks_stop_smoothing_mm
    )

    if impact_velocities_mps is None:
        impact_velocities_mps = [TOEN_IMPACT_V_MPS]

    targets = _pick_targets(target_set)
    subj = TOEN_SUBJECTS[subject_id]

    k_butt_subj, c_butt_subj = subject_buttocks_kc(subject_id)
    k_butt = float(k_butt_subj if buttocks_k_n_per_m is None else buttocks_k_n_per_m)
    c_butt = float(c_butt_subj if buttocks_c_ns_per_m is None else buttocks_c_ns_per_m)

    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50_mass_kg)

    print("\n=== TOEN DROP SUITE ===")
    print(
        f"Subject: {subject_id}, subject_total_mass={subj.total_mass_kg:.2f} kg, height={subj.height_cm:.1f} cm"
    )
    print(f"Male50 total mass assumption: {male50_mass_kg:.2f} kg")
    print(
        f"Torso mass (Table1 sum scaled): {torso_mass:.2f} kg (Table1 sum={toen_torso_mass_50th_kg():.2f} kg)"
    )
    print(f"Buttocks: k={k_butt:.1f} N/m, c={c_butt:.1f} Ns/m")
    print(
        f"Buttocks densification: limit={buttocks_limit_mm}, stop_k={buttocks_stop_k_n_per_m:.3g}, smoothing={buttocks_stop_smoothing_mm}"
    )
    print(f"Targets: {target_set} -> {json.dumps(targets)}")
    print(f"Velocity scale applied: {velocity_scale:.6g}")
    print(f"Velocities requested (m/s): {impact_velocities_mps}")
    print(
        f"Solver: dt={dt_s*1000.0:.3f} ms, duration={duration_s*1000.0:.1f} ms, max_newton_iter={max_newton_iter}"
    )

    results: list[ToenDropResult] = []

    for v in impact_velocities_mps:
        v = float(v)
        v_eff = v * float(velocity_scale)
        energy_j = 0.5 * torso_mass * v_eff * v_eff
        print(f"\n--- Impact velocity: {v:.2f} m/s (effective: {v_eff:.3f} m/s), KE≈{energy_j:.1f} J ---")

        # Only show (target=..., %+...) for the canonical Toen velocity 3.5 m/s.
        show_targets = abs(v - TOEN_IMPACT_V_MPS) < 1e-6

        for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
            r = simulate_toen_drop(
                floor_name=floor_name,
                body_mass_kg=torso_mass,
                buttocks_k_n_per_m=k_butt,
                buttocks_c_ns_per_m=c_butt,
                floor_k_n_per_m=k_floor,
                impact_velocity_mps=v_eff,
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

            # Static reference is ALWAYS shown (per your request).
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
    subject_id: str = "avg",
    target_set: str = "avg",
    male50_mass_kg: float = DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    v0_mps: float = TOEN_IMPACT_V_MPS,
    calib_floors: list[str] | None = None,
    buttocks_stop_k_n_per_m: float = 5.0e6,
    buttocks_stop_smoothing_mm: float = 1.0,
    k0_n_per_m: float = 180_500.0,
    c0_ns_per_m: float = 3_130.0,
    limit0_mm: float = 39.0,
    debug_every: int = 5,
) -> dict:
    """
    Buttocks calibration is ALWAYS performed at 3.5 m/s (Toen regime),
    regardless of caller inputs or environment.

    Optional ENV overrides:
      SPINE_SIM_TOEN_CALIB_FLOORS="firm_95,rigid_400"
      SPINE_SIM_TOEN_BUTTOCKS_STOP_K_N_PER_M
      SPINE_SIM_TOEN_BUTTOCKS_STOP_SMOOTHING_MM
      SPINE_SIM_TOEN_CALIB_LIMIT_BOUNDS_MM="30,45"
    """
    # Force calibration velocity to Toen regime
    v0_mps = TOEN_IMPACT_V_MPS

    floors_env = env_str("SPINE_SIM_TOEN_CALIB_FLOORS")
    if floors_env:
        calib_floors = [s.strip() for s in floors_env.replace(" ", ",").split(",") if s.strip()]

    buttocks_stop_k_n_per_m = (
        env_float("SPINE_SIM_TOEN_BUTTOCKS_STOP_K_N_PER_M") or buttocks_stop_k_n_per_m
    )
    buttocks_stop_smoothing_mm = (
        env_float("SPINE_SIM_TOEN_BUTTOCKS_STOP_SMOOTHING_MM") or buttocks_stop_smoothing_mm
    )

    # Limit bounds to prevent the optimizer from "escaping" by making the limit huge.
    limit_bounds = (30.0, 45.0)
    lb_env = env_str("SPINE_SIM_TOEN_CALIB_LIMIT_BOUNDS_MM")
    if lb_env:
        parts = [p.strip() for p in lb_env.replace(" ", ",").split(",") if p.strip()]
        if len(parts) == 2:
            limit_bounds = (float(parts[0]), float(parts[1]))

    if calib_floors is None:
        calib_floors = ["firm_95", "rigid_400"]

    targets = _pick_targets(target_set)
    subj = TOEN_SUBJECTS[subject_id]
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50_mass_kg)

    x0 = np.log(np.array([k0_n_per_m, c0_ns_per_m, limit0_mm], dtype=float))

    # Bounds:
    # k: 50k–800k N/m, c: 200–20k Ns/m, limit: bounded (default 30–45 mm).
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
                dt_s=0.0005,
                duration_s=0.15,
                max_newton_iter=10,
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
            dt_s=0.0005,
            duration_s=0.15,
            max_newton_iter=10,
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
        dt_s=0.0005,
        duration_s=0.15,
        max_newton_iter=10,
    )

    # Calibration-time-only reference reporting is done here (not in run_toen_suite).
    print("\n=== CALIBRATION REFERENCE (rigid_400 @ 3.5 m/s) ===")
    print(f"  static_buttocks_comp_mm: {r_rigid.static_buttocks_comp_mm:.3f}")
    print(f"  max_buttocks_comp_mm:    {r_rigid.max_buttocks_comp_mm:.3f}")
    print(f"  delta_comp_mm:           {r_rigid.delta_buttocks_comp_mm:.3f}")
    print(f"  limit_mm:                {limit_mm:.3f}")

    return {
        "subject_id": subject_id,
        "target_set": target_set,
        "male50_mass_kg": float(male50_mass_kg),
        "subject_total_mass_kg": float(subj.total_mass_kg),
        "torso_mass_kg": float(torso_mass),
        "impact_velocity_mps": float(v0_mps),
        "calib_floors": calib_floors,
        "buttocks_k_n_per_m": float(k_butt),
        "buttocks_c_ns_per_m": float(c_butt),
        "buttocks_limit_mm": float(limit_mm),
        "buttocks_stop_k_n_per_m": float(buttocks_stop_k_n_per_m),
        "buttocks_stop_smoothing_mm": float(buttocks_stop_smoothing_mm),
        "rigid_400_check": {
            "peak_ground_kN": float(r_rigid.peak_ground_kN),
            "t_peak_ms": float(r_rigid.t_peak_ms),
            "max_buttocks_comp_mm": float(r_rigid.max_buttocks_comp_mm),
            "static_buttocks_comp_mm": float(r_rigid.static_buttocks_comp_mm),
            "delta_buttocks_comp_mm": float(r_rigid.delta_buttocks_comp_mm),
        },
        "success": bool(out.success),
        "cost": float(out.cost),
        "residual_norm": float(np.linalg.norm(out.fun)),
        "nfev": int(out.nfev),
        "limit_bounds_mm": [float(limit_bounds[0]), float(limit_bounds[1])],
    }
