from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from scipy.optimize import least_squares

from spine_sim.model import SpineModel, newmark_nonlinear
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

# Foam mat thicknesses from Toen 2012 methods:
# soft: 10.5 cm, medium: 7.5 cm, firm: 4.5 cm.
TOEN_FLOOR_THICKNESS_MM: dict[str, float | None] = {
    "soft_59": 105.0,
    "medium_67": 75.0,
    "firm_95": 45.0,
    # "rigid" floor isn't a compressible mat; leave None (no stop) by default.
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
    # Optional smooth bottom-out (softplus stop)
    floor_limit_mm: float | None = None,
    buttocks_limit_mm: float | None = None,
    floor_stop_k_n_per_m: float = 0.0,
    buttocks_stop_k_n_per_m: float = 0.0,
    floor_stop_smoothing_mm: float = 3.0,
    buttocks_stop_smoothing_mm: float = 5.0,
) -> SpineModel:
    """
    2-node drop model:
      base --(floor spring)--> skin --(buttocks Voigt)--> body

    Ground reaction force = element 0 force ("floor").

    If *_limit_mm and *_stop_k_n_per_m are set, a smooth stop engages:
      F_stop = stop_k * smooth * softplus((x - limit)/smooth),
    which is smooth and avoids infinite jerk.
    """
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

    # Optional stops
    compression_limit_m = None
    compression_stop_k = None
    compression_stop_smoothing_m = None

    if (floor_limit_mm is not None and floor_stop_k_n_per_m > 0.0) or (
        buttocks_limit_mm is not None and buttocks_stop_k_n_per_m > 0.0
    ):
        compression_limit_m = np.zeros(2, dtype=float)
        compression_stop_k = np.zeros(2, dtype=float)
        compression_stop_smoothing_m = np.zeros(2, dtype=float)

        if floor_limit_mm is not None and floor_stop_k_n_per_m > 0.0:
            compression_limit_m[0] = float(floor_limit_mm) / 1000.0
            compression_stop_k[0] = float(floor_stop_k_n_per_m)
            compression_stop_smoothing_m[0] = float(floor_stop_smoothing_mm) / 1000.0

        if buttocks_limit_mm is not None and buttocks_stop_k_n_per_m > 0.0:
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


def simulate_toen_drop(
    *,
    floor_name: str,
    body_mass_kg: float,
    buttocks_k_n_per_m: float,
    buttocks_c_ns_per_m: float,
    floor_k_n_per_m: float,
    impact_velocity_mps: float,
    # Optional stops
    floor_limit_mm: float | None = None,
    buttocks_limit_mm: float | None = None,
    floor_stop_k_n_per_m: float = 0.0,
    buttocks_stop_k_n_per_m: float = 0.0,
    floor_stop_smoothing_mm: float = 3.0,
    buttocks_stop_smoothing_mm: float = 5.0,
    # Simulation controls
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
        floor_limit_mm=floor_limit_mm,
        buttocks_limit_mm=buttocks_limit_mm,
        floor_stop_k_n_per_m=floor_stop_k_n_per_m,
        buttocks_stop_k_n_per_m=buttocks_stop_k_n_per_m,
        floor_stop_smoothing_mm=floor_stop_smoothing_mm,
        buttocks_stop_smoothing_mm=buttocks_stop_smoothing_mm,
    )

    t = np.arange(0.0, duration_s + dt_s, dt_s, dtype=float)
    base_accel_g = np.zeros_like(t)

    # IMPORTANT: skin and body start with same downward impact velocity
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

    x_floor = np.maximum(-y_skin, 0.0)
    x_butt = np.maximum(-(y_body - y_skin), 0.0)

    max_floor_mm = float(np.max(x_floor) * 1000.0)
    max_butt_mm = float(np.max(x_butt) * 1000.0)

    floor_overshoot_mm = None
    if floor_limit_mm is not None:
        floor_overshoot_mm = max(0.0, max_floor_mm - float(floor_limit_mm))

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
        buttocks_overshoot_mm=butt_overshoot_mm,
        floor_overshoot_mm=floor_overshoot_mm,
    )


def run_toen_suite(
    *,
    subject_id: str,
    target_set: str,
    male50_mass_kg: float = DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    impact_velocities_mps: list[float] | None = None,
    velocity_scale: float = 1.0,
    # bottom-out controls
    enable_buttocks_bottomout: bool = False,
    buttocks_limit_mm: float = 45.0,
    buttocks_stop_k_n_per_m: float = 2.0e7,
    buttocks_stop_smoothing_mm: float = 5.0,
    enable_floor_bottomout: bool = False,
    floor_stop_k_n_per_m: float = 5.0e7,
    floor_stop_smoothing_mm: float = 3.0,
    warn_buttocks_comp_mm: float = 60.0,
) -> list[ToenDropResult]:
    if impact_velocities_mps is None:
        impact_velocities_mps = [TOEN_IMPACT_V_MPS]

    targets = _pick_targets(target_set)
    subj = TOEN_SUBJECTS[subject_id]
    k_butt, c_butt = subject_buttocks_kc(subject_id)
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50_mass_kg)

    print("\n=== TOEN DROP SUITE ===")
    print(f"Subject: {subject_id}, subject_total_mass={subj.total_mass_kg:.2f} kg, height={subj.height_cm:.1f} cm")
    print(f"Male50 total mass assumption: {male50_mass_kg:.2f} kg")
    print(f"Torso mass (Table1 sum scaled): {torso_mass:.2f} kg (Table1 sum={toen_torso_mass_50th_kg():.2f} kg)")
    print(f"Buttocks: k={k_butt:.1f} N/m, c={c_butt:.1f} Ns/m")
    print(f"Targets: {target_set} -> {json.dumps(targets)}")
    print(f"Velocity scale applied: {velocity_scale:.6g}")
    print(f"Velocities requested (m/s): {impact_velocities_mps}")

    if enable_buttocks_bottomout:
        print(
            "DEBUG bottom-out (buttocks): "
            f"limit={buttocks_limit_mm:.1f} mm, stop_k={buttocks_stop_k_n_per_m:.3g} N/m, smoothing={buttocks_stop_smoothing_mm:.1f} mm"
        )
    if enable_floor_bottomout:
        print(
            "DEBUG bottom-out (floor): "
            f"stop_k={floor_stop_k_n_per_m:.3g} N/m, smoothing={floor_stop_smoothing_mm:.1f} mm, thickness_map={TOEN_FLOOR_THICKNESS_MM}"
        )

    results: list[ToenDropResult] = []

    for v in impact_velocities_mps:
        v_eff = float(v) * float(velocity_scale)
        energy_j = 0.5 * torso_mass * v_eff * v_eff
        print(f"\n--- Impact velocity: {v:.2f} m/s (effective: {v_eff:.3f} m/s), KE≈{energy_j:.1f} J ---")

        for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
            floor_limit = None
            if enable_floor_bottomout:
                floor_limit = TOEN_FLOOR_THICKNESS_MM.get(floor_name, None)

            r = simulate_toen_drop(
                floor_name=floor_name,
                body_mass_kg=torso_mass,
                buttocks_k_n_per_m=k_butt,
                buttocks_c_ns_per_m=c_butt,
                floor_k_n_per_m=k_floor,
                impact_velocity_mps=v_eff,
                floor_limit_mm=floor_limit,
                buttocks_limit_mm=(buttocks_limit_mm if enable_buttocks_bottomout else None),
                floor_stop_k_n_per_m=(floor_stop_k_n_per_m if enable_floor_bottomout else 0.0),
                buttocks_stop_k_n_per_m=(buttocks_stop_k_n_per_m if enable_buttocks_bottomout else 0.0),
                floor_stop_smoothing_mm=floor_stop_smoothing_mm,
                buttocks_stop_smoothing_mm=buttocks_stop_smoothing_mm,
                max_newton_iter=12 if (enable_buttocks_bottomout or enable_floor_bottomout) else 8,
            )

            tgt = float(targets[floor_name])
            err = (r.peak_ground_kN - tgt) / tgt * 100.0

            warn = ""
            if r.max_buttocks_comp_mm >= warn_buttocks_comp_mm:
                warn += f"  WARNING: buttocks_comp={r.max_buttocks_comp_mm:.1f} mm >= {warn_buttocks_comp_mm:.1f} mm"
            if r.buttocks_overshoot_mm is not None and r.buttocks_overshoot_mm > 0.1:
                warn += f"  WARNING: buttocks_overshoot={r.buttocks_overshoot_mm:.1f} mm"
            if r.floor_overshoot_mm is not None and r.floor_overshoot_mm > 0.1:
                warn += f"  WARNING: floor_overshoot={r.floor_overshoot_mm:.1f} mm"

            print(
                f"  {floor_name}: peak_ground={r.peak_ground_kN:.3f} kN (target={tgt:.3f} kN, {err:+.1f}%), "
                f"t_peak={r.t_peak_ms:.1f} ms, butt_comp={r.max_buttocks_comp_mm:.1f} mm, floor_comp={r.max_floor_comp_mm:.1f} mm"
                + warn
            )
            results.append(r)

    return results


def calibrate_toen_velocity_scale(
    *,
    subject_id: str,
    target_set: str,
    male50_mass_kg: float = DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    v0_mps: float = TOEN_IMPACT_V_MPS,
    debug_every: int = 5,
) -> dict:
    targets = _pick_targets(target_set)
    subj = TOEN_SUBJECTS[subject_id]
    k_butt, c_butt = subject_buttocks_kc(subject_id)
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50_mass_kg)

    x0 = np.log(np.array([1.0], dtype=float))
    lb = np.log(np.array([0.6], dtype=float))
    ub = np.log(np.array([1.4], dtype=float))

    eval_counter = {"n": 0}

    def residuals(logx: np.ndarray) -> np.ndarray:
        eval_counter["n"] += 1
        vscale = float(np.exp(logx[0]))

        res = []
        for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
            tgt = float(targets[floor_name])
            r = simulate_toen_drop(
                floor_name=floor_name,
                body_mass_kg=torso_mass,
                buttocks_k_n_per_m=k_butt,
                buttocks_c_ns_per_m=c_butt,
                floor_k_n_per_m=k_floor,
                impact_velocity_mps=v0_mps * vscale,
                dt_s=0.0005,
                duration_s=0.15,
                max_newton_iter=8,
            )
            res.append((r.peak_ground_kN - tgt) / max(tgt, 1e-6))

        if (eval_counter["n"] % max(debug_every, 1)) == 0:
            rms = float(np.sqrt(np.mean(np.square(res))))
            print(f"  DEBUG toen calib eval={eval_counter['n']}: vscale={vscale:.6f}, rms_rel={rms:.6f}")

        return np.asarray(res, dtype=float)

    out = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=40, verbose=2)
    vscale = float(np.exp(out.x[0]))

    return {
        "subject_id": subject_id,
        "target_set": target_set,
        "male50_mass_kg": float(male50_mass_kg),
        "subject_total_mass_kg": float(subj.total_mass_kg),
        "torso_mass_kg": float(torso_mass),
        "buttocks_k_n_per_m": float(k_butt),
        "buttocks_c_ns_per_m": float(c_butt),
        "impact_velocity_mps": float(v0_mps),
        "velocity_scale": float(vscale),
        "success": bool(out.success),
        "cost": float(out.cost),
        "residual_norm": float(np.linalg.norm(out.fun)),
        "nfev": int(out.nfev),
    }
