from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from spine_sim.model import SpineModel, newmark_nonlinear
from spine_sim.toen_subjects import (
    DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    TOEN_SUBJECTS,
    TOEN_TABLE1_MASSES_50TH_KG,
    subject_buttocks_kc,
    toen_torso_mass_50th_kg,
    toen_torso_mass_scaled_kg,
)
from spine_sim.toen_targets import (
    TOEN_FLOOR_STIFFNESS_N_PER_M,
    TOEN_GROUND_PEAKS_KN_AVG_MEASURED_FIG4,
    TOEN_GROUND_PEAKS_KN_AVG_PAPER,
    TOEN_GROUND_PEAKS_KN_SUBJECT3_FIG3_APPROX,
)


TOEN_IMPACT_V_MPS = 3.5


@dataclass(frozen=True)
class ToenDropResult:
    floor_name: str
    floor_k_n_per_m: float
    peak_ground_kN: float
    t_peak_ms: float
    peak_buttocks_kN: float
    max_buttocks_comp_mm: float
    max_floor_comp_mm: float


def build_toen_drop_model(
    *,
    body_mass_kg: float,
    skin_mass_kg: float,
    buttocks_k_n_per_m: float,
    buttocks_c_ns_per_m: float,
    floor_k_n_per_m: float,
) -> SpineModel:
    """
    2-node model (effectively single mass because skin is tiny):

      base --(floor spring)--> skin --(buttocks Voigt)--> body

    Ground reaction force = force in element 0 ("floor").
    """
    node_names = ['skin', 'body']
    masses = np.array([skin_mass_kg, body_mass_kg], dtype=float)

    element_names = ['floor', 'buttocks']
    k_elem = np.array([floor_k_n_per_m, buttocks_k_n_per_m], dtype=float)
    c_elem = np.array([0.0, buttocks_c_ns_per_m], dtype=float)

    compression_ref_m = np.array([0.01, 0.04], dtype=float)  # unused when k_mult=1
    compression_k_mult = np.array([1.0, 1.0], dtype=float)
    tension_k_mult = np.ones(2, dtype=float)

    compression_only = np.array([True, True], dtype=bool)
    damping_compression_only = np.array([False, True], dtype=bool)
    gap_m = np.array([0.0, 0.0], dtype=float)

    maxwell_k = np.zeros((2, 0), dtype=float)
    maxwell_tau_s = np.zeros((2, 0), dtype=float)
    maxwell_compression_only = np.ones(2, dtype=bool)

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


def simulate_toen_drop(
    *,
    floor_name: str,
    body_mass_kg: float,
    buttocks_k_n_per_m: float,
    buttocks_c_ns_per_m: float,
    floor_k_n_per_m: float,
    skin_mass_kg: float = TOEN_TABLE1_MASSES_50TH_KG['skin'],
    impact_velocity_mps: float = TOEN_IMPACT_V_MPS,
    dt_s: float = 0.0005,
    duration_s: float = 0.15,
    max_newton_iter: int = 8,
) -> tuple[ToenDropResult, SpineModel, object]:
    model = build_toen_drop_model(
        body_mass_kg=body_mass_kg,
        skin_mass_kg=skin_mass_kg,
        buttocks_k_n_per_m=buttocks_k_n_per_m,
        buttocks_c_ns_per_m=buttocks_c_ns_per_m,
        floor_k_n_per_m=floor_k_n_per_m,
    )

    t = np.arange(0.0, duration_s + dt_s, dt_s, dtype=float)
    base_accel_g = np.zeros_like(t)

    # IMPORTANT: at contact onset, skin and body are both moving downward.
    y0 = np.zeros(model.size(), dtype=float)
    v0 = np.zeros(model.size(), dtype=float)
    v0[0] = -float(impact_velocity_mps)  # skin
    v0[1] = -float(impact_velocity_mps)  # body
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

    res = ToenDropResult(
        floor_name=floor_name,
        floor_k_n_per_m=float(floor_k_n_per_m),
        peak_ground_kN=peak_ground_kN,
        t_peak_ms=t_peak_ms,
        peak_buttocks_kN=peak_butt_kN,
        max_buttocks_comp_mm=float(np.max(x_butt) * 1000.0),
        max_floor_comp_mm=float(np.max(x_floor) * 1000.0),
    )
    return res, model, sim


def _pick_targets(target_set: str) -> dict[str, float]:
    key = target_set.strip().lower()
    if key in {'avg', 'paper'}:
        return TOEN_GROUND_PEAKS_KN_AVG_PAPER
    if key in {'avg_measured', 'fig4_measured'}:
        return TOEN_GROUND_PEAKS_KN_AVG_MEASURED_FIG4
    if key in {'subj3', 'subject3', 'fig3_subj3'}:
        return TOEN_GROUND_PEAKS_KN_SUBJECT3_FIG3_APPROX
    raise ValueError('target_set must be one of: avg, avg_measured, subj3')


def run_toen_suite(
    *,
    subject_id: str,
    target_set: str,
    male50_mass_kg: float = DEFAULT_50TH_MALE_TOTAL_MASS_KG,
    impact_velocity_mps: float = TOEN_IMPACT_V_MPS,
) -> list[ToenDropResult]:
    targets = _pick_targets(target_set)
    subj = TOEN_SUBJECTS[subject_id]
    k_butt, c_butt = subject_buttocks_kc(subject_id)

    # Use Toen torso mass (Table1 sum) scaled reminder:
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50_mass_kg)

    print('\n=== TOEN DROP SUITE ===')
    print(
        f'Subject: {subject_id}, subject_total_mass={subj.total_mass_kg:.2f} kg, height={subj.height_cm:.1f} cm'
    )
    print(f'Male50 total mass assumption: {male50_mass_kg:.2f} kg')
    print(
        f'Torso mass (Table1 sum scaled): {torso_mass:.2f} kg (Table1 sum={toen_torso_mass_50th_kg():.2f} kg)'
    )
    print(f'Buttocks: k={k_butt:.1f} N/m, c={c_butt:.1f} Ns/m')
    print(f'Impact velocity: {impact_velocity_mps:.2f} m/s')
    print(f'Targets: {target_set} -> {json.dumps(targets)}')

    results: list[ToenDropResult] = []
    for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
        r, _model, _sim = simulate_toen_drop(
            floor_name=floor_name,
            body_mass_kg=torso_mass,
            buttocks_k_n_per_m=k_butt,
            buttocks_c_ns_per_m=c_butt,
            floor_k_n_per_m=k_floor,
            impact_velocity_mps=impact_velocity_mps,
        )
        tgt = float(targets[floor_name])
        err = (r.peak_ground_kN - tgt) / tgt * 100.0
        print(
            f'  {floor_name}: peak_ground={r.peak_ground_kN:.3f} kN (target={tgt:.3f} kN, {err:+.1f}%), '
            f't_peak={r.t_peak_ms:.1f} ms, butt_comp={r.max_buttocks_comp_mm:.1f} mm, floor_comp={r.max_floor_comp_mm:.1f} mm'
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
    """
    Fast calibration: keep buttocks k,c fixed to Toen subject/mean, and fit ONLY a velocity scale.
    This is meant to compensate for single-mass simplification and posture/mass recruitment.
    """
    targets = _pick_targets(target_set)
    subj = TOEN_SUBJECTS[subject_id]
    k_butt, c_butt = subject_buttocks_kc(subject_id)
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50_mass_kg)

    x0 = np.log(np.array([1.0], dtype=float))
    lb = np.log(np.array([0.6], dtype=float))
    ub = np.log(np.array([1.4], dtype=float))

    eval_counter = {'n': 0}

    def residuals(logx: np.ndarray) -> np.ndarray:
        eval_counter['n'] += 1
        vscale = float(np.exp(logx[0]))

        res = []
        for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
            tgt = float(targets[floor_name])
            r, _m, _sim = simulate_toen_drop(
                floor_name=floor_name,
                body_mass_kg=torso_mass,
                buttocks_k_n_per_m=k_butt,
                buttocks_c_ns_per_m=c_butt,
                floor_k_n_per_m=k_floor,
                impact_velocity_mps=v0_mps * vscale,
                dt_s=0.0005,
                duration_s=0.15,
                max_newton_iter=6,
            )
            res.append((r.peak_ground_kN - tgt) / max(tgt, 1e-6))

        if (eval_counter['n'] % max(debug_every, 1)) == 0:
            rms = float(np.sqrt(np.mean(np.square(res))))
            print(
                f'  DEBUG toen calib eval={eval_counter["n"]}: vscale={vscale:.4f}, rms_rel={rms:.5f}'
            )

        return np.asarray(res, dtype=float)

    out = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=40, verbose=2)

    vscale = float(np.exp(out.x[0]))
    return {
        'subject_id': subject_id,
        'target_set': target_set,
        'male50_mass_kg': float(male50_mass_kg),
        'subject_total_mass_kg': float(subj.total_mass_kg),
        'torso_mass_kg': float(torso_mass),
        'buttocks_k_n_per_m': float(k_butt),
        'buttocks_c_ns_per_m': float(c_butt),
        'impact_velocity_mps': float(v0_mps),
        'velocity_scale': float(vscale),
        'success': bool(out.success),
        'cost': float(out.cost),
        'residual_norm': float(np.linalg.norm(out.fun)),
        'nfev': int(out.nfev),
    }
