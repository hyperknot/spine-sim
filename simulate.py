#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spine_sim.calibration import PeakCalibrationCase
from spine_sim.calibration_store import load_calibration_scales, write_calibration_result
from spine_sim.calibration_targets import CALIBRATION_T12L1_PEAKS_KN
from spine_sim.filters import cfc_filter
from spine_sim.io import parse_csv_series, resample_to_uniform
from spine_sim.model import newmark_nonlinear
from spine_sim.model_paths import get_model_path
from spine_sim.range import find_hit_range

from spine_sim.toen_drop import (
    TOEN_IMPACT_V_MPS,
    calibrate_toen_buttocks_model,
    run_toen_suite,
    simulate_toen_drop_trace,
)
from spine_sim.toen_store import load_toen_drop_calibration, write_toen_drop_calibration
from spine_sim.toen_subjects import TOEN_SUBJECTS, subject_buttocks_kc, toen_torso_mass_scaled_kg
from spine_sim.toen_targets import TOEN_FLOOR_STIFFNESS_N_PER_M
from spine_sim.plotting import plot_toen_buttocks_force_compression


REPO_ROOT = Path(__file__).parent
DEFAULT_MASSES_JSON_PATH = REPO_ROOT / "opensim" / "fullbody.json"
CALIBRATION_ROOT = REPO_ROOT / "calibration"
CALIBRATION_YOGANANDAN_DIR = CALIBRATION_ROOT / "yoganandan"


@dataclass
class ProcessingInfo:
    sample_rate_hz: float
    dt_s: float
    duration_ms: float
    style: str  # "drop" or "flat"
    bias_correction_applied: bool
    bias_correction_g: float


def _read_config() -> dict:
    return json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))


def _resolve_path(p: str) -> Path:
    # allow config to use relative paths
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _detect_style(duration_ms: float, threshold_ms: float) -> str:
    return "flat" if duration_ms < threshold_ms else "drop"


def _freefall_bias_correct(accel_g: np.ndarray, apply_correction: bool) -> tuple[np.ndarray, bool, float]:
    # Same heuristic as before; kept minimal.
    if len(accel_g) < 100:
        samples = accel_g[50:] if len(accel_g) > 50 else accel_g
    else:
        samples = accel_g[50:100]

    if samples.size == 0:
        return accel_g, False, 0.0

    ff_median = float(np.median(samples))
    bias = -1.0 - ff_median

    if apply_correction:
        return accel_g + bias, True, bias
    return accel_g, False, bias


def process_input_file(
    path: Path,
    *,
    cfc: float,
    sim_duration_ms: float,
    style_duration_threshold_ms: float,
    peak_threshold_g: float,
    freefall_threshold_g: float,
    drop_baseline_correction: bool,
) -> tuple[np.ndarray, np.ndarray, ProcessingInfo]:
    series = parse_csv_series(
        path,
        time_candidates=["time", "time0", "t"],
        value_candidates=["accel", "acceleration"],
    )
    series, sample_rate = resample_to_uniform(series)
    dt = 1.0 / sample_rate

    accel_raw = np.asarray(series.values, dtype=float)
    accel_filtered = np.asarray(cfc_filter(accel_raw.tolist(), sample_rate, cfc), dtype=float)
    t_all = np.asarray(series.time_s, dtype=float)

    total_duration_ms = float((t_all[-1] - t_all[0]) * 1000.0) if t_all.size >= 2 else 0.0
    style = _detect_style(total_duration_ms, style_duration_threshold_ms)

    # Segment selection:
    if style == "flat":
        t_seg = t_all - t_all[0]
        a_seg = accel_filtered.copy()

        desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
        if len(t_seg) < desired_n:
            pad_n = desired_n - len(t_seg)
            t_pad = t_seg[-1] + dt * (np.arange(pad_n) + 1)
            a_pad = np.zeros(pad_n, dtype=float)
            t_seg = np.concatenate([t_seg, t_pad])
            a_seg = np.concatenate([a_seg, a_pad])

        info = ProcessingInfo(
            sample_rate_hz=float(sample_rate),
            dt_s=float(dt),
            duration_ms=float(t_seg[-1] * 1000.0),
            style="flat",
            bias_correction_applied=False,
            bias_correction_g=0.0,
        )
        return t_seg, a_seg, info

    hit = find_hit_range(
        accel_filtered.tolist(),
        peak_threshold_g=peak_threshold_g,
        freefall_threshold_g=freefall_threshold_g,
    )

    if hit:
        start_idx = hit.start_idx
    else:
        start_idx = 0

    end_idx = min(len(t_all) - 1, start_idx + int(round((sim_duration_ms / 1000.0) / dt)))
    t_seg = t_all[start_idx : end_idx + 1] - t_all[start_idx]
    a_seg = accel_filtered[start_idx : end_idx + 1]

    a_seg, applied, bias = _freefall_bias_correct(a_seg, apply_correction=drop_baseline_correction)

    desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
    if len(t_seg) < desired_n:
        pad_n = desired_n - len(t_seg)
        t_pad = t_seg[-1] + dt * (np.arange(pad_n) + 1)
        a_pad = -1.0 * np.ones(pad_n, dtype=float)
        t_seg = np.concatenate([t_seg, t_pad])
        a_seg = np.concatenate([a_seg, a_pad])

    info = ProcessingInfo(
        sample_rate_hz=float(sample_rate),
        dt_s=float(dt),
        duration_ms=float(t_seg[-1] * 1000.0),
        style="drop",
        bias_correction_applied=applied,
        bias_correction_g=float(bias),
    )
    return t_seg, a_seg, info


def load_masses_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_mass_map(masses: dict, arm_recruitment: float, helmet_mass: float) -> dict:
    b = masses["bodies"]

    arm_mass = (
        b["humerus_R"]
        + b["humerus_L"]
        + b["ulna_R"]
        + b["ulna_L"]
        + b["radius_R"]
        + b["radius_L"]
        + b["hand_R"]
        + b["hand_L"]
    ) * arm_recruitment

    head_total = b["head_neck"] + helmet_mass + arm_mass

    return {
        "pelvis": b["pelvis"],
        "l5": b["lumbar5"],
        "l4": b["lumbar4"],
        "l3": b["lumbar3"],
        "l2": b["lumbar2"],
        "l1": b["lumbar1"],
        "t12": b["thoracic12"],
        "t11": b["thoracic11"],
        "t10": b["thoracic10"],
        "t9": b["thoracic9"],
        "t8": b["thoracic8"],
        "t7": b["thoracic7"],
        "t6": b["thoracic6"],
        "t5": b["thoracic5"],
        "t4": b["thoracic4"],
        "t3": b["thoracic3"],
        "t2": b["thoracic2"],
        "t1": b["thoracic1"],
        "head": head_total,
    }


def simulate_drop() -> None:
    config = _read_config()
    drop_cfg = config["drop"]

    model_type = str(config["model"].get("type", "zwt")).lower()
    model_path = get_model_path(model_type)

    masses_json_path = _resolve_path(str(config["model"].get("masses_json", str(DEFAULT_MASSES_JSON_PATH))))
    masses_data = load_masses_json(masses_json_path)
    mass_map = build_mass_map(
        masses_data,
        arm_recruitment=float(config["model"].get("arm_recruitment", 0.5)),
        helmet_mass=float(config["model"].get("helmet_mass_kg", 0.7)),
    )

    model = model_path.build_model(mass_map, config)

    calib_mode = str(drop_cfg.get("calibration_mode", "peaks")).lower()
    scales = load_calibration_scales(model_type, calib_mode, model_path.default_scales)
    model = model_path.apply_calibration(model, scales)

    inputs_dir = _resolve_path(str(drop_cfg.get("inputs_dir", "drops")))
    pattern = str(drop_cfg.get("pattern", "*.csv"))
    out_dir = _resolve_path(str(drop_cfg.get("output_dir", "output/drop")))
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(inputs_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No drop inputs found: {inputs_dir}/{pattern}")

    print(f"DROP simulate: model={model_type}, calibration_mode={calib_mode}, files={len(files)}")
    print(f"DROP outputs: {out_dir}")

    summary: list[dict] = []

    for fpath in files:
        print(f"\nDROP simulate: {fpath.name}")

        t, a_g, info = process_input_file(
            fpath,
            cfc=float(drop_cfg.get("cfc", 75)),
            sim_duration_ms=float(drop_cfg.get("sim_duration_ms", 200.0)),
            style_duration_threshold_ms=float(drop_cfg.get("style_duration_threshold_ms", 300.0)),
            peak_threshold_g=float(drop_cfg.get("peak_threshold_g", 5.0)),
            freefall_threshold_g=float(drop_cfg.get("freefall_threshold_g", -0.85)),
            drop_baseline_correction=bool(drop_cfg.get("drop_baseline_correction", True)),
        )

        # Optional gravity settling for "flat" style inputs
        y0 = np.zeros(model.size(), dtype=float)
        v0 = np.zeros(model.size(), dtype=float)
        s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

        if info.style == "flat":
            settle_ms = float(drop_cfg.get("gravity_settle_ms", 150.0))
            if settle_ms > 0.0:
                dt = info.dt_s
                n_settle = int(round((settle_ms / 1000.0) / dt)) + 1
                t_settle = dt * np.arange(n_settle)
                a_settle = np.zeros_like(t_settle)
                sim_settle = newmark_nonlinear(model, t_settle, a_settle, y0, v0, s0)
                y0 = sim_settle.y[-1].copy()
                v0 = sim_settle.v[-1].copy()
                s0 = sim_settle.maxwell_state_n[-1].copy()

        sim = newmark_nonlinear(model, t, a_g, y0, v0, s0)
        forces = sim.element_forces_n

        # Save timeseries (CSV per run)
        run_dir = out_dir / fpath.stem
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        _write_timeseries_csv(
            run_dir / "timeseries.csv",
            sim.time_s,
            sim.base_accel_g,
            model.node_names,
            model.element_names,
            sim.y,
            sim.v,
            sim.a,
            forces,
        )

        t12_idx = model.element_names.index("T12-L1")
        butt_idx = model.element_names.index("buttocks")

        peak_t12 = float(np.max(forces[:, t12_idx]) / 1000.0)
        peak_butt = float(np.max(forces[:, butt_idx]) / 1000.0)

        print(
            f"  style={info.style}, dt={info.dt_s*1000.0:.3f} ms, "
            f"peak_T12L1={peak_t12:.3f} kN, peak_butt={peak_butt:.3f} kN"
        )

        summary.append(
            {
                "file": fpath.name,
                "style": info.style,
                "dt_s": info.dt_s,
                "sample_rate_hz": info.sample_rate_hz,
                "bias_correction_applied": info.bias_correction_applied,
                "bias_correction_g": info.bias_correction_g,
                "peak_T12L1_kN": peak_t12,
                "peak_buttocks_kN": peak_butt,
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nDROP simulate complete.")


def calibrate_drop() -> None:
    config = _read_config()
    drop_cfg = config["drop"]

    model_type = str(config["model"].get("type", "zwt")).lower()
    model_path = get_model_path(model_type)

    masses_json_path = _resolve_path(str(config["model"].get("masses_json", str(DEFAULT_MASSES_JSON_PATH))))
    masses_data = load_masses_json(masses_json_path)
    mass_map = build_mass_map(
        masses_data,
        arm_recruitment=float(config["model"].get("arm_recruitment", 0.5)),
        helmet_mass=float(config["model"].get("helmet_mass_kg", 0.7)),
    )
    base_model = model_path.build_model(mass_map, config)

    # Only "peaks" calibration kept (simplified)
    cases: list[PeakCalibrationCase] = []
    for name in ["50ms", "75ms", "100ms", "150ms", "200ms"]:
        accel_path = CALIBRATION_YOGANANDAN_DIR / f"accel_{name}.csv"
        if not accel_path.exists():
            raise FileNotFoundError(f"Missing calibration input: {accel_path}")

        t, a_g, info = process_input_file(
            accel_path,
            cfc=float(drop_cfg.get("cfc", 75)),
            sim_duration_ms=float(drop_cfg.get("sim_duration_ms", 200.0)),
            style_duration_threshold_ms=float(drop_cfg.get("style_duration_threshold_ms", 300.0)),
            peak_threshold_g=float(drop_cfg.get("peak_threshold_g", 5.0)),
            freefall_threshold_g=float(drop_cfg.get("freefall_threshold_g", -0.85)),
            drop_baseline_correction=bool(drop_cfg.get("drop_baseline_correction", True)),
        )

        cases.append(
            PeakCalibrationCase(
                name=name,
                time_s=np.asarray(t, dtype=float),
                accel_g=np.asarray(a_g, dtype=float),
                target_peak_force_n=float(CALIBRATION_T12L1_PEAKS_KN[name]) * 1000.0,
                settle_ms=(float(drop_cfg.get("gravity_settle_ms", 150.0)) if info.style == "flat" else 0.0),
            )
        )

    init_scales = load_calibration_scales(model_type, "peaks", model_path.default_scales)
    t12_idx = base_model.element_names.index("T12-L1")

    print(f"DROP calibrate: model={model_type}, cases={len(cases)}")
    result = model_path.calibrate_peaks(
        base_model,
        cases,
        t12_element_index=t12_idx,
        init_scales=init_scales,
        calibrate_damping=False,
    )

    write_calibration_result(
        model_type=model_type,
        mode="peaks",
        result=result,
        cases=[{"name": c.name, "target_peak_kN": c.target_peak_force_n / 1000.0} for c in cases],
        default_scales=model_path.default_scales,
    )

    print("DROP calibration complete.")
    print(json.dumps(result.scales, indent=2))


def simulate_buttock() -> None:
    config = _read_config()
    bcfg = config["buttock"]

    subject = str(bcfg.get("subject", "avg")).lower()
    target_set = str(bcfg.get("target_set", "avg")).lower()
    if target_set == "fig3":
        target_set = "subj3"

    velocities = [float(v) for v in bcfg.get("velocities_mps", [TOEN_IMPACT_V_MPS])]
    male50 = float(bcfg.get("male50_mass_kg", 75.4))

    solver = bcfg.get("solver", {})
    dt_s = float(solver.get("dt_s", 0.0005))
    duration_s = float(solver.get("duration_s", 0.15))
    max_newton_iter = int(solver.get("max_newton_iter", 10))

    dens = bcfg.get("densification", {})
    butt = dens.get("buttocks", {})

    butt_limit_mm = float(butt.get("limit_mm", 39.0))
    butt_stop_k = float(butt.get("stop_k_n_per_m", 5.0e6))
    butt_smooth_mm = float(butt.get("smoothing_mm", 1.0))

    butt_k_override = None
    butt_c_override = None
    butt_limit_override = None
    butt_stop_k_override = None
    butt_smooth_override = None

    if bool(bcfg.get("use_saved_calibration", True)):
        doc, path = load_toen_drop_calibration()
        if doc is not None:
            r = doc.get("result", {})
            saved_target = str(r.get("target_set", "")).lower()
            if saved_target == target_set:
                butt_k_override = r.get("buttocks_k_n_per_m", None)
                butt_c_override = r.get("buttocks_c_ns_per_m", None)
                butt_limit_override = r.get("buttocks_limit_mm", None)
                butt_stop_k_override = r.get("buttocks_stop_k_n_per_m", None)
                butt_smooth_override = r.get("buttocks_stop_smoothing_mm", None)
                print(f"BUTTOCK simulate: loaded calibration from {path}")
            else:
                print(
                    f"BUTTOCK simulate: saved calibration target_set mismatch "
                    f"(saved={saved_target!r}, requested={target_set!r}); ignoring {path}"
                )

    # Final params (calibration overrides config)
    if butt_k_override is not None:
        butt_k_override = float(butt_k_override)
    if butt_c_override is not None:
        butt_c_override = float(butt_c_override)
    if butt_limit_override is not None:
        butt_limit_mm = float(butt_limit_override)
    if butt_stop_k_override is not None:
        butt_stop_k = float(butt_stop_k_override)
    if butt_smooth_override is not None:
        butt_smooth_mm = float(butt_smooth_override)

    subjects = ["avg", "3"] if subject == "both" else [subject]

    out_dir = _resolve_path(str(bcfg.get("output_dir", "output/toen_drop")))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    print(f"BUTTOCK simulate: target_set={target_set}, subject={subject}, velocities={velocities}")
    print(
        f"BUTTOCK densification: limit_mm={butt_limit_mm}, stop_k={butt_stop_k:.3g}, smoothing_mm={butt_smooth_mm}"
    )

    for sid in subjects:
        results = run_toen_suite(
            subject_id=sid,
            target_set=target_set,
            male50_mass_kg=male50,
            impact_velocities_mps=velocities,
            velocity_scale=1.0,
            dt_s=dt_s,
            duration_s=duration_s,
            max_newton_iter=max_newton_iter,
            buttocks_k_n_per_m=butt_k_override,
            buttocks_c_ns_per_m=butt_c_override,
            buttocks_limit_mm=butt_limit_mm,
            buttocks_stop_k_n_per_m=butt_stop_k,
            buttocks_stop_smoothing_mm=butt_smooth_mm,
        )
        all_results += [r.__dict__ for r in results]

    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(f"BUTTOCK simulate complete. Wrote {out_dir / 'summary.json'}")


def calibrate_buttock() -> None:
    config = _read_config()
    bcfg = config["buttock"]

    subject = str(bcfg.get("subject", "avg")).lower()
    if subject == "both":
        subject = "avg"

    target_set = str(bcfg.get("target_set", "avg")).lower()
    if target_set == "fig3":
        target_set = "subj3"

    male50 = float(bcfg.get("male50_mass_kg", 75.4))

    dens = bcfg.get("densification", {})
    butt = dens.get("buttocks", {})
    butt_stop_k = float(butt.get("stop_k_n_per_m", 5.0e6))
    butt_smooth_mm = float(butt.get("smoothing_mm", 1.0))

    print(f"BUTTOCK calibrate (Toen-based): subject={subject}, target_set={target_set}")
    result = calibrate_toen_buttocks_model(
        subject_id=subject,
        target_set=target_set,
        male50_mass_kg=male50,
        buttocks_stop_k_n_per_m=butt_stop_k,
        buttocks_stop_smoothing_mm=butt_smooth_mm,
    )
    out_path = write_toen_drop_calibration(result, active=True)
    print(f"BUTTOCK calibration saved: {out_path}")
    print(json.dumps(result, indent=2))


def plot_toen_buttocks() -> None:
    """
    Produce a buttocks-only plot (force + compression vs time) for a Toen-style run.

    - Uses the saved Toen calibration if present and matching target_set.
    - Overlays all floors (soft/medium/firm/rigid) in one figure.
    """
    config = _read_config()
    bcfg = config["buttock"]

    subject = str(bcfg.get("subject", "avg")).lower()
    if subject == "both":
        subject = "avg"

    target_set = str(bcfg.get("target_set", "avg")).lower()
    if target_set == "fig3":
        target_set = "subj3"

    male50 = float(bcfg.get("male50_mass_kg", 75.4))

    solver = bcfg.get("solver", {})
    dt_s = float(solver.get("dt_s", 0.0005))
    duration_s = float(solver.get("duration_s", 0.15))
    max_newton_iter = int(solver.get("max_newton_iter", 10))

    # Plot velocity: default to canonical Toen 3.5 m/s.
    v_plot = float(bcfg.get("plot_velocity_mps", TOEN_IMPACT_V_MPS))

    # Densification defaults; may be overridden by saved calibration.
    dens = bcfg.get("densification", {})
    butt = dens.get("buttocks", {})
    butt_limit_mm = float(butt.get("limit_mm", 39.0))
    butt_stop_k = float(butt.get("stop_k_n_per_m", 5.0e6))
    butt_smooth_mm = float(butt.get("smoothing_mm", 1.0))

    butt_k_override = None
    butt_c_override = None
    butt_limit_override = None
    butt_stop_k_override = None
    butt_smooth_override = None

    if bool(bcfg.get("use_saved_calibration", True)):
        doc, path = load_toen_drop_calibration()
        if doc is not None:
            r = doc.get("result", {})
            saved_target = str(r.get("target_set", "")).lower()
            if saved_target == target_set:
                butt_k_override = r.get("buttocks_k_n_per_m", None)
                butt_c_override = r.get("buttocks_c_ns_per_m", None)
                butt_limit_override = r.get("buttocks_limit_mm", None)
                butt_stop_k_override = r.get("buttocks_stop_k_n_per_m", None)
                butt_smooth_override = r.get("buttocks_stop_smoothing_mm", None)
                print(f"BUTTOCK plot: loaded calibration from {path}")
            else:
                print(
                    f"BUTTOCK plot: saved calibration target_set mismatch "
                    f"(saved={saved_target!r}, requested={target_set!r}); ignoring {path}"
                )

    if butt_limit_override is not None:
        butt_limit_mm = float(butt_limit_override)
    if butt_stop_k_override is not None:
        butt_stop_k = float(butt_stop_k_override)
    if butt_smooth_override is not None:
        butt_smooth_mm = float(butt_smooth_override)

    subj = TOEN_SUBJECTS[subject]
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50)

    k_subj, c_subj = subject_buttocks_kc(subject)
    k_butt = float(k_subj if butt_k_override is None else butt_k_override)
    c_butt = float(c_subj if butt_c_override is None else butt_c_override)

    out_dir = _resolve_path(str(bcfg.get("output_dir", "output/toen_drop")))
    out_dir.mkdir(parents=True, exist_ok=True)

    compression_by_floor_mm: dict[str, np.ndarray] = {}
    force_by_floor_kN: dict[str, np.ndarray] = {}
    time_s: np.ndarray | None = None

    for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
        _res, trace = simulate_toen_drop_trace(
            floor_name=floor_name,
            body_mass_kg=torso_mass,
            buttocks_k_n_per_m=k_butt,
            buttocks_c_ns_per_m=c_butt,
            floor_k_n_per_m=float(k_floor),
            impact_velocity_mps=v_plot,
            buttocks_limit_mm=butt_limit_mm,
            buttocks_stop_k_n_per_m=butt_stop_k,
            buttocks_stop_smoothing_mm=butt_smooth_mm,
            dt_s=dt_s,
            duration_s=duration_s,
            max_newton_iter=max_newton_iter,
        )

        if time_s is None:
            time_s = trace.time_s

        compression_by_floor_mm[floor_name] = trace.buttocks_compression_m * 1000.0
        force_by_floor_kN[floor_name] = trace.buttocks_force_n / 1000.0

    if time_s is None:
        raise SystemExit("No Toen trace data generated (unexpected).")

    out_path = out_dir / f"buttocks_force_compression_v{v_plot:.2f}.png"
    plot_toen_buttocks_force_compression(
        time_s,
        compression_by_floor_mm=compression_by_floor_mm,
        force_by_floor_kN=force_by_floor_kN,
        out_path=out_path,
        title=f"Toen buttocks response (subject={subject}, target_set={target_set}, v={v_plot:.2f} m/s)",
    )
    print(f"BUTTOCK plot complete. Wrote {out_path}")


def _write_timeseries_csv(
    out_path: Path,
    time_s: np.ndarray,
    base_accel_g: np.ndarray,
    node_names: list[str],
    elem_names: list[str],
    y: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    forces_n: np.ndarray,
) -> None:
    import csv

    headers = ["time_s", "base_accel_g"]
    headers += [f"y_{n}_mm" for n in node_names]
    headers += [f"v_{n}_mps" for n in node_names]
    headers += [f"a_{n}_mps2" for n in node_names]
    headers += [f"F_{e}_kN" for e in elem_names]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(time_s.size):
            row = [f"{time_s[i]:.6f}", f"{base_accel_g[i]:.6f}"]
            row += [f"{(y[i, j] * 1000.0):.6f}" for j in range(y.shape[1])]
            row += [f"{v[i, j]:.6f}" for j in range(v.shape[1])]
            row += [f"{a[i, j]:.6f}" for j in range(a.shape[1])]
            row += [f"{(forces_n[i, j] / 1000.0):.6f}" for j in range(forces_n.shape[1])]
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "simulate-drop",
            "calibrate-drop",
            "simulate-buttocks",
            "calibrate-buttocks",
            "plot-toen-buttocks",
        ],
    )
    args = parser.parse_args()

    if args.mode == "simulate-drop":
        simulate_drop()
        return
    if args.mode == "calibrate-drop":
        calibrate_drop()
        return
    if args.mode in {"simulate-buttocks"}:
        simulate_buttock()
        return
    if args.mode in {"calibrate-buttocks"}:
        calibrate_buttock()
        return
    if args.mode in {"plot-toen-buttocks"}:
        plot_toen_buttocks()
        return

    raise SystemExit("Unknown mode.")


if __name__ == "__main__":
    main()
