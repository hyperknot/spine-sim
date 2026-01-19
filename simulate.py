#!/usr/bin/env -S uv run

"""
Spine simulation CLI.

Pipeline:
1. calibrate-buttocks: Calibrate buttocks model from Toen 2012 paper data
2. simulate-buttocks: Simulate Toen drop suite and plot results
3. calibrate-drop: Calibrate spine model against Yoganandan data (uses Toen buttocks)
4. simulate-drop: Run drop simulations on acceleration data
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import click
import numpy as np

from spine_sim.calibration import PeakCalibrationCase
from spine_sim.calibration_store import load_calibration_scales, write_calibration_result
from spine_sim.calibration_targets import CALIBRATION_T12L1_PEAKS_KN
from spine_sim.filters import cfc_filter
from spine_sim.io import parse_csv_series, resample_to_uniform
from spine_sim.model import SpineModel, newmark_nonlinear
from spine_sim.model_paths import get_model_path
from spine_sim.plotting import plot_toen_buttocks_force_compression
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

REPO_ROOT = Path(__file__).parent
DEFAULT_MASSES_JSON = REPO_ROOT / "opensim" / "fullbody.json"
CALIBRATION_ROOT = REPO_ROOT / "calibration"
CALIBRATION_YOGANANDAN_DIR = CALIBRATION_ROOT / "yoganandan"


def _read_config() -> dict:
    return json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _load_masses(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_mass_map(masses: dict, arm_recruitment: float, helmet_mass: float) -> dict:
    b = masses["bodies"]
    arm_mass = (
        b["humerus_R"] + b["humerus_L"] + b["ulna_R"] + b["ulna_L"] +
        b["radius_R"] + b["radius_L"] + b["hand_R"] + b["hand_L"]
    ) * arm_recruitment
    return {
        "pelvis": b["pelvis"], "l5": b["lumbar5"], "l4": b["lumbar4"],
        "l3": b["lumbar3"], "l2": b["lumbar2"], "l1": b["lumbar1"],
        "t12": b["thoracic12"], "t11": b["thoracic11"], "t10": b["thoracic10"],
        "t9": b["thoracic9"], "t8": b["thoracic8"], "t7": b["thoracic7"],
        "t6": b["thoracic6"], "t5": b["thoracic5"], "t4": b["thoracic4"],
        "t3": b["thoracic3"], "t2": b["thoracic2"], "t1": b["thoracic1"],
        "head": b["head_neck"] + helmet_mass + arm_mass,
    }


def _detect_style(duration_ms: float, threshold_ms: float) -> str:
    return "flat" if duration_ms < threshold_ms else "drop"


def _freefall_bias_correct(accel_g: np.ndarray, apply: bool) -> tuple[np.ndarray, bool, float]:
    samples = accel_g[50:100] if len(accel_g) >= 100 else accel_g[50:] if len(accel_g) > 50 else accel_g
    if samples.size == 0:
        return accel_g, False, 0.0
    bias = -1.0 - float(np.median(samples))
    return (accel_g + bias, True, bias) if apply else (accel_g, False, bias)


def _process_input(
    path: Path, cfc: float, sim_duration_ms: float, style_threshold_ms: float,
    peak_threshold_g: float, freefall_threshold_g: float, drop_baseline_correction: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    series = parse_csv_series(path, ["time", "time0", "t"], ["accel", "acceleration"])
    series, sample_rate = resample_to_uniform(series)
    dt = 1.0 / sample_rate

    accel_raw = np.asarray(series.values, dtype=float)
    accel_filtered = np.asarray(cfc_filter(accel_raw.tolist(), sample_rate, cfc), dtype=float)
    t_all = np.asarray(series.time_s, dtype=float)

    total_ms = float((t_all[-1] - t_all[0]) * 1000.0) if t_all.size >= 2 else 0.0
    style = _detect_style(total_ms, style_threshold_ms)

    if style == "flat":
        t_seg, a_seg = t_all - t_all[0], accel_filtered.copy()
        desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
        if len(t_seg) < desired_n:
            pad_n = desired_n - len(t_seg)
            t_seg = np.concatenate([t_seg, t_seg[-1] + dt * (np.arange(pad_n) + 1)])
            a_seg = np.concatenate([a_seg, np.zeros(pad_n)])
        return t_seg, a_seg, {"style": "flat", "dt_s": dt, "sample_rate_hz": sample_rate,
                              "bias_applied": False, "bias_g": 0.0}

    hit = find_hit_range(accel_filtered.tolist(), peak_threshold_g, freefall_threshold_g)
    start_idx = hit.start_idx if hit else 0
    end_idx = min(len(t_all) - 1, start_idx + int(round((sim_duration_ms / 1000.0) / dt)))

    t_seg = t_all[start_idx:end_idx + 1] - t_all[start_idx]
    a_seg = accel_filtered[start_idx:end_idx + 1]
    a_seg, applied, bias = _freefall_bias_correct(a_seg, drop_baseline_correction)

    desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
    if len(t_seg) < desired_n:
        pad_n = desired_n - len(t_seg)
        t_seg = np.concatenate([t_seg, t_seg[-1] + dt * (np.arange(pad_n) + 1)])
        a_seg = np.concatenate([a_seg, -1.0 * np.ones(pad_n)])

    return t_seg, a_seg, {"style": "drop", "dt_s": dt, "sample_rate_hz": sample_rate,
                          "bias_applied": applied, "bias_g": bias}


def _write_timeseries_csv(
    path: Path, time_s: np.ndarray, base_accel_g: np.ndarray,
    node_names: list[str], elem_names: list[str],
    y: np.ndarray, v: np.ndarray, a: np.ndarray, forces_n: np.ndarray,
) -> None:
    headers = ["time_s", "base_accel_g"]
    headers += [f"y_{n}_mm" for n in node_names]
    headers += [f"v_{n}_mps" for n in node_names]
    headers += [f"a_{n}_mps2" for n in node_names]
    headers += [f"F_{e}_kN" for e in elem_names]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(time_s.size):
            row = [f"{time_s[i]:.6f}", f"{base_accel_g[i]:.6f}"]
            row += [f"{y[i, j] * 1000.0:.6f}" for j in range(y.shape[1])]
            row += [f"{v[i, j]:.6f}" for j in range(v.shape[1])]
            row += [f"{a[i, j]:.6f}" for j in range(a.shape[1])]
            row += [f"{forces_n[i, j] / 1000.0:.6f}" for j in range(forces_n.shape[1])]
            w.writerow(row)


def _get_toen_buttocks_params(config: dict) -> dict | None:
    """Load Toen calibration if available and matching target_set."""
    bcfg = config.get("buttock", {})
    target_set = str(bcfg.get("target_set", "avg")).lower()
    if target_set == "fig3":
        target_set = "subj3"

    if not bcfg.get("use_saved_calibration", True):
        return None

    doc, path = load_toen_drop_calibration()
    if doc is None:
        return None

    r = doc.get("result", {})
    if str(r.get("target_set", "")).lower() != target_set:
        return None

    return {
        "k": r.get("buttocks_k_n_per_m"),
        "c": r.get("buttocks_c_ns_per_m"),
        "limit_mm": r.get("buttocks_limit_mm"),
        "stop_k": r.get("buttocks_stop_k_n_per_m"),
        "smoothing_mm": r.get("buttocks_stop_smoothing_mm"),
    }


def _apply_toen_buttocks_to_model(model: SpineModel, toen_params: dict) -> SpineModel:
    """Apply Toen-calibrated buttocks parameters to spine model."""
    if toen_params is None:
        return model

    k_elem = model.k_elem.copy()
    c_elem = model.c_elem.copy()

    if toen_params.get("k") is not None:
        k_elem[0] = float(toen_params["k"])
    if toen_params.get("c") is not None:
        c_elem[0] = float(toen_params["c"])

    limit_m = model.compression_limit_m
    stop_k = model.compression_stop_k
    smoothing_m = model.compression_stop_smoothing_m

    if toen_params.get("limit_mm") is not None:
        if limit_m is None:
            limit_m = np.zeros(model.n_elems(), dtype=float)
        else:
            limit_m = limit_m.copy()
        limit_m[0] = float(toen_params["limit_mm"]) / 1000.0

    if toen_params.get("stop_k") is not None:
        if stop_k is None:
            stop_k = np.zeros(model.n_elems(), dtype=float)
        else:
            stop_k = stop_k.copy()
        stop_k[0] = float(toen_params["stop_k"])

    if toen_params.get("smoothing_mm") is not None:
        if smoothing_m is None:
            smoothing_m = np.zeros(model.n_elems(), dtype=float)
        else:
            smoothing_m = smoothing_m.copy()
        smoothing_m[0] = float(toen_params["smoothing_mm"]) / 1000.0

    return SpineModel(
        node_names=model.node_names,
        masses_kg=model.masses_kg,
        element_names=model.element_names,
        k_elem=k_elem,
        c_elem=c_elem,
        compression_ref_m=model.compression_ref_m,
        compression_k_mult=model.compression_k_mult,
        tension_k_mult=model.tension_k_mult,
        compression_only=model.compression_only,
        damping_compression_only=model.damping_compression_only,
        gap_m=model.gap_m,
        maxwell_k=model.maxwell_k,
        maxwell_tau_s=model.maxwell_tau_s,
        maxwell_compression_only=model.maxwell_compression_only,
        poly_k2=model.poly_k2,
        poly_k3=model.poly_k3,
        compression_limit_m=limit_m,
        compression_stop_k=stop_k,
        compression_stop_smoothing_m=smoothing_m,
    )


def _compute_free_buttocks_height_mm(toen_params: dict | None) -> float:
    """Compute the free (uncompressed) buttocks height from Toen parameters."""
    # The buttocks "height" for plotting is the uncompressed rest length.
    # From Toen model: limit_mm represents max compression at rigid floor impact.
    # A reasonable rest length is ~1.5x the limit (tissue is compressible to ~60% strain).
    # Default to 100mm if no calibration available.
    if toen_params is None or toen_params.get("limit_mm") is None:
        return 100.0
    limit_mm = float(toen_params["limit_mm"])
    # Buttocks tissue compresses ~60% at extreme impact, so rest length ≈ limit / 0.6
    return limit_mm / 0.6


@click.group()
def cli():
    """Spine-sim: 1D axial spine impact simulation."""
    pass


@cli.command("calibrate-buttocks")
def calibrate_buttocks():
    """Calibrate buttocks model from Toen 2012 paper data."""
    config = _read_config()
    bcfg = config.get("buttock", {})

    subject = str(bcfg.get("subject", "avg")).lower()
    if subject == "both":
        subject = "avg"

    target_set = str(bcfg.get("target_set", "avg")).lower()
    if target_set == "fig3":
        target_set = "subj3"

    dens = bcfg.get("densification", {}).get("buttocks", {})

    click.echo(f"Calibrating buttocks model: subject={subject}, target_set={target_set}")
    result = calibrate_toen_buttocks_model(
        subject_id=subject,
        target_set=target_set,
        male50_mass_kg=float(bcfg.get("male50_mass_kg", 75.4)),
        buttocks_stop_k_n_per_m=float(dens.get("stop_k_n_per_m", 5.0e6)),
        buttocks_stop_smoothing_mm=float(dens.get("smoothing_mm", 1.0)),
    )

    out_path = write_toen_drop_calibration(result, active=True)
    click.echo(f"Calibration saved: {out_path}")
    click.echo(json.dumps(result, indent=2))


@cli.command("simulate-buttocks")
def simulate_buttocks():
    """Simulate Toen drop suite and generate plots."""
    config = _read_config()
    bcfg = config.get("buttock", {})

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

    dens = bcfg.get("densification", {}).get("buttocks", {})
    butt_limit_mm = float(dens.get("limit_mm", 39.0))
    butt_stop_k = float(dens.get("stop_k_n_per_m", 5.0e6))
    butt_smooth_mm = float(dens.get("smoothing_mm", 1.0))

    # Load saved calibration if available
    butt_k, butt_c = None, None
    if bcfg.get("use_saved_calibration", True):
        doc, path = load_toen_drop_calibration()
        if doc is not None:
            r = doc.get("result", {})
            saved_target = str(r.get("target_set", "")).lower()
            if saved_target == target_set:
                butt_k = r.get("buttocks_k_n_per_m")
                butt_c = r.get("buttocks_c_ns_per_m")
                butt_limit_mm = float(r.get("buttocks_limit_mm", butt_limit_mm))
                butt_stop_k = float(r.get("buttocks_stop_k_n_per_m", butt_stop_k))
                butt_smooth_mm = float(r.get("buttocks_stop_smoothing_mm", butt_smooth_mm))
                click.echo(f"Loaded calibration from {path}")

    subjects = ["avg", "3"] if subject == "both" else [subject]
    out_dir = _resolve_path(str(bcfg.get("output_dir", "output/toen_drop")))
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Simulating: target_set={target_set}, subjects={subjects}, velocities={velocities}")

    all_results = []
    for sid in subjects:
        results = run_toen_suite(
            subject_id=sid, target_set=target_set, male50_mass_kg=male50,
            impact_velocities_mps=velocities, dt_s=dt_s, duration_s=duration_s,
            max_newton_iter=max_newton_iter, buttocks_k_n_per_m=butt_k,
            buttocks_c_ns_per_m=butt_c, buttocks_limit_mm=butt_limit_mm,
            buttocks_stop_k_n_per_m=butt_stop_k, buttocks_stop_smoothing_mm=butt_smooth_mm,
        )
        all_results.extend([r.__dict__ for r in results])

    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")

    # Generate plots for each velocity
    for v_plot in velocities:
        _generate_buttocks_plot(
            config, out_dir, v_plot, subject if subject != "both" else "avg",
            target_set, male50, butt_k, butt_c, butt_limit_mm, butt_stop_k, butt_smooth_mm,
            dt_s, duration_s, max_newton_iter,
        )

    click.echo(f"Simulation and plots complete. Output: {out_dir}")


def _generate_buttocks_plot(
    config: dict, out_dir: Path, v_plot: float, subject: str, target_set: str,
    male50: float, butt_k: float | None, butt_c: float | None, butt_limit_mm: float,
    butt_stop_k: float, butt_smooth_mm: float, dt_s: float, duration_s: float,
    max_newton_iter: int,
) -> None:
    """Generate buttocks force/compression plot for a velocity."""
    subj = TOEN_SUBJECTS[subject]
    torso_mass = toen_torso_mass_scaled_kg(subj.total_mass_kg, male50_mass_kg=male50)

    k_subj, c_subj = subject_buttocks_kc(subject)
    k_butt = float(k_subj if butt_k is None else butt_k)
    c_butt = float(c_subj if butt_c is None else butt_c)

    compression_by_floor: dict[str, np.ndarray] = {}
    force_by_floor: dict[str, np.ndarray] = {}
    time_s = None

    for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
        _res, trace = simulate_toen_drop_trace(
            floor_name=floor_name, body_mass_kg=torso_mass,
            buttocks_k_n_per_m=k_butt, buttocks_c_ns_per_m=c_butt,
            floor_k_n_per_m=float(k_floor), impact_velocity_mps=v_plot,
            buttocks_limit_mm=butt_limit_mm, buttocks_stop_k_n_per_m=butt_stop_k,
            buttocks_stop_smoothing_mm=butt_smooth_mm,
            dt_s=dt_s, duration_s=duration_s, max_newton_iter=max_newton_iter,
        )
        if time_s is None:
            time_s = trace.time_s
        compression_by_floor[floor_name] = trace.buttocks_compression_m * 1000.0
        force_by_floor[floor_name] = trace.buttocks_force_n / 1000.0

    out_path = out_dir / f"buttocks_force_compression_v{v_plot:.2f}.png"
    plot_toen_buttocks_force_compression(
        time_s, compression_by_floor_mm=compression_by_floor,
        force_by_floor_kN=force_by_floor, out_path=out_path,
        title=f"Toen buttocks response (subject={subject}, target_set={target_set}, v={v_plot:.2f} m/s)",
    )
    click.echo(f"  Plot: {out_path}")


@cli.command("calibrate-drop")
def calibrate_drop():
    """Calibrate spine model against Yoganandan data (uses Toen buttocks)."""
    config = _read_config()
    drop_cfg = config.get("drop", {})

    model_type = str(config.get("model", {}).get("type", "zwt")).lower()
    model_path = get_model_path(model_type)

    masses_path = _resolve_path(str(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON))))
    masses = _load_masses(masses_path)
    mass_map = _build_mass_map(
        masses,
        arm_recruitment=float(config.get("model", {}).get("arm_recruitment", 0.5)),
        helmet_mass=float(config.get("model", {}).get("helmet_mass_kg", 0.7)),
    )

    base_model = model_path.build_model(mass_map, config)

    # Apply Toen buttocks parameters
    toen_params = _get_toen_buttocks_params(config)
    if toen_params:
        click.echo("Using Toen-calibrated buttocks model")
        base_model = _apply_toen_buttocks_to_model(base_model, toen_params)

    # Build calibration cases
    cases = []
    for name in ["50ms", "75ms", "100ms", "150ms", "200ms"]:
        accel_path = CALIBRATION_YOGANANDAN_DIR / f"accel_{name}.csv"
        if not accel_path.exists():
            raise click.ClickException(f"Missing calibration input: {accel_path}")

        t, a_g, info = _process_input(
            accel_path,
            cfc=float(drop_cfg.get("cfc", 75)),
            sim_duration_ms=float(drop_cfg.get("sim_duration_ms", 200.0)),
            style_threshold_ms=float(drop_cfg.get("style_duration_threshold_ms", 300.0)),
            peak_threshold_g=float(drop_cfg.get("peak_threshold_g", 5.0)),
            freefall_threshold_g=float(drop_cfg.get("freefall_threshold_g", -0.85)),
            drop_baseline_correction=bool(drop_cfg.get("drop_baseline_correction", True)),
        )

        cases.append(PeakCalibrationCase(
            name=name,
            time_s=np.asarray(t, dtype=float),
            accel_g=np.asarray(a_g, dtype=float),
            target_peak_force_n=float(CALIBRATION_T12L1_PEAKS_KN[name]) * 1000.0,
            settle_ms=float(drop_cfg.get("gravity_settle_ms", 150.0)) if info["style"] == "flat" else 0.0,
        ))

    init_scales = load_calibration_scales(model_type, "peaks", model_path.default_scales)
    t12_idx = base_model.element_names.index("T12-L1")

    click.echo(f"Calibrating: model={model_type}, cases={len(cases)}")
    result = model_path.calibrate_peaks(
        base_model, cases, t12_element_index=t12_idx,
        init_scales=init_scales, calibrate_damping=False,
    )

    write_calibration_result(
        model_type=model_type, mode="peaks", result=result,
        cases=[{"name": c.name, "target_peak_kN": c.target_peak_force_n / 1000.0} for c in cases],
        default_scales=model_path.default_scales,
    )

    click.echo("Calibration complete.")
    click.echo(json.dumps(result.scales, indent=2))


@cli.command("simulate-drop")
def simulate_drop():
    """Run drop simulations on acceleration data."""
    config = _read_config()
    drop_cfg = config.get("drop", {})

    model_type = str(config.get("model", {}).get("type", "zwt")).lower()
    model_path = get_model_path(model_type)

    masses_path = _resolve_path(str(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON))))
    masses = _load_masses(masses_path)
    mass_map = _build_mass_map(
        masses,
        arm_recruitment=float(config.get("model", {}).get("arm_recruitment", 0.5)),
        helmet_mass=float(config.get("model", {}).get("helmet_mass_kg", 0.7)),
    )

    model = model_path.build_model(mass_map, config)

    # Apply Toen buttocks parameters
    toen_params = _get_toen_buttocks_params(config)
    if toen_params:
        click.echo("Using Toen-calibrated buttocks model")
        model = _apply_toen_buttocks_to_model(model, toen_params)

    # Apply spine calibration
    calib_mode = str(drop_cfg.get("calibration_mode", "peaks")).lower()
    scales = load_calibration_scales(model_type, calib_mode, model_path.default_scales)
    model = model_path.apply_calibration(model, scales)

    # Compute buttocks height from Toen parameters (free/uncompressed)
    buttocks_height_mm = _compute_free_buttocks_height_mm(toen_params)

    inputs_dir = _resolve_path(str(drop_cfg.get("inputs_dir", "drops")))
    pattern = str(drop_cfg.get("pattern", "*.csv"))
    out_dir = _resolve_path(str(drop_cfg.get("output_dir", "output/drop")))
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(inputs_dir.glob(pattern))
    if not files:
        raise click.ClickException(f"No drop inputs found: {inputs_dir}/{pattern}")

    click.echo(f"Simulating: model={model_type}, calibration={calib_mode}, files={len(files)}")
    click.echo(f"Buttocks height (free): {buttocks_height_mm:.1f} mm")
    click.echo(f"Output: {out_dir}")

    summary = []
    for fpath in files:
        click.echo(f"\nProcessing: {fpath.name}")

        t, a_g, info = _process_input(
            fpath,
            cfc=float(drop_cfg.get("cfc", 75)),
            sim_duration_ms=float(drop_cfg.get("sim_duration_ms", 200.0)),
            style_threshold_ms=float(drop_cfg.get("style_duration_threshold_ms", 300.0)),
            peak_threshold_g=float(drop_cfg.get("peak_threshold_g", 5.0)),
            freefall_threshold_g=float(drop_cfg.get("freefall_threshold_g", -0.85)),
            drop_baseline_correction=bool(drop_cfg.get("drop_baseline_correction", True)),
        )

        # Initial state
        y0 = np.zeros(model.size(), dtype=float)
        v0 = np.zeros(model.size(), dtype=float)
        s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

        # Gravity settling for flat-style inputs
        if info["style"] == "flat":
            settle_ms = float(drop_cfg.get("gravity_settle_ms", 150.0))
            if settle_ms > 0.0:
                dt = info["dt_s"]
                n_settle = int(round((settle_ms / 1000.0) / dt)) + 1
                t_settle = dt * np.arange(n_settle)
                a_settle = np.zeros_like(t_settle)
                sim_settle = newmark_nonlinear(model, t_settle, a_settle, y0, v0, s0)
                y0, v0 = sim_settle.y[-1].copy(), sim_settle.v[-1].copy()
                s0 = sim_settle.maxwell_state_n[-1].copy()

        sim = newmark_nonlinear(model, t, a_g, y0, v0, s0)

        # Save outputs
        run_dir = out_dir / fpath.stem
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        _write_timeseries_csv(
            run_dir / "timeseries.csv", sim.time_s, sim.base_accel_g,
            model.node_names, model.element_names,
            sim.y, sim.v, sim.a, sim.element_forces_n,
        )

        t12_idx = model.element_names.index("T12-L1")
        butt_idx = model.element_names.index("buttocks")

        peak_t12 = float(np.max(sim.element_forces_n[:, t12_idx]) / 1000.0)
        peak_butt = float(np.max(sim.element_forces_n[:, butt_idx]) / 1000.0)

        click.echo(f"  style={info['style']}, peak_T12L1={peak_t12:.3f} kN, peak_butt={peak_butt:.3f} kN")

        summary.append({
            "file": fpath.name,
            "style": info["style"],
            "dt_s": info["dt_s"],
            "sample_rate_hz": info["sample_rate_hz"],
            "bias_correction_applied": info["bias_applied"],
            "bias_correction_g": info["bias_g"],
            "peak_T12L1_kN": peak_t12,
            "peak_buttocks_kN": peak_butt,
        })

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    click.echo("\nSimulation complete.")


if __name__ == "__main__":
    cli()
