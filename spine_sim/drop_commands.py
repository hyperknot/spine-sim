"""Drop calibration and simulation commands."""

from __future__ import annotations

import json
import shutil

import numpy as np

from spine_sim.buttocks import (
    apply_toen_buttocks_to_model,
    compute_free_buttocks_height_mm,
    get_toen_buttocks_params,
)
from spine_sim.calibration import PeakCalibrationCase
from spine_sim.calibration_store import load_calibration_scales, write_calibration_result
from spine_sim.calibration_targets import CALIBRATION_T12L1_PEAKS_KN
from spine_sim.config import (
    CALIBRATION_YOGANANDAN_DIR,
    DEFAULT_MASSES_JSON,
    load_masses,
    read_config,
    resolve_path,
)
from spine_sim.input_processing import process_input
from spine_sim.mass import build_mass_map
from spine_sim.model import newmark_nonlinear
from spine_sim.model_paths import get_model_path
from spine_sim.output import write_timeseries_csv


def run_calibrate_drop(echo=print) -> dict:
    """Calibrate spine model against Yoganandan data (uses Toen buttocks)."""
    config = read_config()
    drop_cfg = config.get("drop", {})

    model_type = str(config.get("model", {}).get("type", "zwt")).lower()
    model_path = get_model_path(model_type)

    masses_path = resolve_path(str(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON))))
    masses = load_masses(masses_path)
    mass_map = build_mass_map(
        masses,
        arm_recruitment=float(config.get("model", {}).get("arm_recruitment", 0.5)),
        helmet_mass=float(config.get("model", {}).get("helmet_mass_kg", 0.7)),
    )

    base_model = model_path.build_model(mass_map, config)

    # Apply Toen buttocks parameters
    toen_params = get_toen_buttocks_params(config)
    if toen_params:
        echo("Using Toen-calibrated buttocks model")
        base_model = apply_toen_buttocks_to_model(base_model, toen_params)

    # Build calibration cases
    cases = []
    for name in ["50ms", "75ms", "100ms", "150ms", "200ms"]:
        accel_path = CALIBRATION_YOGANANDAN_DIR / f"accel_{name}.csv"
        if not accel_path.exists():
            raise FileNotFoundError(f"Missing calibration input: {accel_path}")

        t, a_g, info = process_input(
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

    echo(f"Calibrating: model={model_type}, cases={len(cases)}")
    result = model_path.calibrate_peaks(
        base_model, cases, t12_element_index=t12_idx,
        init_scales=init_scales, calibrate_damping=False,
    )

    write_calibration_result(
        model_type=model_type, mode="peaks", result=result,
        cases=[{"name": c.name, "target_peak_kN": c.target_peak_force_n / 1000.0} for c in cases],
        default_scales=model_path.default_scales,
    )

    echo("Calibration complete.")
    echo(json.dumps(result.scales, indent=2))
    return result.scales


def run_simulate_drop(echo=print) -> list[dict]:
    """Run drop simulations on acceleration data."""
    config = read_config()
    drop_cfg = config.get("drop", {})

    model_type = str(config.get("model", {}).get("type", "zwt")).lower()
    model_path = get_model_path(model_type)

    masses_path = resolve_path(str(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON))))
    masses = load_masses(masses_path)
    mass_map = build_mass_map(
        masses,
        arm_recruitment=float(config.get("model", {}).get("arm_recruitment", 0.5)),
        helmet_mass=float(config.get("model", {}).get("helmet_mass_kg", 0.7)),
    )

    model = model_path.build_model(mass_map, config)

    # Apply Toen buttocks parameters
    toen_params = get_toen_buttocks_params(config)
    if toen_params:
        echo("Using Toen-calibrated buttocks model")
        model = apply_toen_buttocks_to_model(model, toen_params)

    # Apply spine calibration
    calib_mode = str(drop_cfg.get("calibration_mode", "peaks")).lower()
    scales = load_calibration_scales(model_type, calib_mode, model_path.default_scales)
    model = model_path.apply_calibration(model, scales)

    # Compute buttocks height from Toen parameters (free/uncompressed)
    buttocks_height_mm = compute_free_buttocks_height_mm(toen_params)

    inputs_dir = resolve_path(str(drop_cfg.get("inputs_dir", "drops")))
    pattern = str(drop_cfg.get("pattern", "*.csv"))
    out_dir = resolve_path(str(drop_cfg.get("output_dir", "output/drop")))
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(inputs_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No drop inputs found: {inputs_dir}/{pattern}")

    echo(f"Simulating: model={model_type}, calibration={calib_mode}, files={len(files)}")
    echo(f"Buttocks height (free): {buttocks_height_mm:.1f} mm")
    echo(f"Output: {out_dir}")

    summary = []
    for fpath in files:
        echo(f"\nProcessing: {fpath.name}")

        t, a_g, info = process_input(
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

        write_timeseries_csv(
            run_dir / "timeseries.csv", sim.time_s, sim.base_accel_g,
            model.node_names, model.element_names,
            sim.y, sim.v, sim.a, sim.element_forces_n,
        )

        t12_idx = model.element_names.index("T12-L1")
        butt_idx = model.element_names.index("buttocks")

        peak_t12 = float(np.max(sim.element_forces_n[:, t12_idx]) / 1000.0)
        peak_butt = float(np.max(sim.element_forces_n[:, butt_idx]) / 1000.0)

        echo(f"  style={info['style']}, peak_T12L1={peak_t12:.3f} kN, peak_butt={peak_butt:.3f} kN")

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
    echo("\nSimulation complete.")
    return summary
