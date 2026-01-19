"""Drop calibration and simulation commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from spine_sim.buttocks import (
    apply_toen_buttocks_to_model,
    compute_free_buttocks_height_mm,
    get_toen_buttocks_params,
)
from spine_sim.calibration import CalibrationCase, PeakCalibrationCase
from spine_sim.calibration_store import load_calibration_scales, write_calibration_result
from spine_sim.calibration_targets import CALIBRATION_T12L1_PEAKS_KN, get_case_name_from_filename
from spine_sim.config import (
    CALIBRATION_ROOT,
    CALIBRATION_YOGANANDAN_DIR,
    DEFAULT_DROP_INPUTS_DIR,
    DEFAULT_DROP_OUTPUT_DIR,
    DEFAULT_DROP_PATTERN,
    DEFAULT_MASSES_JSON,
    load_masses,
    read_config,
    resolve_path,
)
from spine_sim.input_processing import DEFAULT_DROP_BASELINE_CORRECTION, process_input
from spine_sim.range import DEFAULT_FREEFALL_THRESHOLD_G, DEFAULT_PEAK_THRESHOLD_G
from spine_sim.io import parse_csv_series, resample_to_uniform
from spine_sim.mass import build_mass_map
from spine_sim.model import newmark_nonlinear
from spine_sim.model_paths import get_model_path
from spine_sim.output import write_timeseries_csv
from spine_sim.plotting import (
    DEFAULT_BUTTOCKS_HEIGHT_MM,
    plot_displacement_colored_by_force,
    plot_displacements,
    plot_forces,
    plot_gravity_settling,
)

# Calibration case names
CALIBRATION_CASE_NAMES = ["50ms", "75ms", "100ms", "150ms", "200ms"]

# Calibration force sign: -1 because Yoganandan reports compression as negative
CALIBRATION_FORCE_SIGN = -1.0

# Calibration scale bounds
CALIBRATION_SCALE_BOUNDS = (0.05, 20.0)


def _infer_element_k2_k3(model, e_idx: int) -> tuple[float, float, str]:
    """
    Mirror the force-law selection in spine_sim/model.py:
      - k2 from poly_k2 if present else 0
      - k3 from poly_k3 if present else derived from (compression_ref_m, compression_k_mult)
    """
    k2 = 0.0
    if model.poly_k2 is not None:
        k2 = float(model.poly_k2[e_idx])

    if model.poly_k3 is not None:
        k3 = float(model.poly_k3[e_idx])
        source = "poly"
    else:
        k_lin = float(model.k_elem[e_idx])
        x_ref = float(model.compression_ref_m[e_idx])
        k_mult = float(model.compression_k_mult[e_idx])
        if x_ref <= 0.0 or k_mult <= 1.0:
            k3 = 0.0
        else:
            # Same as _k3_from_multiplier in spine_sim/model.py
            k3 = (k_mult - 1.0) * k_lin / (3.0 * x_ref * x_ref)
        source = "multiplier"
    return k2, k3, source


def _format_buttocks_model_debug(model, *, buttocks_height_mm: float, buttocks_clamp_to_height: bool) -> str:
    """Format detailed debug info about buttocks model parameters."""
    e0 = 0  # buttocks element index is always 0 in this model
    k2, k3, k3_source = _infer_element_k2_k3(model, e0)

    lines: list[str] = []
    lines.append("DEBUG buttocks element model parameters:")
    lines.append(f"  element_name = {model.element_names[e0]}")
    lines.append(f"  k_lin = {model.k_elem[e0]:.6g} N/m")
    lines.append(f"  c_lin = {model.c_elem[e0]:.6g} Ns/m")
    lines.append(f"  gap = {model.gap_m[e0]*1000.0:.3f} mm")
    lines.append(f"  compression_only = {bool(model.compression_only[e0])}")
    lines.append(f"  damping_compression_only = {bool(model.damping_compression_only[e0])}")
    lines.append(f"  k2 = {k2:.6g} N/m^2")
    lines.append(f"  k3 = {k3:.6g} N/m^3 (source={k3_source})")
    lines.append(f"  ref_compression = {model.compression_ref_m[e0]*1000.0:.3f} mm")
    lines.append(f"  k_mult_at_ref = {model.compression_k_mult[e0]:.6g}")

    if model.compression_limit_m is not None and model.compression_limit_m.size:
        limit_mm = float(model.compression_limit_m[e0] * 1000.0)
        stop_k = 0.0
        smooth_mm = 0.0
        if model.compression_stop_k is not None and model.compression_stop_k.size:
            stop_k = float(model.compression_stop_k[e0])
        if model.compression_stop_smoothing_m is not None and model.compression_stop_smoothing_m.size:
            smooth_mm = float(model.compression_stop_smoothing_m[e0] * 1000.0)
        lines.append(f"  compression_limit = {limit_mm:.3f} mm")
        lines.append(f"  compression_stop_k = {stop_k:.6g} N/m")
        lines.append(f"  compression_stop_smoothing = {smooth_mm:.3f} mm")

    if model.maxwell_k.size:
        mx = model.maxwell_k[e0, :]
        tau = model.maxwell_tau_s[e0, :]
        lines.append(f"  maxwell_branches = {mx.size}")
        for i in range(mx.size):
            lines.append(f"    branch[{i}]: k={mx[i]:.6g} N/m, tau={tau[i]*1000.0:.3f} ms")
    else:
        lines.append("  maxwell_branches = 0")

    lines.append("DEBUG buttocks plotting parameters:")
    lines.append(f"  plotting.buttocks_height_mm = {buttocks_height_mm:.3f} mm")
    lines.append(f"  plotting.buttocks_clamp_to_height = {buttocks_clamp_to_height}")
    return "\n".join(lines)


def _compute_buttocks_debug_metrics(
    *,
    model,
    sim,
    buttocks_height_mm: float,
    buttocks_clamp_to_height: bool,
) -> dict:
    """
    Metrics that explain why plots may require a large buttocks_height_mm.
    """
    pelvis_idx = model.node_names.index("pelvis")
    butt_elem_idx = model.element_names.index("buttocks")

    y_pelvis_m = sim.y[:, pelvis_idx]
    y_pelvis_mm = y_pelvis_m * 1000.0
    y_pelvis_min_mm = float(np.min(y_pelvis_mm))

    gap_mm = float(model.gap_m[butt_elem_idx] * 1000.0)
    butt_comp_m = np.maximum(-(y_pelvis_m + model.gap_m[butt_elem_idx]), 0.0)
    butt_comp_max_mm = float(np.max(butt_comp_m) * 1000.0)

    butt_force_n = sim.element_forces_n[:, butt_elem_idx]
    butt_force_peak_kN = float(np.max(butt_force_n) / 1000.0)

    if buttocks_clamp_to_height:
        min_thickness_mm = float(np.clip(buttocks_height_mm + min(y_pelvis_min_mm, 0.0), 0.0, buttocks_height_mm))
        bottomed_out = (buttocks_height_mm + y_pelvis_min_mm) <= 0.0
    else:
        min_thickness_mm = float(buttocks_height_mm + min(y_pelvis_min_mm, 0.0))
        bottomed_out = False

    recommended_height_mm = float(max(buttocks_height_mm, -y_pelvis_min_mm + 10.0))

    return {
        "buttocks_peak_force_kN": butt_force_peak_kN,
        "buttocks_max_compression_mm": butt_comp_max_mm,
        "pelvis_min_y_mm": y_pelvis_min_mm,
        "plot_min_buttocks_thickness_mm": min_thickness_mm,
        "plot_buttocks_bottomed_out": bool(bottomed_out),
        "plot_recommended_buttocks_height_mm": recommended_height_mm,
        "buttocks_gap_mm": gap_mm,
    }


def _report_scale_bounds(scales: dict, bounds: tuple[float, float], echo=print) -> None:
    """Report scales that are near their bounds."""
    low, high = bounds
    for key, value in scales.items():
        if value <= low * 1.01 or value >= high * 0.99:
            echo(f"    DEBUG: {key}={value:.4f} is near bound [{low}, {high}]")


def _simulate_peak_case(model, case: PeakCalibrationCase):
    """Simulate a single peak calibration case."""
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


def _summarize_peak_cases(
    label: str,
    model,
    cases: list[PeakCalibrationCase],
    t12_element_index: int,
    buttocks_element_index: int,
    *,
    buttocks_height_mm: float,
    buttocks_clamp_to_height: bool,
    echo=print,
) -> None:
    """Print detailed peak-fit summary for all cases."""
    head_idx = model.node_names.index("HEAD")
    pelvis_idx = model.node_names.index("pelvis")

    echo(f"\nDEBUG: {label}")
    for case in cases:
        sim = _simulate_peak_case(model, case)
        f_t12 = sim.element_forces_n[:, t12_element_index]
        f_butt = sim.element_forces_n[:, buttocks_element_index]

        peak_t12_kN = float(np.max(f_t12) / 1000.0)
        peak_butt_kN = float(np.max(f_butt) / 1000.0)
        t_peak_ms = float(sim.time_s[np.argmax(f_t12)] * 1000.0)

        max_head_compression_mm = -float(np.min(sim.y[:, head_idx]) * 1000.0)
        max_pelvis_compression_mm = -float(np.min(sim.y[:, pelvis_idx]) * 1000.0)
        spine_shortening_mm = max_head_compression_mm - max_pelvis_compression_mm

        butt_debug = _compute_buttocks_debug_metrics(
            model=model,
            sim=sim,
            buttocks_height_mm=buttocks_height_mm,
            buttocks_clamp_to_height=buttocks_clamp_to_height,
        )

        target_kN = float(case.target_peak_force_n / 1000.0)
        residual_pct = 0.0 if target_kN == 0.0 else (peak_t12_kN - target_kN) / target_kN * 100.0

        echo(
            f"  {case.name}: target={target_kN:.2f} kN, "
            f"pred={peak_t12_kN:.2f} kN ({residual_pct:+.1f}%), "
            f"t_peak={t_peak_ms:.1f} ms, "
            f"butt_peak={peak_butt_kN:.2f} kN, "
            f"butt_comp_max={butt_debug['buttocks_max_compression_mm']:.1f} mm, "
            f"pelvis_y_min={butt_debug['pelvis_min_y_mm']:.1f} mm, "
            f"plot_bottomed_out={butt_debug['plot_buttocks_bottomed_out']}, "
            f"shortening={spine_shortening_mm:.1f} mm"
        )


def _get_plotting_config(config: dict) -> tuple[float, bool, bool, bool, dict | None]:
    """Extract plotting config from config dict."""
    plot_cfg = config.get("plotting", {})
    buttocks_height_mm = float(plot_cfg.get("buttocks_height_mm", DEFAULT_BUTTOCKS_HEIGHT_MM))
    show_element_thickness = bool(plot_cfg.get("show_element_thickness", False))
    stack_elements = bool(plot_cfg.get("stack_elements", True))
    buttocks_clamp_to_height = bool(plot_cfg.get("buttocks_clamp_to_height", True))

    # Load heights from masses JSON if available
    masses_path = resolve_path(str(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON))))
    masses = load_masses(masses_path)
    heights_from_model = masses.get("heights_relative_to_pelvis_mm", None)

    return buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height, heights_from_model


def _process_input_verbose(
    fpath: Path,
    drop_cfg: dict,
    echo=print,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Process input file with verbose debug output."""
    echo(f"Loading: {fpath}")

    t, a_g, info = process_input(
        fpath,
        cfc=float(drop_cfg.get("cfc", 75)),
        sim_duration_ms=float(drop_cfg.get("sim_duration_ms", 200.0)),
        style_threshold_ms=float(drop_cfg.get("style_duration_threshold_ms", 300.0)),
        peak_threshold_g=float(drop_cfg.get("peak_threshold_g", DEFAULT_PEAK_THRESHOLD_G)),
        freefall_threshold_g=float(drop_cfg.get("freefall_threshold_g", DEFAULT_FREEFALL_THRESHOLD_G)),
        drop_baseline_correction=bool(drop_cfg.get("drop_baseline_correction", DEFAULT_DROP_BASELINE_CORRECTION)),
    )

    echo(f"    DEBUG: total_duration = {info['duration_ms']:.1f} ms -> style = {info['style']}")
    echo(f"    DEBUG: raw min={info.get('raw_min_g', 0.0):.4f} g, max={info.get('raw_max_g', 0.0):.4f} g")
    echo(f"    DEBUG: filtered min={float(np.min(a_g)):.4f} g, max={float(np.max(a_g)):.4f} g")

    return t, a_g, info


def _run_simulation_batch(
    *,
    model,
    input_files: list[Path],
    output_root: Path,
    drop_cfg: dict,
    buttocks_height_mm: float,
    show_element_thickness: bool,
    stack_elements: bool,
    buttocks_clamp_to_height: bool,
    heights_from_model: dict | None,
    echo=print,
) -> list[dict]:
    """Run simulation on a batch of input files with verbose output and PNG generation."""
    if not input_files:
        echo("No input files found for this run.")
        return []

    t12_elem_idx = model.element_names.index("T12-L1")
    butt_elem_idx = model.element_names.index("buttocks")
    head_idx = model.node_names.index("HEAD")
    pelvis_idx = model.node_names.index("pelvis")

    echo(_format_buttocks_model_debug(model, buttocks_height_mm=buttocks_height_mm, buttocks_clamp_to_height=buttocks_clamp_to_height))

    output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    settle_ms = float(drop_cfg.get("gravity_settle_ms", 150.0))

    for fpath in input_files:
        echo(f"\nProcessing {fpath.name}...")

        t, a_g, info = process_input(
            fpath,
            cfc=float(drop_cfg.get("cfc", 75)),
            sim_duration_ms=float(drop_cfg.get("sim_duration_ms", 200.0)),
            style_threshold_ms=float(drop_cfg.get("style_duration_threshold_ms", 300.0)),
            peak_threshold_g=float(drop_cfg.get("peak_threshold_g", DEFAULT_PEAK_THRESHOLD_G)),
            freefall_threshold_g=float(drop_cfg.get("freefall_threshold_g", DEFAULT_FREEFALL_THRESHOLD_G)),
            drop_baseline_correction=bool(drop_cfg.get("drop_baseline_correction", DEFAULT_DROP_BASELINE_CORRECTION)),
        )

        echo(f"    DEBUG: total_duration = {info['duration_ms']:.1f} ms -> style = {info['style']}")
        echo(f"    DEBUG: raw min={info.get('raw_min_g', 0.0):.4f} g, max={info.get('raw_max_g', 0.0):.4f} g")
        echo(f"    DEBUG: filtered min={float(np.min(a_g)):.4f} g, max={float(np.max(a_g)):.4f} g")

        # Initial state
        y0 = np.zeros(model.size(), dtype=float)
        v0 = np.zeros(model.size(), dtype=float)
        s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

        run_dir = output_root / fpath.stem
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        # Gravity settling for flat-style inputs
        if info["style"] == "flat":
            if settle_ms > 0.0:
                dt = info["dt_s"]
                n_settle = int(round((settle_ms / 1000.0) / dt)) + 1
                t_settle = dt * np.arange(n_settle)
                a_settle = np.zeros_like(t_settle)
                sim_settle = newmark_nonlinear(model, t_settle, a_settle, y0, v0, s0)

                plot_gravity_settling(
                    sim_settle.time_s,
                    sim_settle.y,
                    model.node_names,
                    model.element_names,
                    run_dir / "gravity_settling.png",
                    heights_from_model=heights_from_model,
                    buttocks_height_mm=buttocks_height_mm,
                    show_element_thickness=show_element_thickness,
                    stack_elements=stack_elements,
                    buttocks_clamp_to_height=buttocks_clamp_to_height,
                )

                y0, v0 = sim_settle.y[-1].copy(), sim_settle.v[-1].copy()
                s0 = sim_settle.maxwell_state_n[-1].copy()

        sim = newmark_nonlinear(model, t, a_g, y0, v0, s0)
        forces = sim.element_forces_n

        f_t12 = forces[:, t12_elem_idx]
        f_butt = forces[:, butt_elem_idx]

        max_head_compression_mm = -float(np.min(sim.y[:, head_idx]) * 1000.0)
        max_pelvis_compression_mm = -float(np.min(sim.y[:, pelvis_idx]) * 1000.0)
        max_spine_shortening_mm = max_head_compression_mm - max_pelvis_compression_mm

        peak_base_g = float(np.max(a_g))
        min_base_g = float(np.min(a_g))
        peak_t12_kN = float(np.max(f_t12) / 1000.0)
        peak_butt_kN = float(np.max(f_butt) / 1000.0)
        t_peak_ms = float(sim.time_s[np.argmax(f_t12)] * 1000.0)

        butt_debug = _compute_buttocks_debug_metrics(
            model=model,
            sim=sim,
            buttocks_height_mm=buttocks_height_mm,
            buttocks_clamp_to_height=buttocks_clamp_to_height,
        )

        # Save CSV
        write_timeseries_csv(
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

        # Generate plots
        plot_displacements(
            sim.time_s,
            sim.y,
            a_g,
            model.node_names,
            model.element_names,
            run_dir / "displacements.png",
            heights_from_model=heights_from_model,
            buttocks_height_mm=buttocks_height_mm,
            reference_frame="base",
            show_element_thickness=show_element_thickness,
            stack_elements=stack_elements,
            buttocks_clamp_to_height=buttocks_clamp_to_height,
        )
        plot_forces(
            sim.time_s,
            forces,
            a_g,
            model.element_names,
            run_dir / "forces.png",
            highlight="T12-L1",
        )
        plot_displacement_colored_by_force(
            sim.time_s,
            sim.y,
            forces,
            a_g,
            model.node_names,
            model.element_names,
            run_dir / "mixed.png",
            heights_from_model=heights_from_model,
            buttocks_height_mm=buttocks_height_mm,
            reference_frame="base",
            show_element_thickness=show_element_thickness,
            stack_elements=stack_elements,
            buttocks_clamp_to_height=buttocks_clamp_to_height,
        )

        # Print verbose output
        echo(f"  Style: {info['style']}")
        echo(f"  Sample rate: {info['sample_rate_hz']:.1f} Hz, dt: {info['dt_s']*1000.0:.3f} ms")
        if info["style"] == "drop":
            echo(f"  Baseline correction: {'applied' if info['bias_applied'] else 'not applied'} (bias={info['bias_g']:.4f} g)")
        echo(f"  Base accel: peak={peak_base_g:.2f} g, min={min_base_g:.2f} g")
        echo(f"  Peak buttocks: {peak_butt_kN:.2f} kN")
        echo(f"  Peak T12-L1: {peak_t12_kN:.2f} kN @ {t_peak_ms:.1f} ms")

        echo(
            f"  DEBUG buttocks: max_compression={butt_debug['buttocks_max_compression_mm']:.1f} mm, "
            f"pelvis_y_min={butt_debug['pelvis_min_y_mm']:.1f} mm, "
            f"gap={butt_debug['buttocks_gap_mm']:.2f} mm"
        )
        if buttocks_clamp_to_height:
            echo(
                f"  DEBUG buttocks plot: height={buttocks_height_mm:.1f} mm, "
                f"min_thickness={butt_debug['plot_min_buttocks_thickness_mm']:.1f} mm, "
                f"bottomed_out={butt_debug['plot_buttocks_bottomed_out']}"
            )
            if butt_debug["plot_buttocks_bottomed_out"]:
                echo(
                    f"  DEBUG buttocks plot: RECOMMENDED buttocks_height_mm={butt_debug['plot_recommended_buttocks_height_mm']:.1f} mm "
                    f"to avoid plot bottom-out (purely for visualization)."
                )

        # Show reference values for calibration cases
        case_name = get_case_name_from_filename(fpath.stem)
        if case_name and case_name in CALIBRATION_T12L1_PEAKS_KN:
            ref = CALIBRATION_T12L1_PEAKS_KN[case_name]
            echo(f"  Reference (Yoganandan 2021): {ref:.2f} kN")

        echo(f"  Spine shortening: {max_spine_shortening_mm:.1f} mm")

        # Compute element max compressions
        elem_max_comp_mm: dict[str, float] = {}
        for e_idx, ename in enumerate(model.element_names):
            if e_idx == 0:
                comp = np.maximum(-(sim.y[:, 0] + model.gap_m[0]), 0.0)
            else:
                lower = e_idx - 1
                upper = e_idx
                comp = np.maximum(-(sim.y[:, upper] - sim.y[:, lower] + model.gap_m[e_idx]), 0.0)
            elem_max_comp_mm[ename] = float(np.max(comp) * 1000.0)

        summary.append({
            "file": fpath.name,
            "style": info["style"],
            "sample_rate_hz": info["sample_rate_hz"],
            "baseline_correction_applied": info["bias_applied"],
            "baseline_correction_g": info["bias_g"],
            "base_accel_peak_g": peak_base_g,
            "base_accel_min_g": min_base_g,
            "peak_buttocks_kN": peak_butt_kN,
            "peak_T12L1_kN": peak_t12_kN,
            "time_to_peak_ms": t_peak_ms,
            "max_spine_shortening_mm": max_spine_shortening_mm,
            "max_element_compression_mm": elem_max_comp_mm,
            "buttocks_debug": butt_debug,
        })

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_curve_calibration_cases() -> list[CalibrationCase]:
    """
    Curve calibration: expects files in calibration/yoganandan/:
      accel_50ms.csv + force_50ms.csv, etc.
    """
    if not CALIBRATION_YOGANANDAN_DIR.exists():
        raise FileNotFoundError(f"Missing calibration directory: {CALIBRATION_YOGANANDAN_DIR}")

    cases = []
    for name in CALIBRATION_CASE_NAMES:
        accel_path = CALIBRATION_YOGANANDAN_DIR / f"accel_{name}.csv"
        force_path = CALIBRATION_YOGANANDAN_DIR / f"force_{name}.csv"
        if not accel_path.exists() or not force_path.exists():
            raise FileNotFoundError(
                f"Missing calibration curve files for {name}: {accel_path.name}, {force_path.name}"
            )

        accel_series = parse_csv_series(
            accel_path,
            time_candidates=["time", "time0", "t"],
            value_candidates=["accel", "acceleration"],
        )
        force_series = parse_csv_series(
            force_path,
            time_candidates=["time", "time0", "t"],
            value_candidates=["force", "spinal", "load"],
        )

        accel_series, _ = resample_to_uniform(accel_series)
        force_series, _ = resample_to_uniform(force_series)

        accel_g = np.asarray(accel_series.values, dtype=float)
        force_n = np.asarray(force_series.values, dtype=float) * 1000.0 * CALIBRATION_FORCE_SIGN

        cases.append(
            CalibrationCase(
                name=name,
                time_s=np.asarray(accel_series.time_s, dtype=float),
                accel_g=accel_g,
                force_time_s=np.asarray(force_series.time_s, dtype=float),
                force_n=force_n,
            )
        )
    return cases


def run_calibrate_drop(echo=print, mode: str = "peaks") -> dict:
    """Calibrate spine model against Yoganandan data (uses Toen buttocks).

    Args:
        echo: Function to use for output (default: print)
        mode: Calibration mode - "peaks" or "curves"
    """
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

    heights_from_model = masses.get("heights_relative_to_pelvis_mm", None)

    base_model = model_path.build_model(mass_map, config)

    # Apply Toen buttocks parameters
    toen_params = get_toen_buttocks_params(config)
    if toen_params:
        echo("Using Toen-calibrated buttocks model")
        base_model = apply_toen_buttocks_to_model(base_model, toen_params)

    # Get plotting config
    buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height, _ = _get_plotting_config(config)

    # Print buttocks model debug info
    echo(_format_buttocks_model_debug(base_model, buttocks_height_mm=buttocks_height_mm, buttocks_clamp_to_height=buttocks_clamp_to_height))

    t12_elem_idx = base_model.element_names.index("T12-L1")
    butt_elem_idx = base_model.element_names.index("buttocks")

    settle_ms = float(drop_cfg.get("gravity_settle_ms", 150.0))

    if mode == "curves":
        # Curve calibration
        cases = _load_curve_calibration_cases()
        init_scales = load_calibration_scales(model_type, "curves", model_path.default_scales)

        echo(f"Running CURVE calibration for {model_type} (requires calibration/yoganandan)...")
        echo(f"DEBUG: initial scales = {init_scales}")

        result = model_path.calibrate_curves(
            base_model,
            cases,
            t12_elem_idx,
            init_scales=init_scales,
        )

        echo(f"DEBUG: calibrated scales = {result.scales}")
        _report_scale_bounds(result.scales, CALIBRATION_SCALE_BOUNDS, echo)

        write_calibration_result(
            model_type=model_type, mode="curves", result=result,
            cases=[c.name for c in cases],
            default_scales=model_path.default_scales,
        )

        echo("Curve calibration complete. Updated calibration file:")
        echo(f"  {CALIBRATION_ROOT / f'{model_type}.json'}")

        # Apply calibration for simulation
        calibrated_model = model_path.apply_calibration(base_model, result.scales)
        calibration_out_dir = resolve_path(str(drop_cfg.get("output_dir", str(DEFAULT_DROP_OUTPUT_DIR)))).parent / f"calibration_{model_type}_curves"

    else:
        # Peak calibration (default)
        # Build calibration cases
        cases = []
        for name in CALIBRATION_CASE_NAMES:
            accel_path = CALIBRATION_YOGANANDAN_DIR / f"accel_{name}.csv"
            if not accel_path.exists():
                raise FileNotFoundError(f"Missing calibration input: {accel_path}")

            echo(f"Loading peak-calibration case from {accel_path} -> {name}")

            t, a_g, info = process_input(
                accel_path,
                cfc=float(drop_cfg.get("cfc", 75)),
                sim_duration_ms=float(drop_cfg.get("sim_duration_ms", 200.0)),
                style_threshold_ms=float(drop_cfg.get("style_duration_threshold_ms", 300.0)),
                peak_threshold_g=float(drop_cfg.get("peak_threshold_g", DEFAULT_PEAK_THRESHOLD_G)),
                freefall_threshold_g=float(drop_cfg.get("freefall_threshold_g", DEFAULT_FREEFALL_THRESHOLD_G)),
                drop_baseline_correction=bool(drop_cfg.get("drop_baseline_correction", DEFAULT_DROP_BASELINE_CORRECTION)),
            )

            echo(f"    DEBUG: total_duration = {info['duration_ms']:.1f} ms -> style = {info['style']}")
            echo(f"    DEBUG: raw min={info.get('raw_min_g', 0.0):.4f} g, max={info.get('raw_max_g', 0.0):.4f} g")
            echo(f"    DEBUG: filtered min={float(np.min(a_g)):.4f} g, max={float(np.max(a_g)):.4f} g")

            cases.append(PeakCalibrationCase(
                name=name,
                time_s=np.asarray(t, dtype=float),
                accel_g=np.asarray(a_g, dtype=float),
                target_peak_force_n=float(CALIBRATION_T12L1_PEAKS_KN[name]) * 1000.0,
                settle_ms=settle_ms if info["style"] == "flat" else 0.0,
            ))

        init_scales = load_calibration_scales(model_type, "peaks", model_path.default_scales)

        echo(f"Running PEAK calibration for {model_type} (stiffness-only by default)...")
        echo(f"DEBUG: initial scales = {init_scales}")

        # Show initial peak-fit summary
        init_model = model_path.apply_calibration(base_model, init_scales)
        _summarize_peak_cases(
            "Initial peak-fit summary",
            init_model,
            cases,
            t12_elem_idx,
            butt_elem_idx,
            buttocks_height_mm=buttocks_height_mm,
            buttocks_clamp_to_height=buttocks_clamp_to_height,
            echo=echo,
        )

        result = model_path.calibrate_peaks(
            base_model, cases, t12_element_index=t12_elem_idx,
            init_scales=init_scales, calibrate_damping=False,
        )

        echo(f"DEBUG: calibrated scales = {result.scales}")
        _report_scale_bounds(result.scales, CALIBRATION_SCALE_BOUNDS, echo)

        # Show final peak-fit summary
        calibrated_model = model_path.apply_calibration(base_model, result.scales)
        _summarize_peak_cases(
            "Final peak-fit summary",
            calibrated_model,
            cases,
            t12_elem_idx,
            butt_elem_idx,
            buttocks_height_mm=buttocks_height_mm,
            buttocks_clamp_to_height=buttocks_clamp_to_height,
            echo=echo,
        )

        write_calibration_result(
            model_type=model_type, mode="peaks", result=result,
            cases=[{"name": c.name, "target_peak_kN": c.target_peak_force_n / 1000.0} for c in cases],
            default_scales=model_path.default_scales,
        )

        echo("Peak calibration complete. Updated calibration file:")
        echo(f"  {CALIBRATION_ROOT / f'{model_type}.json'}")

        calibration_out_dir = resolve_path(str(drop_cfg.get("output_dir", str(DEFAULT_DROP_OUTPUT_DIR)))).parent / f"calibration_{model_type}_peaks"

    # Run calibrated simulation on calibration inputs (always use yoganandan folder)
    echo("\nRunning calibrated simulation on calibration inputs...")
    calibration_inputs = sorted(CALIBRATION_YOGANANDAN_DIR.glob("accel_*.csv"))

    _run_simulation_batch(
        model=calibrated_model,
        input_files=calibration_inputs,
        output_root=calibration_out_dir,
        drop_cfg=drop_cfg,
        buttocks_height_mm=buttocks_height_mm,
        show_element_thickness=show_element_thickness,
        stack_elements=stack_elements,
        buttocks_clamp_to_height=buttocks_clamp_to_height,
        heights_from_model=heights_from_model,
        echo=echo,
    )

    echo(f"\nCalibration simulations written to {calibration_out_dir}/")
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

    # Get plotting config
    _, show_element_thickness, stack_elements, buttocks_clamp_to_height, heights_from_model = _get_plotting_config(config)

    inputs_dir = resolve_path(str(drop_cfg.get("inputs_dir", str(DEFAULT_DROP_INPUTS_DIR))))
    pattern = str(drop_cfg.get("pattern", DEFAULT_DROP_PATTERN))
    out_dir = resolve_path(str(drop_cfg.get("output_dir", str(DEFAULT_DROP_OUTPUT_DIR))))

    files = sorted(inputs_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No drop inputs found: {inputs_dir}/{pattern}")

    echo(f"Model path: {model_type}, calibration: {calib_mode}")
    echo(f"Simulating: model={model_type}, calibration={calib_mode}, files={len(files)}")
    echo(f"Buttocks height (free): {buttocks_height_mm:.1f} mm")
    echo(f"Output: {out_dir}")

    summary = _run_simulation_batch(
        model=model,
        input_files=files,
        output_root=out_dir,
        drop_cfg=drop_cfg,
        buttocks_height_mm=buttocks_height_mm,
        show_element_thickness=show_element_thickness,
        stack_elements=stack_elements,
        buttocks_clamp_to_height=buttocks_clamp_to_height,
        heights_from_model=heights_from_model,
        echo=echo,
    )

    echo(f"\nResults written to {out_dir}/")
    return summary
