#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spine_sim.calibration import CalibrationCase, PeakCalibrationCase
from spine_sim.calibration_store import load_calibration_scales, write_calibration_result
from spine_sim.filters import cfc_filter
from spine_sim.io import parse_csv_series, resample_to_uniform
from spine_sim.model import newmark_nonlinear
from spine_sim.model_paths import get_model_path
from spine_sim.plotting import (
    DEFAULT_BUTTOCKS_HEIGHT_MM,
    plot_displacement_colored_by_force,
    plot_displacements,
    plot_forces,
    plot_gravity_settling,
)
from spine_sim.range import find_hit_range
from spine_sim.calibration_targets import CALIBRATION_T12L1_PEAKS_KN, get_case_name_from_filename


# Processing constants
CFC = 75
PEAK_THRESHOLD_G = 5.0
FREEFALL_THRESHOLD_G = -0.85

# Style detection threshold (ms)
STYLE_DURATION_THRESHOLD_MS = 300.0

# Always simulate this long (ms)
SIM_DURATION_MS = 200.0

# Default masses JSON
DEFAULT_MASSES_JSON_PATH = Path(__file__).parent / "opensim" / "fullbody.json"

# Drop inputs
DROPS_DIR = Path(__file__).parent / "drops"
DROPS_PATTERN = "*.csv"

# Calibration folder (required)
CALIBRATION_ROOT = Path(__file__).parent / "calibration"
CALIBRATION_YOGANANDAN_DIR = CALIBRATION_ROOT / "yoganandan"
CALIBRATION_CASE_NAMES = ["50ms", "75ms", "100ms", "150ms", "200ms"]

# Calibration force sign: -1 because Yoganandan reports compression as negative
CALIBRATION_FORCE_SIGN = -1.0

CALIBRATION_SCALE_BOUNDS = (0.05, 20.0)


@dataclass
class ProcessingInfo:
    sample_rate_hz: float
    dt_s: float
    start_idx: int
    end_idx: int
    duration_ms: float
    style: str  # "drop" or "flat"
    freefall_median_g: float | None
    bias_correction_g: float
    bias_correction_applied: bool


def load_masses_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_mass_map(masses: dict, arm_recruitment: float, helmet_mass: float) -> dict:
    """Build mass map using exact body names from OpenSim fullbody model."""
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


def _detect_style(duration_ms: float) -> str:
    if duration_ms < STYLE_DURATION_THRESHOLD_MS:
        return "flat"
    return "drop"


def _freefall_bias_correct(
    accel_g: np.ndarray,
    apply_correction: bool = True,
) -> tuple[np.ndarray, float | None, float, bool]:
    if len(accel_g) < 100:
        if len(accel_g) > 50:
            samples = accel_g[50:]
        else:
            return accel_g, None, 0.0, False
    else:
        samples = accel_g[50:100]

    ff_median = float(np.median(samples))
    bias = -1.0 - ff_median

    print(f"    DEBUG freefall: samples[50:100] median = {ff_median:.4f} g")
    print(f"    DEBUG freefall: bias to reach -1g = {bias:.4f} g")

    if apply_correction:
        print(f"    DEBUG freefall: applying correction")
        return accel_g + bias, ff_median, bias, True
    else:
        print(f"    DEBUG freefall: correction DISABLED, not applied")
        return accel_g, ff_median, bias, False


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _get_model_type(config: dict) -> str:
    model_cfg = config.get("model", {})
    return str(model_cfg.get("type", "maxwell")).lower()


def _get_calibration_mode(config: dict) -> str:
    calibration_cfg = config.get("calibration", {})
    return str(calibration_cfg.get("mode", "peaks")).lower()


def _get_plotting_config(config: dict) -> tuple[float, bool, bool, bool]:
    plot_cfg = config.get("plotting", {})
    buttocks_height_mm = float(plot_cfg.get("buttocks_height_mm", DEFAULT_BUTTOCKS_HEIGHT_MM))
    show_element_thickness = bool(plot_cfg.get("show_element_thickness", False))
    stack_elements = bool(plot_cfg.get("stack_elements", True))
    buttocks_clamp_to_height = bool(plot_cfg.get("buttocks_clamp_to_height", True))
    return buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height


def _get_calibration_accel_files() -> list[Path]:
    files: list[Path] = []
    for name in CALIBRATION_CASE_NAMES:
        accel_path = CALIBRATION_YOGANANDAN_DIR / f"accel_{name}.csv"
        if not accel_path.exists():
            raise FileNotFoundError(
                f"Missing calibration accel file for {name}: {accel_path.name}"
            )
        files.append(accel_path)
    return files


def load_curve_calibration_cases() -> list[CalibrationCase]:
    """
    Curve calibration: expects files in calibration/yoganandan/:
      accel_50ms.csv + force_50ms.csv, etc.
    """
    if not CALIBRATION_YOGANANDAN_DIR.exists():
        raise FileNotFoundError(
            f"Missing calibration directory: {CALIBRATION_YOGANANDAN_DIR}"
        )

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


def load_peak_calibration_cases(
    *,
    drop_baseline_correction: bool,
    settle_ms: float,
) -> tuple[list[PeakCalibrationCase], list[dict]]:
    """
    Peak calibration: expects accel files in calibration/yoganandan/:
      accel_50ms.csv ... accel_200ms.csv
    """
    if not CALIBRATION_YOGANANDAN_DIR.exists():
        raise FileNotFoundError(
            f"Missing calibration directory: {CALIBRATION_YOGANANDAN_DIR}"
        )

    cases: list[PeakCalibrationCase] = []
    meta: list[dict] = []

    for name in CALIBRATION_CASE_NAMES:
        accel_path = CALIBRATION_YOGANANDAN_DIR / f"accel_{name}.csv"
        if not accel_path.exists():
            raise FileNotFoundError(
                f"Missing peak calibration file for {name}: {accel_path.name}"
            )

        print(f"Loading peak-calibration case from {accel_path} -> {name}")

        t, a_filtered_g, _a_raw_g, info = process_input_file(
            accel_path,
            cfc=CFC,
            peak_threshold_g=PEAK_THRESHOLD_G,
            freefall_threshold_g=FREEFALL_THRESHOLD_G,
            sim_duration_ms=SIM_DURATION_MS,
            drop_baseline_correction=drop_baseline_correction,
        )

        target_kN = float(CALIBRATION_T12L1_PEAKS_KN[name])
        cases.append(
            PeakCalibrationCase(
                name=name,
                time_s=np.asarray(t, dtype=float),
                accel_g=np.asarray(a_filtered_g, dtype=float),
                target_peak_force_n=target_kN * 1000.0,
                settle_ms=(settle_ms if info.style == "flat" else 0.0),
            )
        )
        meta.append({"name": name, "target_peak_kN": target_kN})

    return cases, meta


def process_input_file(
    path: Path,
    *,
    cfc: float,
    peak_threshold_g: float,
    freefall_threshold_g: float,
    sim_duration_ms: float,
    drop_baseline_correction: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ProcessingInfo]:
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
    style = _detect_style(total_duration_ms)

    print(f"    DEBUG: total_duration = {total_duration_ms:.1f} ms -> style = {style}")
    print(f"    DEBUG: raw min={np.min(accel_raw):.4f} g, max={np.max(accel_raw):.4f} g")
    print(f"    DEBUG: filtered min={np.min(accel_filtered):.4f} g, max={np.max(accel_filtered):.4f} g")

    if style == "flat":
        start_idx = 0
        end_idx = len(t_all) - 1

        t_seg = t_all - t_all[0]
        a_seg = accel_filtered.copy()
        a_raw_seg = accel_raw.copy()

        desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
        if len(t_seg) < desired_n:
            pad_n = desired_n - len(t_seg)
            t_pad = t_seg[-1] + dt * (np.arange(pad_n) + 1)
            a_pad = np.zeros(pad_n, dtype=float)
            t_seg = np.concatenate([t_seg, t_pad])
            a_seg = np.concatenate([a_seg, a_pad])
            a_raw_seg = np.concatenate([a_raw_seg, a_pad])

        info = ProcessingInfo(
            sample_rate_hz=float(sample_rate),
            dt_s=float(dt),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            duration_ms=float(t_seg[-1] * 1000.0),
            style="flat",
            freefall_median_g=None,
            bias_correction_g=0.0,
            bias_correction_applied=False,
        )
        return t_seg, a_seg, a_raw_seg, info

    hit = find_hit_range(
        accel_filtered.tolist(),
        peak_threshold_g=peak_threshold_g,
        freefall_threshold_g=freefall_threshold_g,
    )

    if hit:
        start_idx = hit.start_idx
        end_idx = min(len(t_all) - 1, start_idx + int(round((sim_duration_ms / 1000.0) / dt)))
    else:
        start_idx = 0
        end_idx = min(len(t_all) - 1, int(round((sim_duration_ms / 1000.0) / dt)))

    t_seg = t_all[start_idx : end_idx + 1] - t_all[start_idx]
    a_seg = accel_filtered[start_idx : end_idx + 1]
    a_raw_seg = accel_raw[start_idx : end_idx + 1]

    a_seg, ff_median, bias, applied = _freefall_bias_correct(a_seg, apply_correction=drop_baseline_correction)
    if applied:
        a_raw_seg = a_raw_seg + bias

    desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
    if len(t_seg) < desired_n:
        pad_n = desired_n - len(t_seg)
        t_pad = t_seg[-1] + dt * (np.arange(pad_n) + 1)
        a_pad = -1.0 * np.ones(pad_n, dtype=float)
        t_seg = np.concatenate([t_seg, t_pad])
        a_seg = np.concatenate([a_seg, a_pad])
        a_raw_seg = np.concatenate([a_raw_seg, a_pad])

    info = ProcessingInfo(
        sample_rate_hz=float(sample_rate),
        dt_s=float(dt),
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        duration_ms=float(t_seg[-1] * 1000.0),
        style="drop",
        freefall_median_g=ff_median,
        bias_correction_g=float(bias),
        bias_correction_applied=applied,
    )
    return t_seg, a_seg, a_raw_seg, info


def _simulate_peak_case(model, case: PeakCalibrationCase):
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
) -> None:
    head_idx = model.node_names.index("HEAD")
    pelvis_idx = model.node_names.index("pelvis")

    print(f"\nDEBUG: {label}")
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

        target_kN = float(case.target_peak_force_n / 1000.0)
        residual_pct = 0.0 if target_kN == 0.0 else (peak_t12_kN - target_kN) / target_kN * 100.0

        print(
            f"  {case.name}: target={target_kN:.2f} kN, "
            f"pred={peak_t12_kN:.2f} kN ({residual_pct:+.1f}%), "
            f"t_peak={t_peak_ms:.1f} ms, "
            f"butt={peak_butt_kN:.2f} kN, "
            f"shortening={spine_shortening_mm:.1f} mm"
        )


def _report_scale_bounds(scales: dict, bounds: tuple[float, float]) -> None:
    low, high = bounds
    for key, value in scales.items():
        if value <= low * 1.01 or value >= high * 0.99:
            print(f"    DEBUG: {key}={value:.4f} is near bound [{low}, {high}]")


def _run_simulation_batch(
    *,
    model,
    input_files: list[Path],
    output_root: Path,
    heights_from_model: dict[str, float] | None,
    drop_baseline_correction: bool,
    settle_ms: float,
    buttocks_height_mm: float,
    show_element_thickness: bool,
    stack_elements: bool,
    buttocks_clamp_to_height: bool,
) -> list[dict]:
    if not input_files:
        print("No input files found for this run.")
        return []

    t12_elem_idx = model.element_names.index("T12-L1")
    butt_elem_idx = model.element_names.index("buttocks")
    head_idx = model.node_names.index("HEAD")
    pelvis_idx = model.node_names.index("pelvis")

    output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for fpath in input_files:
        print(f"\nProcessing {fpath.name}...")

        t, a_filtered_g, a_raw_g, info = process_input_file(
            fpath,
            cfc=CFC,
            peak_threshold_g=PEAK_THRESHOLD_G,
            freefall_threshold_g=FREEFALL_THRESHOLD_G,
            sim_duration_ms=SIM_DURATION_MS,
            drop_baseline_correction=drop_baseline_correction,
        )

        y0 = np.zeros(model.size(), dtype=float)
        v0 = np.zeros(model.size(), dtype=float)
        s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

        run_dir = output_root / fpath.stem
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        if info.style == "flat":
            dt = info.dt_s
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

            y0 = sim_settle.y[-1].copy()
            v0 = sim_settle.v[-1].copy()
            s0 = sim_settle.maxwell_state_n[-1].copy()

        sim = newmark_nonlinear(model, t, a_filtered_g, y0, v0, s0)
        forces = sim.element_forces_n

        f_t12 = forces[:, t12_elem_idx]
        f_butt = forces[:, butt_elem_idx]

        max_head_compression_mm = -float(np.min(sim.y[:, head_idx]) * 1000.0)
        max_pelvis_compression_mm = -float(np.min(sim.y[:, pelvis_idx]) * 1000.0)
        max_spine_shortening_mm = max_head_compression_mm - max_pelvis_compression_mm

        peak_base_g = float(np.max(a_filtered_g))
        min_base_g = float(np.min(a_filtered_g))
        peak_t12_kN = float(np.max(f_t12) / 1000.0)
        peak_butt_kN = float(np.max(f_butt) / 1000.0)

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

        plot_displacements(
            sim.time_s,
            sim.y,
            a_filtered_g,
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
            a_filtered_g,
            model.element_names,
            run_dir / "forces.png",
            highlight="T12-L1",
        )
        plot_displacement_colored_by_force(
            sim.time_s,
            sim.y,
            forces,
            a_filtered_g,
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

        elem_max_comp_mm: dict[str, float] = {}
        for e_idx, ename in enumerate(model.element_names):
            if e_idx == 0:
                comp = np.maximum(-(sim.y[:, 0] + model.gap_m[0]), 0.0)
            else:
                lower = e_idx - 1
                upper = e_idx
                comp = np.maximum(-(sim.y[:, upper] - sim.y[:, lower] + model.gap_m[e_idx]), 0.0)
            elem_max_comp_mm[ename] = float(np.max(comp) * 1000.0)

        print(f"  Style: {info.style}")
        print(f"  Sample rate: {info.sample_rate_hz:.1f} Hz, dt: {info.dt_s*1000.0:.3f} ms")
        if info.style == "drop":
            print(f"  Baseline correction: {'applied' if info.bias_correction_applied else 'not applied'} (bias={info.bias_correction_g:.4f} g)")
        print(f"  Base accel: peak={peak_base_g:.2f} g, min={min_base_g:.2f} g")
        print(f"  Peak buttocks: {peak_butt_kN:.2f} kN")
        print(f"  Peak T12-L1: {peak_t12_kN:.2f} kN @ {float(sim.time_s[np.argmax(f_t12)]*1000.0):.1f} ms")

        case_name = get_case_name_from_filename(fpath.stem)
        if case_name and case_name in CALIBRATION_T12L1_PEAKS_KN:
            ref = CALIBRATION_T12L1_PEAKS_KN[case_name]
            print(f"  Reference (Yoganandan 2021): {ref:.2f} kN")

        print(f"  Spine shortening: {max_spine_shortening_mm:.1f} mm")

        summary.append(
            {
                "file": fpath.name,
                "style": info.style,
                "sample_rate_hz": info.sample_rate_hz,
                "baseline_correction_applied": info.bias_correction_applied,
                "baseline_correction_g": info.bias_correction_g,
                "base_accel_peak_g": peak_base_g,
                "base_accel_min_g": min_base_g,
                "peak_buttocks_kN": peak_butt_kN,
                "peak_T12L1_kN": peak_t12_kN,
                "time_to_peak_ms": float(sim.time_s[np.argmax(f_t12)] * 1000.0),
                "max_spine_shortening_mm": max_spine_shortening_mm,
                "max_element_compression_mm": elem_max_comp_mm,
            }
        )

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_calibrate_peaks(config_path: Path) -> None:
    config = _read_config(config_path)
    model_type = _get_model_type(config)
    model_path = get_model_path(model_type)

    masses_json_path = Path(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON_PATH)))
    masses_data = load_masses_json(masses_json_path)

    arm_recruitment = float(config["model"].get("arm_recruitment", 0.5))
    helmet_mass = float(config["model"].get("helmet_mass_kg", 0.7))
    mass_map = build_mass_map(masses_data, arm_recruitment=arm_recruitment, helmet_mass=helmet_mass)

    heights_from_model = masses_data.get("heights_relative_to_pelvis_mm", None)

    base_model = model_path.build_model(mass_map, config)
    t12_elem_idx = base_model.element_names.index("T12-L1")
    butt_elem_idx = base_model.element_names.index("buttocks")

    drop_baseline_correction = bool(config.get("drop_baseline_correction", True))
    settle_ms = float(config.get("gravity_settle_ms", 150.0))
    buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height = _get_plotting_config(config)

    cases, case_meta = load_peak_calibration_cases(
        drop_baseline_correction=drop_baseline_correction,
        settle_ms=settle_ms,
    )

    init_scales = load_calibration_scales(model_type, "peaks", model_path.default_scales)
    print(f"Running PEAK calibration for {model_type} (stiffness-only by default)...")
    print(f"DEBUG: initial scales = {init_scales}")

    init_model = model_path.apply_calibration(base_model, init_scales)
    _summarize_peak_cases(
        "Initial peak-fit summary",
        init_model,
        cases,
        t12_elem_idx,
        butt_elem_idx,
    )

    result = model_path.calibrate_peaks(
        base_model,
        cases,
        t12_element_index=t12_elem_idx,
        init_scales=init_scales,
        calibrate_damping=False,
    )

    print(f"DEBUG: calibrated scales = {result.scales}")
    _report_scale_bounds(result.scales, CALIBRATION_SCALE_BOUNDS)

    calibrated_model = model_path.apply_calibration(base_model, result.scales)
    _summarize_peak_cases(
        "Final peak-fit summary",
        calibrated_model,
        cases,
        t12_elem_idx,
        butt_elem_idx,
    )

    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_peaks_result.json").write_text(
        json.dumps(
            {
                "mode": "peaks",
                "model_type": model_type,
                "success": result.success,
                "cost": result.cost,
                "residual_norm": result.residual_norm,
                "scales": result.scales,
                "cases": case_meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_calibration_result(model_type, "peaks", result, case_meta, model_path.default_scales)

    print("Peak calibration complete. Updated calibration file:")
    print(f"  {CALIBRATION_ROOT / f'{model_type}.json'}")

    print("\nRunning calibrated simulation on calibration inputs...")
    calibration_inputs = _get_calibration_accel_files()
    calibration_out_dir = out_dir / f"calibration_{model_type}_peaks"
    _run_simulation_batch(
        model=calibrated_model,
        input_files=calibration_inputs,
        output_root=calibration_out_dir,
        heights_from_model=heights_from_model,
        drop_baseline_correction=drop_baseline_correction,
        settle_ms=settle_ms,
        buttocks_height_mm=buttocks_height_mm,
        show_element_thickness=show_element_thickness,
        stack_elements=stack_elements,
        buttocks_clamp_to_height=buttocks_clamp_to_height,
    )
    print(f"\nCalibration simulations written to {calibration_out_dir}/")


def _run_calibrate_curves(config_path: Path) -> None:
    config = _read_config(config_path)
    model_type = _get_model_type(config)
    model_path = get_model_path(model_type)

    masses_json_path = Path(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON_PATH)))
    masses_data = load_masses_json(masses_json_path)

    arm_recruitment = float(config["model"].get("arm_recruitment", 0.5))
    helmet_mass = float(config["model"].get("helmet_mass_kg", 0.7))
    mass_map = build_mass_map(masses_data, arm_recruitment=arm_recruitment, helmet_mass=helmet_mass)

    heights_from_model = masses_data.get("heights_relative_to_pelvis_mm", None)

    base_model = model_path.build_model(mass_map, config)
    t12_elem_idx = base_model.element_names.index("T12-L1")

    cases = load_curve_calibration_cases()
    init_scales = load_calibration_scales(model_type, "curves", model_path.default_scales)

    print(f"Running CURVE calibration for {model_type} (requires calibration/yoganandan)...")
    result = model_path.calibrate_curves(
        base_model,
        cases,
        t12_elem_idx,
        init_scales=init_scales,
    )

    out_dir = Path(config.get("output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_curves_result.json").write_text(
        json.dumps(
            {
                "mode": "curves",
                "model_type": model_type,
                "success": result.success,
                "cost": result.cost,
                "residual_norm": result.residual_norm,
                "scales": result.scales,
                "cases": [c.name for c in cases],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_calibration_result(
        model_type,
        "curves",
        result,
        [c.name for c in cases],
        model_path.default_scales,
    )

    print("Curve calibration complete. Updated calibration file:")
    print(f"  {CALIBRATION_ROOT / f'{model_type}.json'}")

    drop_baseline_correction = bool(config.get("drop_baseline_correction", True))
    settle_ms = float(config.get("gravity_settle_ms", 150.0))
    buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height = _get_plotting_config(config)

    calibrated_model = model_path.apply_calibration(base_model, result.scales)
    print("\nRunning calibrated simulation on calibration inputs...")
    calibration_inputs = _get_calibration_accel_files()
    calibration_out_dir = out_dir / f"calibration_{model_type}_curves"
    _run_simulation_batch(
        model=calibrated_model,
        input_files=calibration_inputs,
        output_root=calibration_out_dir,
        heights_from_model=heights_from_model,
        drop_baseline_correction=drop_baseline_correction,
        settle_ms=settle_ms,
        buttocks_height_mm=buttocks_height_mm,
        show_element_thickness=show_element_thickness,
        stack_elements=stack_elements,
        buttocks_clamp_to_height=buttocks_clamp_to_height,
    )
    print(f"\nCalibration simulations written to {calibration_out_dir}/")


def write_timeseries_csv(
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
            row = [
                f"{time_s[i]:.6f}",
                f"{base_accel_g[i]:.6f}",
            ]
            row += [f"{(y[i, j] * 1000.0):.6f}" for j in range(y.shape[1])]
            row += [f"{v[i, j]:.6f}" for j in range(v.shape[1])]
            row += [f"{a[i, j]:.6f}" for j in range(a.shape[1])]
            row += [f"{(forces_n[i, j] / 1000.0):.6f}" for j in range(forces_n.shape[1])]
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate-peaks", action="store_true", help="Calibrate stiffness scales to Yoganandan peak forces and update calibration/<model>.json, then exit.")
    parser.add_argument("--calibrate-curves", action="store_true", help="Calibrate to force-time curves (requires calibration/yoganandan/force_*.csv) and update calibration/<model>.json, then exit.")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.json"

    if args.calibrate_peaks and args.calibrate_curves:
        raise SystemExit("Choose only one: --calibrate-peaks OR --calibrate-curves.")

    if args.calibrate_peaks:
        _run_calibrate_peaks(config_path)
        return

    if args.calibrate_curves:
        _run_calibrate_curves(config_path)
        return

    # Normal simulation run
    config = _read_config(config_path)

    model_type = _get_model_type(config)
    model_path = get_model_path(model_type)
    calibration_mode = _get_calibration_mode(config)

    masses_json_path = Path(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON_PATH)))
    masses_data = load_masses_json(masses_json_path)

    arm_recruitment = float(config["model"].get("arm_recruitment", 0.5))
    helmet_mass = float(config["model"].get("helmet_mass_kg", 0.7))
    mass_map = build_mass_map(masses_data, arm_recruitment=arm_recruitment, helmet_mass=helmet_mass)

    heights_from_model = masses_data.get("heights_relative_to_pelvis_mm", None)

    model = model_path.build_model(mass_map, config)

    # Always apply calibration for the selected path + mode
    scales = load_calibration_scales(model_type, calibration_mode, model_path.default_scales)
    model = model_path.apply_calibration(model, scales)

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model path: {model_type}, calibration: {calibration_mode}")
    drop_baseline_correction = config.get("drop_baseline_correction", True)
    print(f"Drop baseline correction: {'enabled' if drop_baseline_correction else 'disabled'}")

    drop_files = sorted(DROPS_DIR.glob(DROPS_PATTERN))
    settle_ms = float(config.get("gravity_settle_ms", 150.0))
    buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height = _get_plotting_config(config)

    _run_simulation_batch(
        model=model,
        input_files=drop_files,
        output_root=out_dir,
        heights_from_model=heights_from_model,
        drop_baseline_correction=drop_baseline_correction,
        settle_ms=settle_ms,
        buttocks_height_mm=buttocks_height_mm,
        show_element_thickness=show_element_thickness,
        stack_elements=stack_elements,
        buttocks_clamp_to_height=buttocks_clamp_to_height,
    )

    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
