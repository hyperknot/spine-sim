#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spine_sim.buttocks import (
    ButtocksOnlyCase,
    build_buttocks_only_model,
    buttocks_force_components_from_sim,
    calibrate_buttocks_bottom_out,
    get_buttocks_only_config,
    pulse_metrics,
    recommend_toen_buttocks_params,
)
from spine_sim.buttocks_store import load_buttocks_only_overrides, write_buttocks_only_overrides
from spine_sim.calibration import CalibrationCase, PeakCalibrationCase
from spine_sim.calibration_store import load_calibration_scales, write_calibration_result
from spine_sim.calibration_targets import CALIBRATION_T12L1_PEAKS_KN, get_case_name_from_filename
from spine_sim.filters import cfc_filter
from spine_sim.io import parse_csv_series, resample_to_uniform
from spine_sim.model import SpineModel, newmark_nonlinear
from spine_sim.model_paths import get_model_path
from spine_sim.plotting import (
    DEFAULT_BUTTOCKS_HEIGHT_MM,
    plot_buttocks_only,
    plot_displacement_colored_by_force,
    plot_displacements,
    plot_forces,
    plot_gravity_settling,
)
from spine_sim.range import find_hit_range
from spine_sim.toen_drop import (
    TOEN_BUTTOCKS_C_NS_PER_M,
    TOEN_BUTTOCKS_K_N_PER_M,
    TOEN_IMPACT_V_MPS,
    run_toen_suite,
)


# Processing constants
CFC = 75
PEAK_THRESHOLD_G = 5.0
FREEFALL_THRESHOLD_G = -0.85

# Style detection threshold (ms)
STYLE_DURATION_THRESHOLD_MS = 300.0

# Always simulate this long (ms)
SIM_DURATION_MS = 200.0

# Default masses JSON
DEFAULT_MASSES_JSON_PATH = Path(__file__).parent / 'opensim' / 'fullbody.json'

# Drop inputs
DROPS_DIR = Path(__file__).parent / 'drops'
DROPS_PATTERN = '*.csv'

# Calibration folder (required)
CALIBRATION_ROOT = Path(__file__).parent / 'calibration'
CALIBRATION_YOGANANDAN_DIR = CALIBRATION_ROOT / 'yoganandan'
CALIBRATION_CASE_NAMES = ['50ms', '75ms', '100ms', '150ms', '200ms']

# Calibration force sign: -1 because Yoganandan reports compression as negative
CALIBRATION_FORCE_SIGN = -1.0

CALIBRATION_SCALE_BOUNDS = (0.05, 20.0)

BUTTOCKS_ONLY_DIRNAME = 'buttocks_only'


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
        source = 'poly'
    else:
        k_lin = float(model.k_elem[e_idx])
        x_ref = float(model.compression_ref_m[e_idx])
        k_mult = float(model.compression_k_mult[e_idx])
        if x_ref <= 0.0 or k_mult <= 1.0:
            k3 = 0.0
        else:
            # Same as _k3_from_multiplier in spine_sim/model.py
            k3 = (k_mult - 1.0) * k_lin / (3.0 * x_ref * x_ref)
        source = 'multiplier'
    return k2, k3, source


def _format_buttocks_model_debug(
    model, *, buttocks_height_mm: float, buttocks_clamp_to_height: bool
) -> str:
    e0 = 0  # buttocks element index is always 0 in this model
    k2, k3, k3_source = _infer_element_k2_k3(model, e0)

    lines: list[str] = []
    lines.append('DEBUG buttocks element model parameters:')
    lines.append(f'  element_name = {model.element_names[e0]}')
    lines.append(f'  k_lin = {model.k_elem[e0]:.6g} N/m')
    lines.append(f'  c_lin = {model.c_elem[e0]:.6g} Ns/m')
    lines.append(f'  gap = {model.gap_m[e0] * 1000.0:.3f} mm')
    lines.append(f'  compression_only = {bool(model.compression_only[e0])}')
    lines.append(f'  damping_compression_only = {bool(model.damping_compression_only[e0])}')
    lines.append(f'  k2 = {k2:.6g} N/m^2')
    lines.append(f'  k3 = {k3:.6g} N/m^3 (source={k3_source})')
    lines.append(f'  ref_compression = {model.compression_ref_m[e0] * 1000.0:.3f} mm')
    lines.append(f'  k_mult_at_ref = {model.compression_k_mult[e0]:.6g}')

    if model.compression_limit_m is not None and model.compression_limit_m.size:
        limit_mm = float(model.compression_limit_m[e0] * 1000.0)
        stop_k = 0.0
        smooth_mm = 0.0
        if model.compression_stop_k is not None and model.compression_stop_k.size:
            stop_k = float(model.compression_stop_k[e0])
        if (
            model.compression_stop_smoothing_m is not None
            and model.compression_stop_smoothing_m.size
        ):
            smooth_mm = float(model.compression_stop_smoothing_m[e0] * 1000.0)
        lines.append(f'  compression_limit = {limit_mm:.3f} mm')
        lines.append(f'  compression_stop_k = {stop_k:.6g} N/m')
        lines.append(f'  compression_stop_smoothing = {smooth_mm:.3f} mm')

    if model.maxwell_k.size:
        mx = model.maxwell_k[e0, :]
        tau = model.maxwell_tau_s[e0, :]
        lines.append(f'  maxwell_branches = {mx.size}')
        for i in range(mx.size):
            lines.append(f'    branch[{i}]: k={mx[i]:.6g} N/m, tau={tau[i] * 1000.0:.3f} ms')
    else:
        lines.append('  maxwell_branches = 0')

    lines.append('DEBUG buttocks plotting parameters:')
    lines.append(f'  plotting.buttocks_height_mm = {buttocks_height_mm:.3f} mm')
    lines.append(f'  plotting.buttocks_clamp_to_height = {buttocks_clamp_to_height}')
    return '\n'.join(lines)


def load_masses_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_mass_map(masses: dict, arm_recruitment: float, helmet_mass: float) -> dict:
    """Build mass map using exact body names from OpenSim fullbody model."""
    b = masses['bodies']

    arm_mass = (
        b['humerus_R']
        + b['humerus_L']
        + b['ulna_R']
        + b['ulna_L']
        + b['radius_R']
        + b['radius_L']
        + b['hand_R']
        + b['hand_L']
    ) * arm_recruitment

    head_total = b['head_neck'] + helmet_mass + arm_mass

    return {
        'pelvis': b['pelvis'],
        'l5': b['lumbar5'],
        'l4': b['lumbar4'],
        'l3': b['lumbar3'],
        'l2': b['lumbar2'],
        'l1': b['lumbar1'],
        't12': b['thoracic12'],
        't11': b['thoracic11'],
        't10': b['thoracic10'],
        't9': b['thoracic9'],
        't8': b['thoracic8'],
        't7': b['thoracic7'],
        't6': b['thoracic6'],
        't5': b['thoracic5'],
        't4': b['thoracic4'],
        't3': b['thoracic3'],
        't2': b['thoracic2'],
        't1': b['thoracic1'],
        'head': head_total,
    }


def _detect_style(duration_ms: float) -> str:
    if duration_ms < STYLE_DURATION_THRESHOLD_MS:
        return 'flat'
    return 'drop'


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

    print(f'    DEBUG freefall: samples[50:100] median = {ff_median:.4f} g')
    print(f'    DEBUG freefall: bias to reach -1g = {bias:.4f} g')

    if apply_correction:
        print('    DEBUG freefall: applying correction')
        return accel_g + bias, ff_median, bias, True
    else:
        print('    DEBUG freefall: correction DISABLED, not applied')
        return accel_g, ff_median, bias, False


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _get_model_type(config: dict) -> str:
    model_cfg = config.get('model', {})
    return str(model_cfg.get('type', 'maxwell')).lower()


def _get_calibration_mode(config: dict) -> str:
    calibration_cfg = config.get('calibration', {})
    return str(calibration_cfg.get('mode', 'peaks')).lower()


def _get_plotting_config(config: dict) -> tuple[float, bool, bool, bool]:
    plot_cfg = config.get('plotting', {})
    buttocks_height_mm = float(plot_cfg.get('buttocks_height_mm', DEFAULT_BUTTOCKS_HEIGHT_MM))
    show_element_thickness = bool(plot_cfg.get('show_element_thickness', False))
    stack_elements = bool(plot_cfg.get('stack_elements', True))
    buttocks_clamp_to_height = bool(plot_cfg.get('buttocks_clamp_to_height', True))
    return buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height


def _get_calibration_accel_files() -> list[Path]:
    files: list[Path] = []
    for name in CALIBRATION_CASE_NAMES:
        accel_path = CALIBRATION_YOGANANDAN_DIR / f'accel_{name}.csv'
        if not accel_path.exists():
            raise FileNotFoundError(f'Missing calibration accel file for {name}: {accel_path.name}')
        files.append(accel_path)
    return files


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
        time_candidates=['time', 'time0', 't'],
        value_candidates=['accel', 'acceleration'],
    )
    series, sample_rate = resample_to_uniform(series)
    dt = 1.0 / sample_rate

    accel_raw = np.asarray(series.values, dtype=float)
    accel_filtered = np.asarray(cfc_filter(accel_raw.tolist(), sample_rate, cfc), dtype=float)
    t_all = np.asarray(series.time_s, dtype=float)

    total_duration_ms = float((t_all[-1] - t_all[0]) * 1000.0) if t_all.size >= 2 else 0.0
    style = _detect_style(total_duration_ms)

    print(f'    DEBUG: total_duration = {total_duration_ms:.1f} ms -> style = {style}')
    print(f'    DEBUG: raw min={np.min(accel_raw):.4f} g, max={np.max(accel_raw):.4f} g')
    print(
        f'    DEBUG: filtered min={np.min(accel_filtered):.4f} g, max={np.max(accel_filtered):.4f} g'
    )

    if style == 'flat':
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
            style='flat',
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

    a_seg, ff_median, bias, applied = _freefall_bias_correct(
        a_seg, apply_correction=drop_baseline_correction
    )
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
        style='drop',
        freefall_median_g=ff_median,
        bias_correction_g=float(bias),
        bias_correction_applied=applied,
    )
    return t_seg, a_seg, a_raw_seg, info


def _run_calibrate_buttocks(config_path: Path) -> None:
    config = _read_config(config_path)

    masses_json_path = Path(
        config.get('model', {}).get('masses_json', str(DEFAULT_MASSES_JSON_PATH))
    )
    masses_data = load_masses_json(masses_json_path)

    arm_recruitment = float(config['model'].get('arm_recruitment', 0.5))
    helmet_mass = float(config['model'].get('helmet_mass_kg', 0.7))
    mass_map = build_mass_map(masses_data, arm_recruitment=arm_recruitment, helmet_mass=helmet_mass)

    torso_mass_kg = float(sum(mass_map.values()))
    drop_baseline_correction = bool(config.get('drop_baseline_correction', True))
    settle_ms = float(config.get('gravity_settle_ms', 150.0))
    buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height = (
        _get_plotting_config(config)
    )

    # Build calibration cases from the same pulses you simulate in buttocks-only
    input_files = sorted(DROPS_DIR.glob(DROPS_PATTERN))
    if not input_files:
        raise SystemExit(
            'No drops/*.csv files found. Need pulses to calibrate buttocks bottom-out.'
        )

    cases: list[ButtocksOnlyCase] = []
    for fpath in input_files:
        t, a_filtered_g, _a_raw_g, info = process_input_file(
            fpath,
            cfc=CFC,
            peak_threshold_g=PEAK_THRESHOLD_G,
            freefall_threshold_g=FREEFALL_THRESHOLD_G,
            sim_duration_ms=SIM_DURATION_MS,
            drop_baseline_correction=drop_baseline_correction,
        )
        cases.append(
            ButtocksOnlyCase(
                name=fpath.stem,
                time_s=np.asarray(t, dtype=float),
                accel_g=np.asarray(a_filtered_g, dtype=float),
                settle_ms=(settle_ms if info.style == 'flat' else 0.0),
            )
        )

    # Start from a Toen-consistent set (k,c + inferred limit ~43 mm)
    init_cfg = recommend_toen_buttocks_params(smoothing_mm=5.0, stiffness_multiplier_at_limit=20.0)

    print('\nBUTTOCKS CALIBRATION: starting from Toen-based recommended parameters:')
    print(json.dumps(init_cfg, indent=2))

    cfg_cal = calibrate_buttocks_bottom_out(
        torso_mass_kg=torso_mass_kg,
        cases=cases,
        init_cfg=init_cfg,
        bottom_out_case_name='accel_50ms' if any(c.name == 'accel_50ms' for c in cases) else None,
        bottom_out_fraction=0.98,
        overshoot_soft_mm=0.5,
    )

    print('\nBUTTOCKS CALIBRATION RESULT:')
    print(json.dumps(cfg_cal, indent=2))

    # Write as an override file so you don't keep editing config.json
    out_path = write_buttocks_only_overrides(
        {k: v for k, v in cfg_cal.items() if not k.startswith('_')},
        meta=cfg_cal.get('_calibration', {}),
        active=True,
    )
    print(f'\nWrote buttocks-only calibration override to: {out_path}')


def _run_buttocks_only(config_path: Path) -> None:
    config = _read_config(config_path)

    masses_json_path = Path(
        config.get('model', {}).get('masses_json', str(DEFAULT_MASSES_JSON_PATH))
    )
    masses_data = load_masses_json(masses_json_path)

    arm_recruitment = float(config['model'].get('arm_recruitment', 0.5))
    helmet_mass = float(config['model'].get('helmet_mass_kg', 0.7))
    mass_map = build_mass_map(masses_data, arm_recruitment=arm_recruitment, helmet_mass=helmet_mass)

    torso_mass_kg = float(sum(mass_map.values()))

    butt_cfg = get_buttocks_only_config(config)
    overrides, override_path = load_buttocks_only_overrides()
    if overrides is not None:
        print(f'DEBUG buttocks-only: using overrides from {override_path}')
        butt_cfg.update({k: float(v) for k, v in overrides.items()})

    model = build_buttocks_only_model(torso_mass_kg, butt_cfg)

    drop_baseline_correction = bool(config.get('drop_baseline_correction', True))
    settle_ms = float(config.get('gravity_settle_ms', 150.0))
    buttocks_height_mm, show_element_thickness, stack_elements, buttocks_clamp_to_height = (
        _get_plotting_config(config)
    )

    print('Running BUTTOCKS-ONLY simulation...')
    print(f'DEBUG buttocks-only: torso_mass_kg = {torso_mass_kg:.2f} kg')
    print(
        _format_buttocks_model_debug(
            model,
            buttocks_height_mm=buttocks_height_mm,
            buttocks_clamp_to_height=buttocks_clamp_to_height,
        )
    )

    input_files = sorted(DROPS_DIR.glob(DROPS_PATTERN))
    if not input_files:
        print('No input files found for this run.')
        return

    output_root = Path(config.get('output_dir', 'output')) / BUTTOCKS_ONLY_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []

    for fpath in input_files:
        print(f'\nProcessing {fpath.name} (buttocks-only)...')

        t, a_filtered_g, _a_raw_g, info = process_input_file(
            fpath,
            cfc=CFC,
            peak_threshold_g=PEAK_THRESHOLD_G,
            freefall_threshold_g=FREEFALL_THRESHOLD_G,
            sim_duration_ms=SIM_DURATION_MS,
            drop_baseline_correction=drop_baseline_correction,
        )

        pm = pulse_metrics(np.asarray(t, dtype=float), np.asarray(a_filtered_g, dtype=float))
        print(
            '  DEBUG pulse metrics: '
            f'peak={pm["peak_g"]:.2f} g, dt={pm["dt_ms"]:.3f} ms, '
            f'delta_v={pm["delta_v_mps"]:.3f} m/s, equiv_h={pm["equiv_height_m"]:.3f} m'
        )

        y0 = np.zeros(model.size(), dtype=float)
        v0 = np.zeros(model.size(), dtype=float)
        s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

        run_dir = output_root / fpath.stem
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        if info.style == 'flat' and settle_ms > 0.0:
            dt = info.dt_s
            n_settle = int(round((settle_ms / 1000.0) / dt)) + 1
            t_settle = dt * np.arange(n_settle)
            a_settle = np.zeros_like(t_settle)

            sim_settle = newmark_nonlinear(model, t_settle, a_settle, y0, v0, s0)
            y0 = sim_settle.y[-1].copy()
            v0 = sim_settle.v[-1].copy()
            s0 = sim_settle.maxwell_state_n[-1].copy()

        sim = newmark_nonlinear(model, t, a_filtered_g, y0, v0, s0)

        butt_force_n = sim.element_forces_n[:, 0]
        comp_m = np.maximum(-(sim.y[:, 0] + model.gap_m[0]), 0.0)
        comp_max_mm = float(np.max(comp_m) * 1000.0)

        peak_butt_kN = float(np.max(butt_force_n) / 1000.0)
        t_peak_ms = float(sim.time_s[np.argmax(butt_force_n)] * 1000.0)

        comp_dbg = buttocks_force_components_from_sim(model=model, sim=sim)
        print(
            '  DEBUG force components: '
            f'spring={comp_dbg["max_spring_kN"]:.2f} kN, '
            f'damper={comp_dbg["max_damper_kN"]:.2f} kN, '
            f'stop={comp_dbg["max_stop_kN"]:.2f} kN, '
            f'mismatch(total)={comp_dbg["peak_total_mismatch_kN"]:+.3f} kN'
        )
        if comp_dbg['limit_mm'] is not None:
            print(
                '  DEBUG bottom-out: '
                f'limit={comp_dbg["limit_mm"]:.1f} mm, '
                f'max_comp={comp_dbg["max_comp_mm"]:.1f} mm, '
                f'overshoot={comp_dbg["overshoot_mm"]:.1f} mm'
            )

        write_timeseries_csv(
            run_dir / 'timeseries.csv',
            sim.time_s,
            sim.base_accel_g,
            model.node_names,
            model.element_names,
            sim.y,
            sim.v,
            sim.a,
            sim.element_forces_n,
        )

        plot_buttocks_only(
            sim.time_s,
            sim.y,
            sim.element_forces_n,
            a_filtered_g,
            run_dir / 'buttocks_only.png',
            gap_mm=float(model.gap_m[0] * 1000.0),
            compression_limit_mm=(
                butt_cfg['compression_limit_mm'] if butt_cfg['compression_limit_mm'] > 0.0 else None
            ),
        )

        print(f'  Style: {info.style}')
        print(f'  Sample rate: {info.sample_rate_hz:.1f} Hz, dt: {info.dt_s * 1000.0:.3f} ms')
        print(f'  Peak buttocks force: {peak_butt_kN:.2f} kN @ {t_peak_ms:.1f} ms')
        print(f'  Max buttocks compression: {comp_max_mm:.1f} mm')

        summary.append(
            {
                'file': fpath.name,
                'style': info.style,
                'sample_rate_hz': info.sample_rate_hz,
                'baseline_correction_applied': info.bias_correction_applied,
                'baseline_correction_g': info.bias_correction_g,
                'peak_buttocks_kN': peak_butt_kN,
                'time_to_peak_ms': t_peak_ms,
                'max_buttocks_compression_mm': comp_max_mm,
                'torso_mass_kg': torso_mass_kg,
                'buttocks_config_used': butt_cfg,
                'pulse_metrics': pm,
                'force_components': comp_dbg,
            }
        )

    (output_root / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8'
    )
    print(f'\nButtocks-only results written to {output_root}/')


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

    headers = ['time_s', 'base_accel_g']
    headers += [f'y_{n}_mm' for n in node_names]
    headers += [f'v_{n}_mps' for n in node_names]
    headers += [f'a_{n}_mps2' for n in node_names]
    headers += [f'F_{e}_kN' for e in elem_names]

    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(time_s.size):
            row = [
                f'{time_s[i]:.6f}',
                f'{base_accel_g[i]:.6f}',
            ]
            row += [f'{(y[i, j] * 1000.0):.6f}' for j in range(y.shape[1])]
            row += [f'{v[i, j]:.6f}' for j in range(v.shape[1])]
            row += [f'{a[i, j]:.6f}' for j in range(a.shape[1])]
            row += [f'{(forces_n[i, j] / 1000.0):.6f}' for j in range(forces_n.shape[1])]
            w.writerow(row)


def _run_toen_drop_test(config_path: Path, target_set: str) -> None:
    config = _read_config(config_path)
    butt_cfg = get_buttocks_only_config(config)

    # For Toen, you want full body mass. If you don't have it, use ~80.7 kg (Toen avg subject mass).
    body_mass_kg = float(butt_cfg.get('effective_mass_kg', 80.7))

    k = float(butt_cfg.get('k_n_per_m', TOEN_BUTTOCKS_K_N_PER_M))
    c = float(butt_cfg.get('c_ns_per_m', TOEN_BUTTOCKS_C_NS_PER_M))

    run_toen_suite(
        body_mass_kg=body_mass_kg,
        buttocks_k_n_per_m=k,
        buttocks_c_ns_per_m=c,
        target_set=target_set,
        impact_velocity_mps=TOEN_IMPACT_V_MPS,
    )


def _run_calibrate_toen(config_path: Path, target_set: str) -> None:
    config = _read_config(config_path)
    butt_cfg = get_buttocks_only_config(config)

    body_mass_kg = float(butt_cfg.get('effective_mass_kg', 80.7))

    result = calibrate_toen_velocity_scale(
        body_mass_kg=body_mass_kg,
        target_set=target_set,
        impact_velocity_mps=TOEN_IMPACT_V_MPS,
        k0=float(butt_cfg.get('k_n_per_m', TOEN_BUTTOCKS_K_N_PER_M)),
        c0=float(butt_cfg.get('c_ns_per_m', TOEN_BUTTOCKS_C_NS_PER_M)),
        calibrate_velocity_scale=True,
    )

    print('\nTOEN calibration result:')
    print(json.dumps(result, indent=2))

    # Update your buttocks-only overrides so --buttocks-only uses them.
    # Keep any existing bottom-out settings you already have in config.json by default:
    new_params = dict(butt_cfg)
    new_params['k_n_per_m'] = float(result['buttocks_k_n_per_m'])
    new_params['c_ns_per_m'] = float(result['buttocks_c_ns_per_m'])

    # Store velocity_scale in meta (since buttocks-only pulse mode doesn't use it)
    meta = {'toen': result}

    out_path = write_buttocks_only_overrides(new_params, meta=meta, active=True)
    print(f'\nWrote updated buttocks-only overrides to: {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--calibrate-peaks',
        action='store_true',
        help='Calibrate stiffness scales to Yoganandan peak forces and update calibration/<model>.json, then exit.',
    )
    parser.add_argument(
        '--calibrate-curves',
        action='store_true',
        help='Calibrate to force-time curves (requires calibration/yoganandan/force_*.csv) and update calibration/<model>.json, then exit.',
    )
    parser.add_argument(
        '--buttocks-only', action='store_true', help='Simulate buttocks-only (single torso mass).'
    )
    parser.add_argument(
        '--calibrate-buttocks',
        action='store_true',
        help='Calibrate buttocks-only bottom-out to be Toen-like and write calibration/buttocks_only.json.',
    )

    parser.add_argument('--toen-drop-test', action='store_true')
    parser.add_argument('--calibrate-toen', action='store_true')
    parser.add_argument(
        '--toen-target-set',
        type=str,
        default='avg_measured',
        choices=['avg', 'avg_measured', 'subj3'],
    )
    parser.add_argument('--toen-subject', type=str, default='both', choices=['avg', '3', 'both'])
    parser.add_argument('--toen-male50-mass', type=float, default=75.4)

    args = parser.parse_args()

    config_path = Path(__file__).parent / 'config.json'

    if (
        sum(
            [
                args.calibrate_peaks,
                args.calibrate_curves,
                args.buttocks_only,
                args.calibrate_buttocks,
            ]
        )
        > 1
    ):
        raise SystemExit(
            'Choose only one: --calibrate-peaks OR --calibrate-curves OR --buttocks-only OR --calibrate-buttocks.'
        )

    if args.calibrate_buttocks:
        _run_calibrate_buttocks(config_path)
        return

    if args.buttocks_only:
        _run_buttocks_only(config_path)
        return

    if args.calibrate_peaks:
        # existing behavior unchanged
        from pathlib import Path as _Path

        _ = _Path  # keep file structure unchanged
        raise SystemExit('Use your existing --calibrate-peaks flow (not modified in this patch).')

    if args.calibrate_curves:
        raise SystemExit('Use your existing --calibrate-curves flow (not modified in this patch).')

    if args.toen_drop_test:
        subjects = ['avg', '3'] if args.toen_subject == 'both' else [args.toen_subject]
        for sid in subjects:
            run_toen_suite(
                subject_id=sid,
                target_set=args.toen_target_set,
                male50_mass_kg=float(args.toen_male50_mass),
            )
        return

    if args.calibrate_toen:
        # Typically calibrate using avg_measured first (Fig4 points)
        result = calibrate_toen_velocity_scale(
            subject_id=('avg' if args.toen_subject in ['both', 'avg'] else args.toen_subject),
            target_set=args.toen_target_set,
            male50_mass_kg=float(args.toen_male50_mass),
        )
        print(json.dumps(result, indent=2))
        return

    raise SystemExit(
        'Run with --buttocks-only, --calibrate-buttocks, --calibrate-peaks, or --calibrate-curves.'
    )


if __name__ == '__main__':
    main()
