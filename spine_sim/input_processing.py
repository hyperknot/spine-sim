"""Input processing utilities for acceleration data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spine_sim.filters import cfc_filter
from spine_sim.io import parse_csv_series, resample_to_uniform
from spine_sim.range import find_hit_range


def detect_style(duration_ms: float, threshold_ms: float) -> str:
    """Detect input style based on duration."""
    return "flat" if duration_ms < threshold_ms else "drop"


def freefall_bias_correct(accel_g: np.ndarray, apply: bool) -> tuple[np.ndarray, bool, float]:
    """Apply freefall bias correction to acceleration data.

    Args:
        accel_g: Acceleration in g.
        apply: Whether to apply the correction.

    Returns:
        Tuple of (corrected_accel, was_applied, bias_value).
    """
    samples = accel_g[50:100] if len(accel_g) >= 100 else accel_g[50:] if len(accel_g) > 50 else accel_g
    if samples.size == 0:
        return accel_g, False, 0.0
    bias = -1.0 - float(np.median(samples))
    return (accel_g + bias, True, bias) if apply else (accel_g, False, bias)


def process_input(
    path: Path, cfc: float, sim_duration_ms: float, style_threshold_ms: float,
    peak_threshold_g: float, freefall_threshold_g: float, drop_baseline_correction: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Process acceleration input file for simulation.

    Args:
        path: Path to CSV file with time and acceleration columns.
        cfc: CFC filter frequency.
        sim_duration_ms: Target simulation duration in milliseconds.
        style_threshold_ms: Threshold to distinguish flat vs drop style.
        peak_threshold_g: Threshold for peak detection in g.
        freefall_threshold_g: Threshold for freefall detection in g.
        drop_baseline_correction: Whether to apply baseline correction for drops.

    Returns:
        Tuple of (time_s, accel_g, info_dict).
    """
    series = parse_csv_series(path, ["time", "time0", "t"], ["accel", "acceleration"])
    series, sample_rate = resample_to_uniform(series)
    dt = 1.0 / sample_rate

    accel_raw = np.asarray(series.values, dtype=float)
    accel_filtered = np.asarray(cfc_filter(accel_raw.tolist(), sample_rate, cfc), dtype=float)
    t_all = np.asarray(series.time_s, dtype=float)

    total_ms = float((t_all[-1] - t_all[0]) * 1000.0) if t_all.size >= 2 else 0.0
    style = detect_style(total_ms, style_threshold_ms)

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
    a_seg, applied, bias = freefall_bias_correct(a_seg, drop_baseline_correction)

    desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
    if len(t_seg) < desired_n:
        pad_n = desired_n - len(t_seg)
        t_seg = np.concatenate([t_seg, t_seg[-1] + dt * (np.arange(pad_n) + 1)])
        a_seg = np.concatenate([a_seg, -1.0 * np.ones(pad_n)])

    return t_seg, a_seg, {"style": "drop", "dt_s": dt, "sample_rate_hz": sample_rate,
                          "bias_applied": applied, "bias_g": bias}
