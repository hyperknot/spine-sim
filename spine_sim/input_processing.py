"""Input processing utilities for acceleration data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from spine_sim.filters import cfc_filter
from spine_sim.io import parse_csv_series, resample_to_uniform
from spine_sim.range import find_hit_range


def detect_style(duration_ms: float, threshold_ms: float) -> str:
    """Detect input style based on duration."""
    return 'flat' if duration_ms < threshold_ms else 'drop'


def process_input(
    path: Path,
    cfc: float,
    sim_duration_ms: float,
    style_threshold_ms: float,
    peak_threshold_g: float,
    freefall_threshold_g: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Process acceleration input file for simulation.

    Returns:
        Tuple of (time_s, accel_g, info_dict).
    """
    series = parse_csv_series(path, ['time', 'time0', 't'], ['accel', 'acceleration'])
    series, sample_rate = resample_to_uniform(series)
    dt = 1.0 / sample_rate

    accel_raw = np.asarray(series.values, dtype=float)
    accel_filtered = np.asarray(cfc_filter(accel_raw.tolist(), sample_rate, cfc), dtype=float)
    t_all = np.asarray(series.time_s, dtype=float)

    total_ms = float((t_all[-1] - t_all[0]) * 1000.0) if t_all.size >= 2 else 0.0
    style = detect_style(total_ms, style_threshold_ms)

    # Store raw min/max for debug output
    raw_min_g = float(np.min(accel_raw))
    raw_max_g = float(np.max(accel_raw))

    if style == 'flat':
        t_seg, a_seg = t_all - t_all[0], accel_filtered.copy()
        desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
        if len(t_seg) < desired_n:
            pad_n = desired_n - len(t_seg)
            t_seg = np.concatenate([t_seg, t_seg[-1] + dt * (np.arange(pad_n) + 1)])
            # Pad with -1g (freefall) after pulse ends
            a_seg = np.concatenate([a_seg, -1.0 * np.ones(pad_n)])
        return (
            t_seg,
            a_seg,
            {
                'style': 'flat',
                'dt_s': dt,
                'sample_rate_hz': sample_rate,
                'duration_ms': total_ms,
                'raw_min_g': raw_min_g,
                'raw_max_g': raw_max_g,
            },
        )

    hit = find_hit_range(accel_filtered.tolist(), peak_threshold_g, freefall_threshold_g)
    start_idx = hit.start_idx if hit else 0
    end_idx = min(len(t_all) - 1, start_idx + int(round((sim_duration_ms / 1000.0) / dt)))

    t_seg = t_all[start_idx : end_idx + 1] - t_all[start_idx]
    a_seg = accel_filtered[start_idx : end_idx + 1]

    desired_n = int(round((sim_duration_ms / 1000.0) / dt)) + 1
    if len(t_seg) < desired_n:
        pad_n = desired_n - len(t_seg)
        t_seg = np.concatenate([t_seg, t_seg[-1] + dt * (np.arange(pad_n) + 1)])
        # Pad with -1g (freefall) after impact ends
        a_seg = np.concatenate([a_seg, -1.0 * np.ones(pad_n)])

    return (
        t_seg,
        a_seg,
        {
            'style': 'drop',
            'dt_s': dt,
            'sample_rate_hz': sample_rate,
            'duration_ms': total_ms,
            'raw_min_g': raw_min_g,
            'raw_max_g': raw_max_g,
        },
    )
