from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TimeSeries:
    time_s: list[float]
    values: list[float]


def _detect_delimiter(header_line: str) -> str:
    semicolons = header_line.count(";")
    commas = header_line.count(",")
    return ";" if semicolons >= commas and semicolons > 0 else ","


def _parse_number(s: str) -> float:
    return float(s.strip().replace(",", "."))


def _detect_time_is_ms(max_time: float) -> bool:
    return max_time > 100


def _find_header_idx(lines: list[str], required: Iterable[str]) -> int:
    required = [r.lower() for r in required]
    for i, line in enumerate(lines):
        low = line.lower()
        if all(r in low for r in required):
            return i
    return -1


def _find_col(headers: list[str], candidates: Iterable[str]) -> int:
    candidates = [c.lower() for c in candidates]
    for i, h in enumerate(headers):
        for c in candidates:
            if c in h:
                return i
    return -1


def parse_csv_series(
    path: Path,
    time_candidates: Iterable[str],
    value_candidates: Iterable[str],
) -> TimeSeries:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]

    header_idx = _find_header_idx(
        lines, required=["time", "accel"] if "accel" in value_candidates else ["time"]
    )
    if header_idx == -1:
        header_idx = 0

    header_line = lines[header_idx]
    delimiter = _detect_delimiter(header_line)

    headers = [h.strip().lower() for h in header_line.split(delimiter)]

    col_time = _find_col(headers, time_candidates)
    col_val = _find_col(headers, value_candidates)

    if col_time == -1 or col_val == -1:
        raise ValueError(f"Missing columns in {path.name}: time={col_time}, value={col_val}")

    raw_rows: list[tuple[float, float]] = []
    for line in lines[header_idx + 1 :]:
        parts = line.split(delimiter)
        if len(parts) <= max(col_time, col_val):
            continue
        try:
            t = _parse_number(parts[col_time])
            v = _parse_number(parts[col_val])
        except ValueError:
            continue
        raw_rows.append((t, v))

    if not raw_rows:
        raise ValueError(f"No valid rows found in {path.name}")

    max_time = max(r[0] for r in raw_rows)
    is_ms = _detect_time_is_ms(max_time)

    time_s = [(t / 1000.0) if is_ms else t for t, _ in raw_rows]
    values = [v for _, v in raw_rows]

    return TimeSeries(time_s=time_s, values=values)


def resample_to_uniform(series: TimeSeries) -> tuple[TimeSeries, float]:
    t = np.asarray(series.time_s, dtype=float)
    x = np.asarray(series.values, dtype=float)

    if t.size < 2:
        return series, 1000.0

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return TimeSeries(time_s=[t[0]], values=[x[0]]), 1000.0

    median_dt = float(np.median(dt))
    sample_rate = 1.0 / median_dt

    t0 = float(t[0])
    t1 = float(t[-1])
    n = int(round((t1 - t0) / median_dt)) + 1
    t_uniform = t0 + np.arange(n) * median_dt
    x_uniform = np.interp(t_uniform, t, x)

    return TimeSeries(time_s=t_uniform.tolist(), values=x_uniform.tolist()), sample_rate
