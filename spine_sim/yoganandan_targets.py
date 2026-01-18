from __future__ import annotations

"""
Hard-coded Yoganandan 2021 targets used for calibration.

NOTE:
- The uploaded paper text includes Table 2 peak thoracolumbar disc forces (kN) for the male model.
- You requested "peak compression forces for each vertebrae pair" (multi-level table). That table
  is NOT present in the uploaded text excerpts, so it cannot be hard-coded yet.

When you provide that table (e.g., L5-S1 ... T12-L1 ... T1-T2 ...), add it to
YOG2021_PEAKS_BY_LEVEL_KN below.
"""

YOG2021_T12L1_PEAKS_KN: dict[str, float] = {
    "50ms": 7.64,
    "75ms": 6.20,
    "100ms": 5.28,
    "150ms": 4.22,
    "200ms": 3.30,
}

YOG2021_PEAKS_BY_LEVEL_KN: dict[str, dict[str, float]] = {
    # Fill when you paste the per-level peak table.
}

def get_case_name_from_filename(stem: str) -> str | None:
    """
    Map filenames like 'accel_150ms.csv' -> '150ms'.

    Important: match longer tokens first so '150ms' does NOT match '50ms'.
    """
    s = stem.lower()
    for k in ["200ms", "150ms", "100ms", "75ms", "50ms"]:
        if k in s:
            return k
    return None
