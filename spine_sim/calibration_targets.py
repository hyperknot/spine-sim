from __future__ import annotations


"""
Calibration targets from Yoganandan 2021.

Peak thoracolumbar disc forces (kN) for the male model from Table 2.
"""

CALIBRATION_T12L1_PEAKS_KN: dict[str, float] = {
    '50ms': 7.64,
    '75ms': 6.20,
    '100ms': 5.28,
    '150ms': 4.22,
    '200ms': 3.30,
}


def get_case_name_from_filename(stem: str) -> str | None:
    """
    Map filenames like 'accel_150ms.csv' -> '150ms'.

    Match longer tokens first so '150ms' does NOT match '50ms'.
    """
    s = stem.lower()
    for k in ['200ms', '150ms', '100ms', '75ms', '50ms']:
        if k in s:
            return k
    return None
