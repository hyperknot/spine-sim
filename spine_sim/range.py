from __future__ import annotations

from dataclasses import dataclass

# Default thresholds for drop impact detection
DEFAULT_PEAK_THRESHOLD_G = 5.0
DEFAULT_FREEFALL_THRESHOLD_G = -0.85


@dataclass
class HitRange:
    start_idx: int
    end_idx: int


def find_hit_range(
    accel_g: list[float],
    peak_threshold_g: float = DEFAULT_PEAK_THRESHOLD_G,
    freefall_threshold_g: float = DEFAULT_FREEFALL_THRESHOLD_G,
) -> HitRange | None:
    """
    Find the first impact event in drop-style acceleration data.

    Searches for first peak above threshold, then expands left/right
    to freefall boundaries.
    """
    n = len(accel_g)
    if n == 0:
        return None

    # Find first significant peak
    peak_idx = -1
    for i in range(n):
        if accel_g[i] > peak_threshold_g:
            local_max = accel_g[i]
            local_idx = i
            j = i + 1
            while j < n and accel_g[j] > peak_threshold_g:
                if accel_g[j] > local_max:
                    local_max = accel_g[j]
                    local_idx = j
                j += 1
            peak_idx = local_idx
            break

    if peak_idx == -1:
        return None

    # Expand left to freefall
    start_idx = 0
    for i in range(peak_idx - 1, -1, -1):
        if accel_g[i] < freefall_threshold_g:
            start_idx = i
            break

    # Expand right to freefall
    end_idx = n - 1
    for i in range(peak_idx + 1, n):
        if accel_g[i] < freefall_threshold_g:
            end_idx = i
            break

    return HitRange(start_idx=start_idx, end_idx=end_idx)
