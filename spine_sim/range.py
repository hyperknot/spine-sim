from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HitRange:
    start_idx: int
    end_idx: int


def find_first_hit_range(
    accel_g: list[float],
    peak_threshold_g: float = 5.0,
    free_fall_threshold_g: float = -0.85,
) -> HitRange | None:
    n = len(accel_g)
    if n == 0:
        return None

    peak_idx = -1
    for i in range(n):
        if accel_g[i] > peak_threshold_g:
            # Find local max in this peak region
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

    start_idx = 0
    for i in range(peak_idx - 1, -1, -1):
        if accel_g[i] < free_fall_threshold_g:
            start_idx = i
            break

    end_idx = n - 1
    for i in range(peak_idx + 1, n):
        if accel_g[i] < free_fall_threshold_g:
            end_idx = i
            break

    return HitRange(start_idx=start_idx, end_idx=end_idx)
