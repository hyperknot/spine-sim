from __future__ import annotations


TOEN_FLOOR_STIFFNESS_N_PER_M: dict[str, float] = {
    'soft_59': 59_000.0,
    'medium_67': 67_000.0,
    'firm_95': 95_000.0,
    'rigid_400': 400_000.0,
}

# Paper-reported averages (Fig 4 text numbers, kN)
TOEN_GROUND_PEAKS_KN_AVG_PAPER: dict[str, float] = {
    'soft_59': 4.9,
    'medium_67': 5.1,
    'firm_95': 5.8,
    'rigid_400': 7.8,
}
