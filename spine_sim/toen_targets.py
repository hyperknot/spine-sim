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

# Your measured points from Fig 4 (simulation, ground peaks), N -> kN
TOEN_GROUND_PEAKS_KN_AVG_MEASURED_FIG4: dict[str, float] = {
    'soft_59': 4917.733089579523 / 1000.0,
    'medium_67': 5118.829981718463 / 1000.0,
    'firm_95': 5758.683729433272 / 1000.0,
    'rigid_400': 7769.652650822669 / 1000.0,
}

# Your Subject 3 (Fig 3) approximate ground peaks (kN)
TOEN_GROUND_PEAKS_KN_SUBJECT3_FIG3_APPROX: dict[str, float] = {
    'soft_59': 6.0,
    'medium_67': 6.5,
    'firm_95': 7.2,
    'rigid_400': 9.6,
}
