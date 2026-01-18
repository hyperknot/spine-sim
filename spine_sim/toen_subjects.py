from __future__ import annotations

from dataclasses import dataclass


# Table 1 reference masses (50th male) from Toen 2012
# Pelvis + L4 + L5 + Sacrum + Upper body + Skin over ITs
TOEN_TABLE1_MASSES_50TH_KG = {
    'pelvis': 16.0,
    'l4': 2.5,
    'l5': 1.8,
    'sacrum': 0.7,
    'upper_body': 33.0,
    'skin': 0.01,
}

DEFAULT_50TH_MALE_TOTAL_MASS_KG = 75.4  # practical default; tweak if you prefer


def toen_torso_mass_50th_kg() -> float:
    return float(sum(TOEN_TABLE1_MASSES_50TH_KG.values()))


def toen_torso_mass_scaled_kg(
    subject_total_mass_kg: float, male50_mass_kg: float = DEFAULT_50TH_MALE_TOTAL_MASS_KG
) -> float:
    # Toen: scale masses by (subject_total / male50_total)
    return toen_torso_mass_50th_kg() * float(subject_total_mass_kg) / float(male50_mass_kg)


@dataclass(frozen=True)
class ToenSubject:
    subject_id: str
    height_cm: float
    total_mass_kg: float
    buttocks_k_n_per_m: float | None
    buttocks_c_ns_per_m: float | None


# Table 2 (Toen 2012)
TOEN_SUBJECTS: dict[str, ToenSubject] = {
    'avg': ToenSubject(
        subject_id='avg',
        height_cm=177.2,
        total_mass_kg=80.7,
        buttocks_k_n_per_m=180_500.0,
        buttocks_c_ns_per_m=3_130.0,
    ),
    '3': ToenSubject(
        subject_id='3',
        height_cm=183.0,
        total_mass_kg=92.3,
        buttocks_k_n_per_m=None,
        buttocks_c_ns_per_m=None,
    ),
}

# Across-subject averages for missing k/c (Toen used mean values when missing)
TOEN_BUTTOCKS_K_MEAN_N_PER_M = 180_500.0
TOEN_BUTTOCKS_C_MEAN_NS_PER_M = 3_130.0


def subject_buttocks_kc(subject_id: str) -> tuple[float, float]:
    s = TOEN_SUBJECTS[subject_id]
    k = s.buttocks_k_n_per_m if s.buttocks_k_n_per_m is not None else TOEN_BUTTOCKS_K_MEAN_N_PER_M
    c = (
        s.buttocks_c_ns_per_m
        if s.buttocks_c_ns_per_m is not None
        else TOEN_BUTTOCKS_C_MEAN_NS_PER_M
    )
    return float(k), float(c)
