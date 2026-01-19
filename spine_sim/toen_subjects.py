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
    buttocks_k_n_per_m: float
    buttocks_c_ns_per_m: float


# Table 2 (Toen 2012) - simplified: only keep the avg subject.
TOEN_SUBJECTS: dict[str, ToenSubject] = {
    'avg': ToenSubject(
        subject_id='avg',
        height_cm=177.2,
        total_mass_kg=80.7,
        buttocks_k_n_per_m=180_500.0,
        buttocks_c_ns_per_m=3_130.0,
    ),
}


def subject_buttocks_kc(subject_id: str = "avg") -> tuple[float, float]:
    s = TOEN_SUBJECTS[subject_id]
    return float(s.buttocks_k_n_per_m), float(s.buttocks_c_ns_per_m)
