"""Mass map building for spine model (simplified: head+neck+cervical are lumped)."""

from __future__ import annotations


def build_mass_map(
    masses: dict,
    arm_recruitment: float,
    helmet_mass: float,
    echo=print,
) -> dict:
    """
    Build mass map from OpenSim body masses.

    This simplified model uses OpenSim's provided head/neck lump:
      - OpenSim file provides 'head_neck' as a combined lump already.
      - We do NOT create separate cervical vertebra nodes or masses.

    Arms are optionally "recruited" into the head/neck effective mass (legacy behavior).
    """
    b = masses['bodies']

    arm_mass = (
        b['humerus_R']
        + b['humerus_L']
        + b['ulna_R']
        + b['ulna_L']
        + b['radius_R']
        + b['radius_L']
        + b['hand_R']
        + b['hand_L']
    ) * float(arm_recruitment)

    head_total = float(b['head_neck']) + float(helmet_mass) + float(arm_mass)

    mass_map = {
        'pelvis': float(b['pelvis']),
        'L5': float(b['lumbar5']),
        'L4': float(b['lumbar4']),
        'L3': float(b['lumbar3']),
        'L2': float(b['lumbar2']),
        'L1': float(b['lumbar1']),
        'T12': float(b['thoracic12']),
        'T11': float(b['thoracic11']),
        'T10': float(b['thoracic10']),
        'T9': float(b['thoracic9']),
        'T8': float(b['thoracic8']),
        'T7': float(b['thoracic7']),
        'T6': float(b['thoracic6']),
        'T5': float(b['thoracic5']),
        'T4': float(b['thoracic4']),
        'T3': float(b['thoracic3']),
        'T2': float(b['thoracic2']),
        'T1': float(b['thoracic1']),
        'HEAD': head_total,
    }

    # Debug: print mass at each vertebrae level
    echo('Mass map (modeled vertebrae):')
    for name, mass_kg in mass_map.items():
        echo(f'  {name:6s}: {mass_kg:7.3f} kg')
    total_mass_kg = sum(mass_map.values())
    echo(f'  {"TOTAL":6s}: {total_mass_kg:7.3f} kg')
    echo(f'  (arm_recruitment={arm_recruitment:.1%}, helmet={helmet_mass:.2f} kg)')

    return mass_map
