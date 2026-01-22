"""Mass map building for spine model (with explicit cervical nodes)."""

from __future__ import annotations


def build_mass_map(
    masses: dict,
    arm_recruitment: float,
    helmet_mass: float,
    cervical_vertebra_mass_kg: float,
) -> dict:
    """
    Build mass map from OpenSim body masses.

    Cervical vertebra masses are not present in the OpenSim file used here, so we approximate:
      - assign a fixed small mass to each C1..C7 (configurable),
      - subtract the total cervical mass from head_neck lump mass.

    This preserves total mass while enabling explicit cervical nodes.

    Returns a dict mapping node names to masses in kg.
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
    ) * arm_recruitment

    head_total = b['head_neck'] + helmet_mass + arm_mass

    cerv_m = float(cervical_vertebra_mass_kg)
    if cerv_m < 0.0:
        raise ValueError('model.cervical_vertebra_mass_kg must be >= 0.')

    total_cerv = 7.0 * cerv_m
    head_mass = head_total - total_cerv
    if head_mass <= 0.1:
        raise ValueError(
            'Cervical mass allocation leaves too little head mass. '
            f'head_total={head_total:.3f} kg, total_cerv={total_cerv:.3f} kg.'
        )

    m = {
        'pelvis': b['pelvis'],
        'L5': b['lumbar5'],
        'L4': b['lumbar4'],
        'L3': b['lumbar3'],
        'L2': b['lumbar2'],
        'L1': b['lumbar1'],
        'T12': b['thoracic12'],
        'T11': b['thoracic11'],
        'T10': b['thoracic10'],
        'T9': b['thoracic9'],
        'T8': b['thoracic8'],
        'T7': b['thoracic7'],
        'T6': b['thoracic6'],
        'T5': b['thoracic5'],
        'T4': b['thoracic4'],
        'T3': b['thoracic3'],
        'T2': b['thoracic2'],
        'T1': b['thoracic1'],
        'C7': cerv_m,
        'C6': cerv_m,
        'C5': cerv_m,
        'C4': cerv_m,
        'C3': cerv_m,
        'C2': cerv_m,
        'C1': cerv_m,
        'HEAD': head_mass,
    }

    return m
