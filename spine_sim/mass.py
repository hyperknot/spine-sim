"""Mass map building for spine model (simplified: head+neck+cervical are lumped)."""

from __future__ import annotations

from spine_sim.model import G0


def build_mass_map(
    masses: dict,
    arm_recruitment: float,
    helmet_mass: float,
    echo=print,
) -> dict:
    """
    Build mass map from OpenSim body masses.

    Neck simplification:
      - OpenSim provides 'head_neck' as a combined lump (head + neck soft tissue + cervical vertebrae).
      - We keep this as a single modeled node mass: HEAD.
      - We do NOT create separate cervical vertebra nodes or masses.

    Arms:
      - arm_recruitment is interpreted as the fraction of total arm mass whose
        weight/inertia loads the spine (vs being supported externally by thighs/armrests/table).
      - Recruited arm mass is added to T1 (shoulder/upper thorax region), NOT to HEAD.

    Helmet:
      - helmet_mass is added to HEAD.
    """
    b = masses['bodies']

    # Total arm mass (both sides)
    arm_mass_total = (
        b['humerus_R']
        + b['humerus_L']
        + b['ulna_R']
        + b['ulna_L']
        + b['radius_R']
        + b['radius_L']
        + b['hand_R']
        + b['hand_L']
    )

    arm_mass_to_spine = float(arm_mass_total) * float(arm_recruitment)

    # HEAD: head/neck/cervical lump + optional helmet only (no arms here)
    head_total = float(b['head_neck']) + float(helmet_mass)

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
        # Add recruited arms at T1 (approx shoulder attachment level)
        'T1': float(b['thoracic1']) + float(arm_mass_to_spine),
        'HEAD': head_total,
    }

    # Debug: print mass at each vertebrae level
    echo('Mass map (modeled vertebrae):')
    echo('  node     mass_kg   weight_N')
    for name, mass_kg in mass_map.items():
        w_n = float(mass_kg) * G0
        echo(f'  {name:6s}  {mass_kg:7.3f}  {w_n:8.1f}')
    total_mass_kg = float(sum(mass_map.values()))
    echo(f'  {"TOTAL":6s}  {total_mass_kg:7.3f}  {total_mass_kg * G0:8.1f}')
    echo(f'  arms_total={arm_mass_total:.3f} kg, arms_to_spine={arm_mass_to_spine:.3f} kg')
    echo(f'  (arm_recruitment={arm_recruitment:.1%}, helmet={helmet_mass:.2f} kg)')

    return mass_map
