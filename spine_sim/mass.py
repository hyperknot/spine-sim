"""Mass map building for spine model."""

from __future__ import annotations


def build_mass_map(masses: dict, arm_recruitment: float, helmet_mass: float) -> dict:
    """Build mass map from OpenSim body masses.

    Args:
        masses: Dict with 'bodies' key containing OpenSim body masses.
        arm_recruitment: Fraction of arm mass to include (0-1).
        helmet_mass: Additional mass to add to head (kg).

    Returns:
        Dict mapping spine node names to masses in kg.
    """
    b = masses["bodies"]
    arm_mass = (
        b["humerus_R"] + b["humerus_L"] + b["ulna_R"] + b["ulna_L"] +
        b["radius_R"] + b["radius_L"] + b["hand_R"] + b["hand_L"]
    ) * arm_recruitment
    return {
        "pelvis": b["pelvis"], "l5": b["lumbar5"], "l4": b["lumbar4"],
        "l3": b["lumbar3"], "l2": b["lumbar2"], "l1": b["lumbar1"],
        "t12": b["thoracic12"], "t11": b["thoracic11"], "t10": b["thoracic10"],
        "t9": b["thoracic9"], "t8": b["thoracic8"], "t7": b["thoracic7"],
        "t6": b["thoracic6"], "t5": b["thoracic5"], "t4": b["thoracic4"],
        "t3": b["thoracic3"], "t2": b["thoracic2"], "t1": b["thoracic1"],
        "head": b["head_neck"] + helmet_mass + arm_mass,
    }
