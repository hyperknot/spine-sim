from __future__ import annotations

import numpy as np


def build_spine_elements(
    mass_map: dict,
    c_base: float,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Build the common spine element definitions (names, masses, k, c).

    Shared by all model paths (maxwell, zwt, etc.).
    """
    node_names = [
        'pelvis',
        'L5',
        'L4',
        'L3',
        'L2',
        'L1',
        'T12',
        'T11',
        'T10',
        'T9',
        'T8',
        'T7',
        'T6',
        'T5',
        'T4',
        'T3',
        'T2',
        'T1',
        'HEAD',
    ]

    masses = np.array(
        [
            mass_map['pelvis'],
            mass_map['l5'],
            mass_map['l4'],
            mass_map['l3'],
            mass_map['l2'],
            mass_map['l1'],
            mass_map['t12'],
            mass_map['t11'],
            mass_map['t10'],
            mass_map['t9'],
            mass_map['t8'],
            mass_map['t7'],
            mass_map['t6'],
            mass_map['t5'],
            mass_map['t4'],
            mass_map['t3'],
            mass_map['t2'],
            mass_map['t1'],
            mass_map['head'],
        ],
        dtype=float,
    )

    # Raj 2019 axial stiffnesses (N/m)
    k = {
        'head-c1': 0.55e6,
        'c1-c2': 0.3e6,
        'c2-c3': 0.7e6,
        'c3-c4': 0.76e6,
        'c4-c5': 0.794e6,
        'c5-c6': 0.967e6,
        'c6-c7': 1.014e6,
        'c7-t1': 1.334e6,
        't1-t2': 0.7e6,
        't2-t3': 1.2e6,
        't3-t4': 1.5e6,
        't4-t5': 2.1e6,
        't5-t6': 1.9e6,
        't6-t7': 1.8e6,
        't7-t8': 1.5e6,
        't8-t9': 1.5e6,
        't9-t10': 1.5e6,
        't10-t11': 1.5e6,
        't11-t12': 1.5e6,
        't12-l1': 1.8e6,
        'l1-l2': 2.13e6,
        'l2-l3': 2.0e6,
        'l3-l4': 2.0e6,
        'l4-l5': 1.87e6,
        'l5-s1': 1.47e6,
    }

    cerv_keys = [
        'head-c1',
        'c1-c2',
        'c2-c3',
        'c3-c4',
        'c4-c5',
        'c5-c6',
        'c6-c7',
        'c7-t1',
    ]
    k_cerv_eq = 1.0 / sum(1.0 / k[key] for key in cerv_keys)

    def c_disc(name: str) -> float:
        if name in [
            't10-t11',
            't11-t12',
            't12-l1',
            'l1-l2',
            'l2-l3',
            'l3-l4',
            'l4-l5',
            'l5-s1',
        ]:
            return 3.0 * c_base
        return c_base

    element_names = [
        'buttocks',
        'L5-S1',
        'L4-L5',
        'L3-L4',
        'L2-L3',
        'L1-L2',
        'T12-L1',
        'T11-T12',
        'T10-T11',
        'T9-T10',
        'T8-T9',
        'T7-T8',
        'T6-T7',
        'T5-T6',
        'T4-T5',
        'T3-T4',
        'T2-T3',
        'T1-T2',
        'T1-HEAD',
    ]

    k_elem = np.array(
        [
            8.8425e4,  # buttocks
            k['l5-s1'],
            k['l4-l5'],
            k['l3-l4'],
            k['l2-l3'],
            k['l1-l2'],
            k['t12-l1'],
            k['t11-t12'],
            k['t10-t11'],
            k['t9-t10'],
            k['t8-t9'],
            k['t7-t8'],
            k['t6-t7'],
            k['t5-t6'],
            k['t4-t5'],
            k['t3-t4'],
            k['t2-t3'],
            k['t1-t2'],
            k_cerv_eq,
        ],
        dtype=float,
    )

    c_elem = np.array(
        [
            1700.0,  # buttocks
            c_disc('l5-s1'),
            c_disc('l4-l5'),
            c_disc('l3-l4'),
            c_disc('l2-l3'),
            c_disc('l1-l2'),
            c_disc('t12-l1'),
            c_disc('t11-t12'),
            c_disc('t10-t11'),
            c_disc('t9-t10'),
            c_disc('t8-t9'),
            c_disc('t7-t8'),
            c_disc('t6-t7'),
            c_disc('t5-t6'),
            c_disc('t4-t5'),
            c_disc('t3-t4'),
            c_disc('t2-t3'),
            c_disc('t1-t2'),
            c_base / len(cerv_keys),
        ],
        dtype=float,
    )

    return node_names, masses, element_names, k_elem, c_elem
