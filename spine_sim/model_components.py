from __future__ import annotations

import numpy as np
from spine_sim.model import SpineModel
from spine_sim.settings import req_float


def build_spine_model(mass_map: dict, config: dict) -> SpineModel:
    """
    Build the 1D axial chain model:
      [Base] -- buttocks -- pelvis -- L5 -- ... -- T1 -- HEAD

    Note: HEAD is OpenSim's lumped head/neck (incl. cervical) mass, optionally with helmet + recruited arms.
    """
    stiffness_scale = float(req_float(config, ['spine', 'stiffness_scale']))
    if stiffness_scale <= 0.0:
        raise ValueError('spine.stiffness_scale must be > 0.')

    disc_height_mm = float(req_float(config, ['spine', 'disc_height_mm']))
    if disc_height_mm <= 0.0:
        raise ValueError('spine.disc_height_mm must be > 0.')

    tension_k_mult = float(req_float(config, ['spine', 'tension_k_mult']))
    if tension_k_mult < 0.0:
        raise ValueError('spine.tension_k_mult must be >= 0.')

    c_disc = float(req_float(config, ['spine', 'damping_ns_per_m']))
    if c_disc < 0.0:
        raise ValueError('spine.damping_ns_per_m must be >= 0.')

    # Nodes bottom-to-top
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

    masses = np.array([float(mass_map[n]) for n in node_names], dtype=float)

    # Baseline axial stiffnesses (N/m): Kitazaki/Griffin distribution.
    # Cervical vertebrae are not explicit here, so the top connection is HEAD-T1.
    k = {
        'HEAD-T1': 1.334e6,
        'T1-T2': 0.7e6,
        'T2-T3': 1.2e6,
        'T3-T4': 1.5e6,
        'T4-T5': 2.1e6,
        'T5-T6': 1.9e6,
        'T6-T7': 1.8e6,
        'T7-T8': 1.5e6,
        'T8-T9': 1.5e6,
        'T9-T10': 1.5e6,
        'T10-T11': 1.5e6,
        'T11-T12': 1.5e6,
        'T12-L1': 1.8e6,
        'L1-L2': 2.13e6,
        'L2-L3': 2.0e6,
        'L3-L4': 2.0e6,
        'L4-L5': 1.87e6,
        'L5-S1': 1.47e6,
    }

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
        'HEAD-T1',
    ]

    # Stiffness per element (element 0 is buttocks, filled from config)
    k_elem = np.zeros(len(element_names), dtype=float)
    k_elem[0] = 1.0  # placeholder; overwritten by buttocks config in SpineModel
    for i, ename in enumerate(element_names[1:], start=1):
        if ename not in k:
            raise KeyError(f'Missing baseline stiffness for element: {ename}')
        k_elem[i] = float(k[ename]) * stiffness_scale

    c_elem = np.zeros(len(element_names), dtype=float)
    c_elem[0] = 0.0  # buttocks filled from config
    c_elem[1:] = c_disc

    # Buttocks parameters (bilinear + contact), gap removed (always 0)
    k1 = float(req_float(config, ['buttock', 'k1_n_per_m']))
    c_butt = float(req_float(config, ['buttock', 'c_ns_per_m']))
    bottom_out_force_kN = float(req_float(config, ['buttock', 'bottom_out_force_kN']))
    k2 = float(req_float(config, ['buttock', 'k2_n_per_m']))

    if k1 <= 0.0:
        raise ValueError('buttock.k1_n_per_m must be > 0.')
    if k2 <= 0.0:
        raise ValueError('buttock.k2_n_per_m must be > 0.')
    if bottom_out_force_kN < 0.0:
        raise ValueError('buttock.bottom_out_force_kN must be >= 0.')
    if c_butt < 0.0:
        raise ValueError('buttock.c_ns_per_m must be >= 0.')

    return SpineModel(
        node_names=node_names,
        masses_kg=masses,
        element_names=element_names,
        k0_elem_n_per_m=k_elem,
        c_elem_ns_per_m=c_elem,
        disc_height_m=float(disc_height_mm) / 1000.0,
        tension_k_mult=tension_k_mult,
        buttocks_k1_n_per_m=k1,
        buttocks_k2_n_per_m=k2,
        buttocks_bottom_out_force_n=float(bottom_out_force_kN) * 1000.0,
        buttocks_c_ns_per_m=c_butt,
        kemper_normalize_to_eps_per_s=float(
            req_float(config, ['spine', 'kemper', 'normalize_to_eps_per_s'])
        ),
        strain_rate_smoothing_tau_s=float(
            req_float(config, ['spine', 'kemper', 'strain_rate_smoothing_tau_ms'])
        )
        / 1000.0,
        warn_over_eps_per_s=float(req_float(config, ['spine', 'kemper', 'warn_over_eps_per_s'])),
    )
