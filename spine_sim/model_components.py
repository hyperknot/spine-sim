from __future__ import annotations

import numpy as np
from spine_sim.model import SpineModel
from spine_sim.settings import req_float


def _series_equivalent_stiffness(k_list_n_per_m: list[float]) -> float:
    """
    Springs in series:
      1/k_eq = sum(1/k_i).
    """
    if not k_list_n_per_m:
        raise ValueError('k_list_n_per_m must be non-empty.')
    inv = 0.0
    for k in k_list_n_per_m:
        kk = float(k)
        if kk <= 0.0:
            raise ValueError('All stiffness values must be > 0 for series equivalent.')
        inv += 1.0 / kk
    return 1.0 / inv


def _series_equivalent_damping(c_list_ns_per_m: list[float]) -> float:
    """
    Dashpots in series:
      1/c_eq = sum(1/c_i).

    Note: A Kelvin-Voigt stack in series is not perfectly reducible to a single Kelvin-Voigt
    element across all frequencies. Here we use the standard "dashpots in series" reduction
    as a physically reasonable approximation consistent with this simplified 1D model.
    """
    if not c_list_ns_per_m:
        raise ValueError('c_list_ns_per_m must be non-empty.')
    inv = 0.0
    for c in c_list_ns_per_m:
        cc = float(c)
        if cc <= 0.0:
            raise ValueError('All damping values must be > 0 for series equivalent.')
        inv += 1.0 / cc
    return 1.0 / inv


def build_spine_model(
    mass_map: dict,
    config: dict,
    *,
    buttocks_mode: str,
    buttocks_profile: str,
) -> SpineModel:
    """
    Build the 1D axial chain model:

      [Base] -- buttocks -- pelvis -- L5 -- ... -- T1 -- HEAD

    Buttocks:
      - mode is supplied at runtime (CLI), not in config.json.
      - profile is supplied at runtime (CLI), not in config.json.
    """
    buttocks_mode = str(buttocks_mode).strip().lower()
    if buttocks_mode not in ('localized', 'uniform'):
        raise ValueError('buttocks_mode must be "localized" or "uniform".')

    buttocks_profile = str(buttocks_profile).strip()
    if buttocks_profile not in ('sporty', 'avg', 'soft'):
        raise ValueError('buttocks_profile must be sporty/avg/soft.')

    disc_height_mm = float(req_float(config, ['spine', 'disc_height_mm']))
    if disc_height_mm <= 0.0:
        raise ValueError('spine.disc_height_mm must be > 0.')

    cervical_disc_height_single_mm = float(
        req_float(config, ['spine', 'cervical_disc_height_single_mm'])
    )
    if cervical_disc_height_single_mm <= 0.0:
        raise ValueError('spine.cervical_disc_height_single_mm must be > 0.')

    tension_k_mult = float(req_float(config, ['spine', 'tension_k_mult']))
    if tension_k_mult < 0.0:
        raise ValueError('spine.tension_k_mult must be >= 0.')

    c_disc = float(req_float(config, ['spine', 'damping_ns_per_m']))
    if c_disc < 0.0:
        raise ValueError('spine.damping_ns_per_m must be >= 0.')

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

    k_kitazaki_thoracolumbar = {
        'T1-T2': 0.70e6,
        'T2-T3': 1.20e6,
        'T3-T4': 1.50e6,
        'T4-T5': 2.10e6,
        'T5-T6': 1.90e6,
        'T6-T7': 1.80e6,
        'T7-T8': 1.50e6,
        'T8-T9': 1.50e6,
        'T9-T10': 1.50e6,
        'T10-T11': 1.50e6,
        'T11-T12': 1.50e6,
        'T12-L1': 1.80e6,
        'L1-L2': 2.13e6,
        'L2-L3': 2.00e6,
        'L3-L4': 2.00e6,
        'L4-L5': 1.87e6,
        'L5-S1': 1.47e6,
    }

    # Kitazaki & Griffin (1997) Table 4 cervical chain (N/m): Head-C1 ... C7-T1.
    k_kitazaki_cervical_chain = [
        0.550e6,
        0.300e6,
        0.700e6,
        0.760e6,
        0.794e6,
        0.967e6,
        1.014e6,
        1.334e6,
    ]
    neck_chain_count = len(k_kitazaki_cervical_chain)
    k_head_t1_eq = _series_equivalent_stiffness(k_kitazaki_cervical_chain)

    k_elem = np.zeros(len(element_names), dtype=float)
    k_elem[0] = 1.0  # buttocks placeholder

    for i, ename in enumerate(element_names[1:], start=1):
        if ename == 'HEAD-T1':
            k_elem[i] = float(k_head_t1_eq)
            continue
        if ename not in k_kitazaki_thoracolumbar:
            raise KeyError(f'Missing baseline stiffness for element: {ename}')
        k_elem[i] = float(k_kitazaki_thoracolumbar[ename])

    c_elem = np.zeros(len(element_names), dtype=float)
    c_elem[0] = 0.0
    c_elem[1:] = c_disc

    neck_elem_idx = element_names.index('HEAD-T1')
    if c_disc > 0.0:
        c_elem[neck_elem_idx] = _series_equivalent_damping([float(c_disc)] * neck_chain_count)
    else:
        c_elem[neck_elem_idx] = 0.0

    # Per-element effective height for strain-rate:
    disc_height_m_per_elem = np.zeros(len(element_names), dtype=float)
    disc_height_m_per_elem[0] = 0.0
    disc_height_m_per_elem[1:] = float(disc_height_mm) / 1000.0
    disc_height_m_per_elem[neck_elem_idx] = (
        float(neck_chain_count) * float(cervical_disc_height_single_mm) / 1000.0
    )

    # Buttocks profile parameters (k1/c from Van Toen; apex thickness from Sonenblum).
    apex_thickness_mm = float(
        req_float(config, ['buttock', 'profiles', buttocks_profile, 'apex_thickness_mm'])
    )
    k1 = float(req_float(config, ['buttock', 'profiles', buttocks_profile, 'k1_n_per_m']))
    c_butt = float(req_float(config, ['buttock', 'profiles', buttocks_profile, 'c_ns_per_m']))
    k2_mult = float(req_float(config, ['buttock', 'k2_mult']))

    if apex_thickness_mm <= 0.0:
        raise ValueError('buttock.profiles.<name>.apex_thickness_mm must be > 0.')
    if k1 <= 0.0:
        raise ValueError('buttock.profiles.<name>.k1_n_per_m must be > 0.')
    if c_butt < 0.0:
        raise ValueError('buttock.profiles.<name>.c_ns_per_m must be >= 0.')
    if k2_mult <= 0.0:
        raise ValueError('buttock.k2_mult must be > 0.')

    return SpineModel(
        node_names=node_names,
        masses_kg=masses,
        element_names=element_names,
        k0_elem_n_per_m=k_elem,
        c_elem_ns_per_m=c_elem,
        disc_height_m_per_elem=disc_height_m_per_elem,
        tension_k_mult=tension_k_mult,
        buttocks_mode=buttocks_mode,
        buttocks_active_profile=buttocks_profile,
        buttocks_apex_thickness_m=float(apex_thickness_mm) / 1000.0,
        buttocks_k1_n_per_m=k1,
        buttocks_k2_mult=k2_mult,
        buttocks_c_ns_per_m=c_butt,
        buttocks_x_idle_m=float('nan'),
        kemper_normalize_to_eps_per_s=float(
            req_float(config, ['spine', 'kemper', 'normalize_to_eps_per_s'])
        ),
        strain_rate_smoothing_tau_s=float(
            req_float(config, ['spine', 'kemper', 'strain_rate_smoothing_tau_ms'])
        )
        / 1000.0,
        warn_over_eps_per_s=float(req_float(config, ['spine', 'kemper', 'warn_over_eps_per_s'])),
    )
