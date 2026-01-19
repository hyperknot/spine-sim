"""Model path registry - builds spine models with different configurations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from spine_sim.calibration import (
    CalibrationCase,
    CalibrationResult,
    PeakCalibrationCase,
    calibrate_model_curves_joint,
    calibrate_model_peaks_joint,
)
from spine_sim.model import SpineModel
from spine_sim.model_components import build_spine_elements


def _require_path(d: dict, path: str) -> object:
    """
    Require a nested path like "buttock.calibration.init.k_n_per_m".
    """
    cur: object = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f'Missing required config key: {path}')
        cur = cur[part]
    return cur


def _get_model_calibration_init(config: dict, model_key: str) -> dict:
    init = _require_path(config, f'{model_key}.calibration.init')
    if not isinstance(init, dict):
        raise ValueError(f'config.{model_key}.calibration.init must be an object/dict.')
    return init


def default_params_from_config(config: dict, model_key: str) -> dict:
    """
    Default calibration params used to seed zwt/maxwell calibration files
    when they don't exist.

    Includes:
      - spine scales
      - buttocks absolute
      - model-specific calibration init values
    """
    init_butt = _require_path(config, 'buttock.calibration.init')
    if not isinstance(init_butt, dict):
        raise ValueError('config.buttock.calibration.init must be an object/dict.')

    init_model = _get_model_calibration_init(config, model_key)

    ratios = init_model.get('maxwell_k_ratios', [1.0, 0.5])
    taus = init_model.get('maxwell_tau_ms', [10.0, 120.0])
    if not isinstance(ratios, list) or not isinstance(taus, list):
        raise ValueError(f'config.{model_key}.calibration.init.maxwell_* must be lists.')

    B = max(len(ratios), len(taus))
    ratios = (ratios + [0.0] * B)[:B]
    taus = (taus + [0.0] * B)[:B]

    params = {
        's_k_spine': 1.0,
        's_c_spine': 1.0,
        'buttocks_k_n_per_m': float(_require_path(config, 'buttock.calibration.init.k_n_per_m')),
        'buttocks_c_ns_per_m': float(_require_path(config, 'buttock.calibration.init.c_ns_per_m')),
        'buttocks_limit_mm': float(_require_path(config, 'buttock.calibration.init.limit_mm')),
        'c_base_ns_per_m': float(init_model.get('c_base_ns_per_m', 1200.0)),
        'disc_poly_k2_n_per_m2': float(init_model.get('disc_poly_k2_n_per_m2', 0.0)),
        'disc_poly_k3_n_per_m3': float(init_model.get('disc_poly_k3_n_per_m3', 0.0)),
    }

    for b in range(B):
        params[f'maxwell_k_ratio_{b}'] = float(ratios[b])
        params[f'maxwell_tau_ms_{b}'] = float(taus[b])

    return params


def _get_buttocks_fixed_config(config: dict) -> tuple[float, float, float, float, float]:
    """
    Get buttocks config:
      - init k/c/limit (used to initialize the base model)
      - stop_k/smoothing (fixed, not optimized)

    All required from config.json (no numeric fallbacks here by design).
    """
    init_k = float(_require_path(config, 'buttock.calibration.init.k_n_per_m'))
    init_c = float(_require_path(config, 'buttock.calibration.init.c_ns_per_m'))
    init_limit_mm = float(_require_path(config, 'buttock.calibration.init.limit_mm'))

    stop_k = float(_require_path(config, 'buttock.densification.stop_k_n_per_m'))
    smooth_mm = float(_require_path(config, 'buttock.densification.stop_smoothing_mm'))

    return init_k, init_c, init_limit_mm, stop_k, smooth_mm


def _build_spine_model(mass_map: dict, config: dict, model_key: str) -> SpineModel:
    """
    Build a spine model with nonlinear springs and Maxwell branches.

    Buttocks:
      - base_model is initialized from config.buttock.calibration.init (k/c/limit)
      - stop_k/smoothing are fixed from config.buttock.densification
      - during calibration/simulation, apply_calibration() overwrites k/c/limit

    Model-specific:
      - c_base, disc_poly, maxwell branch lists come from config.<model_key>.calibration.init
    """
    cfg_root = config.get(model_key, {})
    if not isinstance(cfg_root, dict):
        raise ValueError(f'config.{model_key} must be an object/dict.')

    init_model = _get_model_calibration_init(config, model_key)

    c_base = float(init_model.get('c_base_ns_per_m', 1200.0))
    node_names, masses, element_names, k_elem, c_elem = build_spine_elements(mass_map, c_base)

    # Inject buttocks initial k/c from config before Maxwell branch generation.
    init_k, init_c, init_limit_mm, stop_k, smooth_mm = _get_buttocks_fixed_config(config)
    k_elem = k_elem.copy()
    c_elem = c_elem.copy()
    k_elem[0] = init_k
    c_elem[0] = init_c

    disc_k2 = float(init_model.get('disc_poly_k2_n_per_m2', 0.0))
    disc_k3 = float(init_model.get('disc_poly_k3_n_per_m3', 0.0))

    disc_ref_mm = float(cfg_root.get('disc_ref_compression_mm', 2.0))
    disc_kmult = float(cfg_root.get('disc_k_mult_at_ref', 8.0))

    n_elem = len(k_elem)
    compression_ref_m = np.zeros(n_elem, dtype=float)
    compression_k_mult = np.ones(n_elem, dtype=float)
    tension_k_mult = np.ones(n_elem, dtype=float)
    compression_only = np.zeros(n_elem, dtype=bool)
    damping_compression_only = np.zeros(n_elem, dtype=bool)
    gap_m = np.zeros(n_elem, dtype=float)

    # Buttocks element (index 0)
    compression_only[0] = True
    damping_compression_only[0] = True

    # Spine elements (index 1+)
    compression_ref_m[1:] = disc_ref_mm / 1000.0
    compression_k_mult[1:] = disc_kmult

    poly_k2 = None
    poly_k3 = None
    if abs(disc_k2) > 0.0 or abs(disc_k3) > 0.0:
        poly_k2 = np.zeros(n_elem, dtype=float)
        poly_k3 = np.zeros(n_elem, dtype=float)
        poly_k2[1:] = disc_k2
        poly_k3[1:] = disc_k3

    # Maxwell branches (lists)
    mx_k_ratios = init_model.get('maxwell_k_ratios', [1.0, 0.5])
    mx_tau_ms = init_model.get('maxwell_tau_ms', [10.0, 120.0])
    if not isinstance(mx_k_ratios, list) or not isinstance(mx_tau_ms, list):
        raise ValueError(f'config.{model_key}.calibration.init.maxwell_* must be lists.')

    mx_k_ratios = [float(x) for x in mx_k_ratios]
    mx_tau_ms = [float(x) for x in mx_tau_ms]
    B = max(len(mx_k_ratios), len(mx_tau_ms))
    mx_k_ratios = (mx_k_ratios + [0.0] * B)[:B]
    mx_tau_ms = (mx_tau_ms + [0.0] * B)[:B]

    maxwell_k = np.zeros((n_elem, B), dtype=float)
    maxwell_tau_s = np.zeros((n_elem, B), dtype=float)
    for e in range(n_elem):
        for b in range(B):
            maxwell_k[e, b] = k_elem[e] * mx_k_ratios[b]
            maxwell_tau_s[e, b] = mx_tau_ms[b] / 1000.0

    maxwell_compression_only = np.ones(n_elem, dtype=bool)

    # Buttocks densification arrays:
    # limit is calibrated (initialized from config), stop params are fixed from config.
    compression_limit_m = np.zeros(n_elem, dtype=float)
    compression_stop_k = np.zeros(n_elem, dtype=float)
    compression_stop_smoothing_m = np.zeros(n_elem, dtype=float)

    compression_limit_m[0] = init_limit_mm / 1000.0
    compression_stop_k[0] = stop_k
    compression_stop_smoothing_m[0] = smooth_mm / 1000.0

    return SpineModel(
        node_names=node_names,
        masses_kg=masses,
        element_names=element_names,
        k_elem=k_elem,
        c_elem=c_elem,
        compression_ref_m=compression_ref_m,
        compression_k_mult=compression_k_mult,
        tension_k_mult=tension_k_mult,
        compression_only=compression_only,
        damping_compression_only=damping_compression_only,
        gap_m=gap_m,
        maxwell_k=maxwell_k,
        maxwell_tau_s=maxwell_tau_s,
        maxwell_compression_only=maxwell_compression_only,
        poly_k2=poly_k2,
        poly_k3=poly_k3,
        compression_limit_m=compression_limit_m,
        compression_stop_k=compression_stop_k,
        compression_stop_smoothing_m=compression_stop_smoothing_m,
    )


def apply_calibration(base_model: SpineModel, params: dict) -> SpineModel:
    """
    Apply calibration params:
      - buttocks absolute params: k/c/limit
      - spine scale params: s_k_spine, s_c_spine
      - optional model-specific: c_base, disc poly, maxwell branch ratios/tau

    Notes:
      - c_base_ns_per_m is applied by scaling the existing spine damping pattern (element 1+),
        using T9-T10 as the "reference that equals c_base" in build_spine_elements().
      - Maxwell branches are rebuilt from current element k and the branch ratios.
      - poly_k2/poly_k3 are additive (see model.py) and are scaled by s_k_spine.
    """
    s_k_spine = float(params.get('s_k_spine', 1.0))
    s_c_spine = float(params.get('s_c_spine', 1.0))

    butt_k = float(params.get('buttocks_k_n_per_m', base_model.k_elem[0]))
    butt_c = float(params.get('buttocks_c_ns_per_m', base_model.c_elem[0]))
    butt_limit_mm = float(params.get('buttocks_limit_mm', 0.0))

    n_elem = base_model.n_elems()
    B = base_model.n_maxwell()

    k = base_model.k_elem.copy()
    c = base_model.c_elem.copy()

    # Optional c_base scaling for spine dampers (element 1+)
    if 'c_base_ns_per_m' in params:
        c_base_new = float(params['c_base_ns_per_m'])
        try:
            ref_idx = base_model.element_names.index('T9-T10')
            c_base_init = float(base_model.c_elem[ref_idx])
        except ValueError:
            c_base_init = (
                float(np.median(base_model.c_elem[1:][base_model.c_elem[1:] > 0.0]))
                if n_elem > 1
                else 0.0
            )

        if c_base_init > 0.0 and n_elem > 1:
            c[1:] *= c_base_new / c_base_init

    # Buttocks absolute
    k[0] = butt_k
    c[0] = butt_c

    # Spine scales
    k[1:] *= s_k_spine
    c[1:] *= s_c_spine

    # Maxwell rebuild (ratios + tau)
    mx_k = base_model.maxwell_k.copy()
    mx_tau_s = base_model.maxwell_tau_s.copy()
    if B > 0:
        ratios = np.array(
            [float(params.get(f'maxwell_k_ratio_{b}', 0.0)) for b in range(B)],
            dtype=float,
        )
        tau_ms = np.array(
            [float(params.get(f'maxwell_tau_ms_{b}', 0.0)) for b in range(B)],
            dtype=float,
        )
        mx_tau_s = np.tile((tau_ms / 1000.0)[None, :], (n_elem, 1))
        mx_k = np.zeros((n_elem, B), dtype=float)
        for e in range(n_elem):
            mx_k[e, :] = k[e] * ratios

    # Poly extras (additive) for spine elements (1+), scaled by s_k_spine
    poly_k2 = None if base_model.poly_k2 is None else base_model.poly_k2.copy()
    poly_k3 = None if base_model.poly_k3 is None else base_model.poly_k3.copy()

    disc_k2 = float(params.get('disc_poly_k2_n_per_m2', 0.0))
    disc_k3 = float(params.get('disc_poly_k3_n_per_m3', 0.0))
    if abs(disc_k2) > 0.0 or abs(disc_k3) > 0.0 or poly_k2 is not None or poly_k3 is not None:
        if poly_k2 is None:
            poly_k2 = np.zeros(n_elem, dtype=float)
        if poly_k3 is None:
            poly_k3 = np.zeros(n_elem, dtype=float)
        poly_k2[1:] = disc_k2
        poly_k3[1:] = disc_k3

        poly_k2[1:] *= s_k_spine
        poly_k3[1:] *= s_k_spine

    # Densification limit for buttocks (stop params already live in base_model)
    limit_m = (
        None if base_model.compression_limit_m is None else base_model.compression_limit_m.copy()
    )
    if limit_m is None:
        limit_m = np.zeros(base_model.n_elems(), dtype=float)
    limit_m[0] = butt_limit_mm / 1000.0

    return SpineModel(
        node_names=base_model.node_names,
        masses_kg=base_model.masses_kg,
        element_names=base_model.element_names,
        k_elem=k,
        c_elem=c,
        compression_ref_m=base_model.compression_ref_m,
        compression_k_mult=base_model.compression_k_mult,
        tension_k_mult=base_model.tension_k_mult,
        compression_only=base_model.compression_only,
        damping_compression_only=base_model.damping_compression_only,
        gap_m=base_model.gap_m,
        maxwell_k=mx_k,
        maxwell_tau_s=mx_tau_s,
        maxwell_compression_only=base_model.maxwell_compression_only,
        poly_k2=poly_k2,
        poly_k3=poly_k3,
        compression_limit_m=limit_m,
        compression_stop_k=base_model.compression_stop_k,
        compression_stop_smoothing_m=base_model.compression_stop_smoothing_m,
    )


def calibrate_peaks(
    base_model: SpineModel,
    cases: list[PeakCalibrationCase],
    t12_element_index: int,
    *,
    init_params: dict,
    bounds: dict[str, tuple[float, float]],
    verbose: bool = True,
    n_starts: int = 5,
    cost_tol: float = 1e-4,
    stall_iters: int = 10,
) -> CalibrationResult:
    return calibrate_model_peaks_joint(
        base_model,
        cases,
        t12_element_index,
        init_params=init_params,
        bounds=bounds,
        apply_params=apply_calibration,
        max_nfev=200,
        verbose=verbose,
        n_starts=n_starts,
        cost_tol=cost_tol,
        stall_iters=stall_iters,
        explore_samples=200,
        explore_keep=40,
        diversity_min_dist=0.25,
        snap_norm_step=0.02,
        cache_norm_step=0.02,
    )


def calibrate_curves(
    base_model: SpineModel,
    cases: list[CalibrationCase],
    t12_element_index: int,
    *,
    init_params: dict,
    bounds: dict[str, tuple[float, float]],
) -> CalibrationResult:
    return calibrate_model_curves_joint(
        base_model,
        cases,
        t12_element_index,
        init_params=init_params,
        bounds=bounds,
        apply_params=apply_calibration,
        max_nfev=200,
    )


@dataclass
class ModelPath:
    name: str
    build_model: Callable[[dict, dict], SpineModel]
    apply_calibration: Callable[[SpineModel, dict], SpineModel]
    calibrate_peaks: Callable[..., CalibrationResult]
    calibrate_curves: Callable[..., CalibrationResult]
    default_params: Callable[[dict], dict]


MODEL_PATHS: dict[str, ModelPath] = {
    'maxwell': ModelPath(
        name='maxwell',
        build_model=lambda m, c: _build_spine_model(m, c, 'maxwell'),
        apply_calibration=apply_calibration,
        calibrate_peaks=calibrate_peaks,
        calibrate_curves=calibrate_curves,
        default_params=lambda cfg: default_params_from_config(cfg, 'maxwell'),
    ),
    'zwt': ModelPath(
        name='zwt',
        build_model=lambda m, c: _build_spine_model(m, c, 'zwt'),
        apply_calibration=apply_calibration,
        calibrate_peaks=calibrate_peaks,
        calibrate_curves=calibrate_curves,
        default_params=lambda cfg: default_params_from_config(cfg, 'zwt'),
    ),
}


def get_model_path(name: str) -> ModelPath:
    key = name.strip().lower()
    if key not in MODEL_PATHS:
        valid = ', '.join(MODEL_PATHS.keys())
        raise ValueError(f"Unknown model type '{name}'. Available: {valid}")
    return MODEL_PATHS[key]
