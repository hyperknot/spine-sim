"""Buttocks calibration and simulation commands (Toen 2-DOF surrogate)."""

from __future__ import annotations

import json
from pathlib import Path

from spine_sim.buttocks import generate_buttocks_plot
from spine_sim.config import read_config
from spine_sim.toen_drop import (
    TOEN_IMPACT_V_MPS,
    TOEN_SOLVER_DT_S,
    TOEN_SOLVER_DURATION_S,
    TOEN_SOLVER_MAX_NEWTON_ITER,
    calibrate_toen_buttocks_model,
    run_toen_suite,
)
from spine_sim.toen_store import require_toen_drop_calibration, write_toen_drop_calibration


OUTPUT_DIR = Path('output/toen_drop')

DT_S = TOEN_SOLVER_DT_S
DURATION_S = TOEN_SOLVER_DURATION_S
MAX_NEWTON_ITER = TOEN_SOLVER_MAX_NEWTON_ITER

DEFAULT_VELOCITIES_MPS = [TOEN_IMPACT_V_MPS]


def _require_path(d: dict, path: str) -> object:
    cur: object = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f'Missing required config key: {path}')
        cur = cur[part]
    return cur


def run_calibrate_buttocks(echo=print) -> dict:
    """
    Calibrate buttocks model from Toen 2012 paper data and save to calibration/toen_drop.json.

    This is a separate mode used for validating the buttocks surrogate only.
    Drop calibration/simulation does NOT use this file.
    """
    config = read_config()

    init_k = float(_require_path(config, 'buttock.calibration.init.k_n_per_m'))
    init_c = float(_require_path(config, 'buttock.calibration.init.c_ns_per_m'))
    init_limit = float(_require_path(config, 'buttock.calibration.init.limit_mm'))

    b0, b1 = _require_path(config, 'buttock.calibration.bounds.k_n_per_m')
    c0, c1 = _require_path(config, 'buttock.calibration.bounds.c_ns_per_m')
    l0, l1 = _require_path(config, 'buttock.calibration.bounds.limit_mm')

    stop_k = float(_require_path(config, 'buttock.densification.stop_k_n_per_m'))
    smooth_mm = float(_require_path(config, 'buttock.densification.stop_smoothing_mm'))

    echo('Calibrating buttocks model (Toen surrogate).')

    result = calibrate_toen_buttocks_model(
        buttocks_stop_k_n_per_m=stop_k,
        buttocks_stop_smoothing_mm=smooth_mm,
        init_k_n_per_m=init_k,
        init_c_ns_per_m=init_c,
        init_limit_mm=init_limit,
        bounds_k_n_per_m=(float(b0), float(b1)),
        bounds_c_ns_per_m=(float(c0), float(c1)),
        bounds_limit_mm=(float(l0), float(l1)),
    )

    out_path = write_toen_drop_calibration(result)
    echo(f'Calibration saved: {out_path}')
    return result


def run_simulate_buttocks(echo=print) -> list[dict]:
    """Simulate Toen drop suite and generate plots. Requires an existing toen_drop.json calibration file."""
    config = read_config()
    bcfg = config.get('buttock', {})

    velocities = [float(v) for v in bcfg.get('velocities_mps', DEFAULT_VELOCITIES_MPS)]

    doc, path = require_toen_drop_calibration()
    echo(f'Loaded calibration from {path}')

    buttocks_params = {
        'k': float(doc['buttocks_k_n_per_m']),
        'c': float(doc['buttocks_c_ns_per_m']),
        'limit_mm': float(doc['buttocks_limit_mm']),
        'stop_k': float(doc['buttocks_stop_k_n_per_m']),
        'smoothing_mm': float(doc['buttocks_stop_smoothing_mm']),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    echo(f'Simulating Toen surrogate suite. velocities={velocities}')

    results = run_toen_suite(
        impact_velocities_mps=velocities,
        dt_s=DT_S,
        duration_s=DURATION_S,
        max_newton_iter=MAX_NEWTON_ITER,
        buttocks_k_n_per_m=buttocks_params['k'],
        buttocks_c_ns_per_m=buttocks_params['c'],
        buttocks_limit_mm=buttocks_params['limit_mm'],
        buttocks_stop_k_n_per_m=buttocks_params['stop_k'],
        buttocks_stop_smoothing_mm=buttocks_params['smoothing_mm'],
    )

    all_results = [r.__dict__ for r in results]
    (OUTPUT_DIR / 'summary.json').write_text(
        json.dumps(all_results, indent=2) + '\n', encoding='utf-8'
    )

    for v_plot in velocities:
        plot_path = generate_buttocks_plot(
            OUTPUT_DIR,
            v_plot=v_plot,
            buttocks_params=buttocks_params,
            dt_s=DT_S,
            duration_s=DURATION_S,
            max_newton_iter=MAX_NEWTON_ITER,
        )
        echo(f'  Plot: {plot_path}')

    echo(f'Simulation and plots complete. Output: {OUTPUT_DIR}')
    return all_results
