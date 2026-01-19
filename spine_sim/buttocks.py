from __future__ import annotations

from pathlib import Path

import numpy as np
from spine_sim.plotting import plot_toen_buttocks_force_compression
from spine_sim.toen_drop import (
    TOEN_SOLVER_DT_S,
    TOEN_SOLVER_DURATION_S,
    TOEN_SOLVER_MAX_NEWTON_ITER,
    TOEN_TABLE1_BODY_MASS_KG,
    TOEN_TABLE1_SKIN_MASS_KG,
    simulate_toen_drop_trace,
)
from spine_sim.toen_targets import TOEN_FLOOR_STIFFNESS_N_PER_M


# Plotting-only heuristic constant (kept local by request).
DEFAULT_FREE_HEIGHT_FROM_LIMIT_DIVISOR = 0.6


def compute_free_buttocks_height_mm(buttocks_limit_mm: float | None) -> float:
    """
    Plotting-only heuristic for buttocks free height.

    Historically: free_height ≈ limit / 0.6.
    """
    if buttocks_limit_mm is None:
        return 100.0
    return float(buttocks_limit_mm) / float(DEFAULT_FREE_HEIGHT_FROM_LIMIT_DIVISOR)


def generate_buttocks_plot(
    out_dir: Path,
    v_plot: float,
    buttocks_params: dict,
    dt_s: float = TOEN_SOLVER_DT_S,
    duration_s: float = TOEN_SOLVER_DURATION_S,
    max_newton_iter: int = TOEN_SOLVER_MAX_NEWTON_ITER,
) -> Path:
    """
    Generate buttocks force/compression plot for a velocity using the 2-DOF Toen surrogate.

    Note: Drop calibration/simulation does NOT use this surrogate anymore.
    """
    body_mass = float(TOEN_TABLE1_BODY_MASS_KG)
    skin_mass = float(TOEN_TABLE1_SKIN_MASS_KG)

    k_butt = float(buttocks_params['k'])
    c_butt = float(buttocks_params['c'])
    limit_mm = float(buttocks_params['limit_mm'])
    stop_k = float(buttocks_params['stop_k'])
    smoothing_mm = float(buttocks_params['smoothing_mm'])

    compression_by_floor: dict[str, np.ndarray] = {}
    force_by_floor: dict[str, np.ndarray] = {}
    time_s = None

    for floor_name, k_floor in TOEN_FLOOR_STIFFNESS_N_PER_M.items():
        _res, trace = simulate_toen_drop_trace(
            floor_name=floor_name,
            body_mass_kg=body_mass,
            skin_mass_kg=skin_mass,
            buttocks_k_n_per_m=k_butt,
            buttocks_c_ns_per_m=c_butt,
            floor_k_n_per_m=float(k_floor),
            impact_velocity_mps=v_plot,
            buttocks_limit_mm=limit_mm,
            buttocks_stop_k_n_per_m=stop_k,
            buttocks_stop_smoothing_mm=smoothing_mm,
            dt_s=dt_s,
            duration_s=duration_s,
            max_newton_iter=max_newton_iter,
        )
        if time_s is None:
            time_s = trace.time_s
        compression_by_floor[floor_name] = trace.buttocks_compression_m * 1000.0
        force_by_floor[floor_name] = trace.buttocks_force_n / 1000.0

    out_path = out_dir / f'buttocks_force_compression_v{v_plot:.2f}.png'
    plot_toen_buttocks_force_compression(
        time_s,
        compression_by_floor_mm=compression_by_floor,
        force_by_floor_kN=force_by_floor,
        out_path=out_path,
        title=f'Toen surrogate buttocks response (v={v_plot:.2f} m/s)',
    )
    return out_path
