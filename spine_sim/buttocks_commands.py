"""Buttocks calibration and simulation commands."""

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

# ----------------------------
# Hard-coded run settings
# ----------------------------
OUTPUT_DIR = Path("output/toen_drop")

MALE50_MASS_KG = 75.4

# Densification/stop settings are not in config anymore.
# They are used during calibration and saved into the calibration file.
BUTTOCKS_STOP_K_N_PER_M = 5.0e6
BUTTOCKS_STOP_SMOOTHING_MM = 1.0

# Hard-coded solver parameters (simulation uses these).
DT_S = TOEN_SOLVER_DT_S
DURATION_S = TOEN_SOLVER_DURATION_S
MAX_NEWTON_ITER = TOEN_SOLVER_MAX_NEWTON_ITER

DEFAULT_VELOCITIES_MPS = [TOEN_IMPACT_V_MPS]


def run_calibrate_buttocks(echo=print) -> dict:
    """Calibrate buttocks model from Toen 2012 paper data and save to calibration/toen_drop.json."""
    echo("Calibrating buttocks model (fixed: subject=avg, targets=avg paper, v=3.5 m/s).")

    result = calibrate_toen_buttocks_model(
        male50_mass_kg=MALE50_MASS_KG,
        buttocks_stop_k_n_per_m=BUTTOCKS_STOP_K_N_PER_M,
        buttocks_stop_smoothing_mm=BUTTOCKS_STOP_SMOOTHING_MM,
    )

    out_path = write_toen_drop_calibration(result)
    echo(f"Calibration saved: {out_path}")
    return result


def run_simulate_buttocks(echo=print) -> list[dict]:
    """Simulate Toen drop suite and generate plots. Requires an existing calibration file."""
    config = read_config()
    bcfg = config.get("buttock", {})

    velocities = [float(v) for v in bcfg.get("velocities_mps", DEFAULT_VELOCITIES_MPS)]

    # Enforce "calibration required"
    doc, path = require_toen_drop_calibration()
    echo(f"Loaded calibration from {path}")

    buttocks_params = {
        "k": float(doc["buttocks_k_n_per_m"]),
        "c": float(doc["buttocks_c_ns_per_m"]),
        "limit_mm": float(doc["buttocks_limit_mm"]),
        "stop_k": float(doc["buttocks_stop_k_n_per_m"]),
        "smoothing_mm": float(doc["buttocks_stop_smoothing_mm"]),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    echo(f"Simulating (fixed: subject=avg). velocities={velocities}")

    results = run_toen_suite(
        impact_velocities_mps=velocities,
        male50_mass_kg=MALE50_MASS_KG,
        dt_s=DT_S,
        duration_s=DURATION_S,
        max_newton_iter=MAX_NEWTON_ITER,
        buttocks_k_n_per_m=buttocks_params["k"],
        buttocks_c_ns_per_m=buttocks_params["c"],
        buttocks_limit_mm=buttocks_params["limit_mm"],
        buttocks_stop_k_n_per_m=buttocks_params["stop_k"],
        buttocks_stop_smoothing_mm=buttocks_params["smoothing_mm"],
    )

    all_results = [r.__dict__ for r in results]
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")

    for v_plot in velocities:
        plot_path = generate_buttocks_plot(
            OUTPUT_DIR,
            v_plot=v_plot,
            buttocks_params=buttocks_params,
            dt_s=DT_S,
            duration_s=DURATION_S,
            max_newton_iter=MAX_NEWTON_ITER,
        )
        echo(f"  Plot: {plot_path}")

    echo(f"Simulation and plots complete. Output: {OUTPUT_DIR}")
    return all_results
