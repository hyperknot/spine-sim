"""Simulation logic and helpers for spine impact simulation.

This module re-exports from sub-modules for backwards compatibility.
"""

from __future__ import annotations

# Config and path utilities
from spine_sim.config import (
    CALIBRATION_ROOT,
    CALIBRATION_YOGANANDAN_DIR,
    DEFAULT_MASSES_JSON,
    REPO_ROOT,
    load_masses,
    read_config,
    resolve_path,
)

# Mass map building
from spine_sim.mass import build_mass_map

# Input processing
from spine_sim.input_processing import (
    detect_style,
    freefall_bias_correct,
    process_input,
)

# Output utilities
from spine_sim.output import write_timeseries_csv

# Buttocks model utilities
from spine_sim.buttocks import (
    apply_toen_buttocks_to_model,
    compute_free_buttocks_height_mm,
    generate_buttocks_plot,
    get_toen_buttocks_params,
)

# Buttocks commands
from spine_sim.buttocks_commands import (
    run_calibrate_buttocks,
    run_simulate_buttocks,
)

# Drop commands
from spine_sim.drop_commands import (
    run_calibrate_drop,
    run_simulate_drop,
)

__all__ = [
    # Config
    "REPO_ROOT",
    "DEFAULT_MASSES_JSON",
    "CALIBRATION_ROOT",
    "CALIBRATION_YOGANANDAN_DIR",
    "read_config",
    "resolve_path",
    "load_masses",
    # Mass
    "build_mass_map",
    # Input processing
    "detect_style",
    "freefall_bias_correct",
    "process_input",
    # Output
    "write_timeseries_csv",
    # Buttocks
    "get_toen_buttocks_params",
    "apply_toen_buttocks_to_model",
    "compute_free_buttocks_height_mm",
    "generate_buttocks_plot",
    # Commands
    "run_calibrate_buttocks",
    "run_simulate_buttocks",
    "run_calibrate_drop",
    "run_simulate_drop",
]
