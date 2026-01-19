"""Simulation logic and helpers for spine impact simulation.

This module re-exports from sub-modules for backwards compatibility.
"""

from __future__ import annotations

# Buttocks utilities (now generic; no Toen-application into spine model)
from spine_sim.buttocks import (
    compute_free_buttocks_height_mm,
    generate_buttocks_plot,
)

# Buttocks commands (Toen surrogate mode kept)
from spine_sim.buttocks_commands import (
    run_calibrate_buttocks,
    run_simulate_buttocks,
)

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

# Drop commands (now use unified model calibration files, not toen_drop.json)
from spine_sim.drop_commands import (
    run_calibrate_drop,
    run_simulate_drop,
)

# Input processing
from spine_sim.input_processing import (
    detect_style,
    freefall_bias_correct,
    process_input,
)

# Mass map building
from spine_sim.mass import build_mass_map

# Output utilities
from spine_sim.output import write_timeseries_csv


__all__ = [
    # Config
    'REPO_ROOT',
    'DEFAULT_MASSES_JSON',
    'CALIBRATION_ROOT',
    'CALIBRATION_YOGANANDAN_DIR',
    'read_config',
    'resolve_path',
    'load_masses',
    # Mass
    'build_mass_map',
    # Input processing
    'detect_style',
    'freefall_bias_correct',
    'process_input',
    # Output
    'write_timeseries_csv',
    # Buttocks
    'compute_free_buttocks_height_mm',
    'generate_buttocks_plot',
    # Commands
    'run_calibrate_buttocks',
    'run_simulate_buttocks',
    'run_calibrate_drop',
    'run_simulate_drop',
]
