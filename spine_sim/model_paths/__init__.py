from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from spine_sim.calibration import CalibrationResult
from spine_sim.model import SpineModel
from . import maxwell, zwt


@dataclass
class ModelPath:
    name: str
    build_model: Callable[[dict, dict], SpineModel]
    apply_calibration: Callable[[SpineModel, dict], SpineModel]
    calibrate_peaks: Callable[..., CalibrationResult]
    calibrate_curves: Callable[..., CalibrationResult]
    default_scales: dict


MODEL_PATHS: dict[str, ModelPath] = {
    "maxwell": ModelPath(
        name="maxwell",
        build_model=maxwell.build_model,
        apply_calibration=maxwell.apply_calibration,
        calibrate_peaks=maxwell.calibrate_peaks,
        calibrate_curves=maxwell.calibrate_curves,
        default_scales=maxwell.DEFAULT_SCALES,
    ),
    "zwt": ModelPath(
        name="zwt",
        build_model=zwt.build_model,
        apply_calibration=zwt.apply_calibration,
        calibrate_peaks=zwt.calibrate_peaks,
        calibrate_curves=zwt.calibrate_curves,
        default_scales=zwt.DEFAULT_SCALES,
    ),
}


def get_model_path(name: str) -> ModelPath:
    key = name.strip().lower()
    if key not in MODEL_PATHS:
        valid = ", ".join(MODEL_PATHS.keys())
        raise ValueError(f"Unknown model type '{name}'. Available: {valid}")
    return MODEL_PATHS[key]
