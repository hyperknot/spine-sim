from __future__ import annotations

import json
from pathlib import Path

from spine_sim.calibration import CalibrationResult


CALIBRATION_ROOT = Path(__file__).resolve().parents[1] / 'calibration'
VALID_MODES = {'peaks', 'curves'}


def calibration_file(model_type: str) -> Path:
    return CALIBRATION_ROOT / f'{model_type}.json'


def _default_doc(model_type: str, default_scales: dict) -> dict:
    return {
        'model': model_type,
        'active_mode': 'peaks',
        'peaks': {k: float(v) for k, v in default_scales.items()},
        'curves': {k: float(v) for k, v in default_scales.items()},
    }


def ensure_calibration_file(model_type: str, default_scales: dict) -> Path:
    path = calibration_file(model_type)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = _default_doc(model_type, default_scales)
        path.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
    return path


def load_calibration_doc(model_type: str, default_scales: dict) -> dict:
    path = ensure_calibration_file(model_type, default_scales)
    return json.loads(path.read_text(encoding='utf-8'))


def load_calibration_scales(model_type: str, mode: str, default_scales: dict) -> dict:
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown calibration mode '{mode}'. Use: {sorted(VALID_MODES)}")

    doc = load_calibration_doc(model_type, default_scales)
    scales = doc.get(mode, {})

    if not isinstance(scales, dict):
        raise ValueError(f"Missing scales for {model_type} mode '{mode}' in calibration file.")

    return {k: float(scales.get(k, v)) for k, v in default_scales.items()}


def write_calibration_result(
    model_type: str,
    mode: str,
    result: CalibrationResult,
    cases: list[dict] | list[str],
    default_scales: dict,
) -> None:
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown calibration mode '{mode}'. Use: {sorted(VALID_MODES)}")

    # Print debug info to console
    print(f"\n=== CALIBRATION RESULT ({model_type}/{mode}) ===")
    print(f"  success: {result.success}")
    print(f"  cost: {result.cost}")
    print(f"  residual_norm: {result.residual_norm}")
    print(f"  cases: {cases}")

    doc = load_calibration_doc(model_type, default_scales)
    doc['active_mode'] = mode
    doc[mode] = result.scales

    path = calibration_file(model_type)
    path.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
