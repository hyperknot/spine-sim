from __future__ import annotations

import json
from pathlib import Path

from spine_sim.calibration import CalibrationResult


CALIBRATION_ROOT = Path(__file__).resolve().parents[1] / 'calibration'
VALID_MODES = {'peaks', 'curves'}


def calibration_file(model_type: str) -> Path:
    return CALIBRATION_ROOT / f'{model_type}.json'


def _default_section(default_scales: dict) -> dict:
    return {
        'scales': {k: float(v) for k, v in default_scales.items()},
        'result': {'success': False, 'cost': None, 'residual_norm': None},
        'cases': [],
    }


def _default_doc(model_type: str, default_scales: dict) -> dict:
    return {
        'model': model_type,
        'active_mode': 'peaks',
        'peaks': _default_section(default_scales),
        'curves': _default_section(default_scales),
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
    section = doc.get(mode, {})
    scales = section.get('scales', None)

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

    doc = load_calibration_doc(model_type, default_scales)
    doc['active_mode'] = mode
    doc[mode] = {
        'scales': result.scales,
        'result': {
            'success': result.success,
            'cost': result.cost,
            'residual_norm': result.residual_norm,
        },
        'cases': cases,
    }

    path = calibration_file(model_type)
    path.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
