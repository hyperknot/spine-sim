from __future__ import annotations

import json
from pathlib import Path


CALIBRATION_ROOT = Path(__file__).resolve().parents[1] / 'calibration'
TOEN_DROP_FILE = CALIBRATION_ROOT / 'toen_drop.json'


def load_toen_drop_calibration() -> tuple[dict | None, Path]:
    if not TOEN_DROP_FILE.exists():
        return None, TOEN_DROP_FILE
    doc = json.loads(TOEN_DROP_FILE.read_text(encoding='utf-8'))
    if not isinstance(doc, dict):
        return None, TOEN_DROP_FILE
    return doc, TOEN_DROP_FILE


def require_toen_drop_calibration() -> tuple[dict, Path]:
    """
    Load calibration and fail loudly if it doesn't exist.

    This enforces: run calibration first, then simulate.
    """
    doc, path = load_toen_drop_calibration()
    if doc is None:
        raise FileNotFoundError(
            f'Missing Toen drop calibration: {path}\nRun the buttocks calibration command first.'
        )
    return doc, path


def write_toen_drop_calibration(params: dict) -> Path:
    """Write buttocks calibration parameters directly to file."""
    TOEN_DROP_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOEN_DROP_FILE.write_text(json.dumps(params, indent=2) + '\n', encoding='utf-8')
    return TOEN_DROP_FILE
