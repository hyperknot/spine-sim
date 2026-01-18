from __future__ import annotations

import json
from pathlib import Path


CALIBRATION_ROOT = Path(__file__).resolve().parents[1] / 'calibration'
BUTTOCKS_ONLY_FILE = CALIBRATION_ROOT / 'buttocks_only.json'


def load_buttocks_only_overrides() -> tuple[dict | None, Path]:
    """
    Optional buttocks-only override file.

    Format:
      {
        "active": true,
        "params": { ... same keys as config.json: buttocks_only ... },
        "meta": {...}
      }
    """
    if not BUTTOCKS_ONLY_FILE.exists():
        return None, BUTTOCKS_ONLY_FILE

    doc = json.loads(BUTTOCKS_ONLY_FILE.read_text(encoding='utf-8'))
    if not bool(doc.get('active', False)):
        return None, BUTTOCKS_ONLY_FILE

    params = doc.get('params', None)
    if not isinstance(params, dict):
        return None, BUTTOCKS_ONLY_FILE

    return params, BUTTOCKS_ONLY_FILE


def write_buttocks_only_overrides(
    params: dict, meta: dict | None = None, *, active: bool = True
) -> Path:
    BUTTOCKS_ONLY_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        'active': bool(active),
        'params': {k: float(v) for k, v in params.items()},
        'meta': meta or {},
    }
    BUTTOCKS_ONLY_FILE.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
    return BUTTOCKS_ONLY_FILE
