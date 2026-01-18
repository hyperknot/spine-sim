from __future__ import annotations

import json
from pathlib import Path

CALIBRATION_ROOT = Path(__file__).resolve().parents[1] / "calibration"
TOEN_DROP_FILE = CALIBRATION_ROOT / "toen_drop.json"


def load_toen_drop_calibration() -> tuple[dict | None, Path]:
    if not TOEN_DROP_FILE.exists():
        return None, TOEN_DROP_FILE
    doc = json.loads(TOEN_DROP_FILE.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return None, TOEN_DROP_FILE
    if not bool(doc.get("active", True)):
        return None, TOEN_DROP_FILE
    return doc, TOEN_DROP_FILE


def write_toen_drop_calibration(result: dict, *, active: bool = True) -> Path:
    TOEN_DROP_FILE.parent.mkdir(parents=True, exist_ok=True)
    doc = {"active": bool(active), "result": result}
    TOEN_DROP_FILE.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return TOEN_DROP_FILE
