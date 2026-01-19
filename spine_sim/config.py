"""Configuration and path utilities for spine simulation."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MASSES_JSON = REPO_ROOT / "opensim" / "fullbody.json"
CALIBRATION_ROOT = REPO_ROOT / "calibration"
CALIBRATION_YOGANANDAN_DIR = CALIBRATION_ROOT / "yoganandan"


def read_config() -> dict:
    return json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_masses(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
