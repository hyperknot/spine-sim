"""Single source of truth for config + repo paths (no env overrides).

Policy:
- No fallback/default config values in code.
- If required config keys are missing, terminate with a clear error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).parent.parent


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _require_path(cfg: dict, keys: list[str]) -> Any:
    cur: Any = cfg
    prefix: list[str] = []
    for k in keys:
        prefix.append(k)
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f'Missing required config key: {".".join(prefix)}')
        cur = cur[k]
    return cur


def req_str(cfg: dict, keys: list[str]) -> str:
    v = _require_path(cfg, keys)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f'Config key {".".join(keys)} must be a non-empty string.')
    return v


def req_float(cfg: dict, keys: list[str]) -> float:
    v = _require_path(cfg, keys)
    try:
        return float(v)
    except Exception as e:
        raise ValueError(f'Config key {".".join(keys)} must be a float-like value.') from e


def req_int(cfg: dict, keys: list[str]) -> int:
    v = _require_path(cfg, keys)
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f'Config key {".".join(keys)} must be an int-like value.') from e


def read_config() -> dict:
    cfg = load_json(REPO_ROOT / 'config.json')
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    # Existence/type checks (no defaults).
    req_str(cfg, ['model', 'masses_json'])
    req_float(cfg, ['model', 'arm_recruitment'])
    req_float(cfg, ['model', 'helmet_mass_kg'])

    req_float(cfg, ['solver', 'dt_internal_s'])
    req_int(cfg, ['solver', 'max_newton_iter'])
    req_float(cfg, ['solver', 'newton_tol'])

    req_float(cfg, ['drop', 'cfc'])
    req_float(cfg, ['drop', 'style_duration_threshold_ms'])
    req_float(cfg, ['drop', 'sim_duration_ms'])
    req_float(cfg, ['drop', 'gravity_settle_ms'])
    req_float(cfg, ['drop', 'peak_threshold_g'])
    req_float(cfg, ['drop', 'freefall_threshold_g'])

    # Buttocks model (mode/profile are supplied at runtime via CLI).
    req_float(cfg, ['buttock', 'k2_mult'])

    for p in ('sporty', 'avg', 'soft'):
        req_float(cfg, ['buttock', 'profiles', p, 'apex_thickness_mm'])
        req_float(cfg, ['buttock', 'profiles', p, 'k1_n_per_m'])
        req_float(cfg, ['buttock', 'profiles', p, 'c_ns_per_m'])

    req_float(cfg, ['spine', 'disc_height_mm'])
    req_float(cfg, ['spine', 'cervical_disc_height_single_mm'])
    req_float(cfg, ['spine', 'tension_k_mult'])
    req_float(cfg, ['spine', 'damping_ns_per_m'])
    req_float(cfg, ['spine', 'kemper', 'normalize_to_eps_per_s'])
    req_float(cfg, ['spine', 'kemper', 'strain_rate_smoothing_tau_ms'])
    req_float(cfg, ['spine', 'kemper', 'warn_over_eps_per_s'])

    req_float(cfg, ['plotting', 'buttocks_height_mm'])
