from __future__ import annotations

import os


def env_str(name: str) -> str | None:
    v = os.getenv(name, None)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def env_float(name: str) -> float | None:
    v = env_str(name)
    if v is None:
        return None
    return float(v)


def env_int(name: str) -> int | None:
    v = env_str(name)
    if v is None:
        return None
    return int(v)


def env_bool(name: str) -> bool | None:
    v = env_str(name)
    if v is None:
        return None
    s = v.strip().lower()
    if s in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Invalid boolean env var {name}={v!r}. Use true/false, 1/0, yes/no.')


def env_float_list(name: str) -> list[float] | None:
    v = env_str(name)
    if v is None:
        return None
    # Accept "3.5,8.0" or "3.5 8.0"
    parts = [p.strip() for p in v.replace(' ', ',').split(',') if p.strip()]
    return [float(p) for p in parts]
