from __future__ import annotations

import numpy as np


def _exp_to_token(exponent: float) -> str:
    # 1.5 -> "1p5" for safe CSV headers
    return f'{float(exponent):.1f}'.replace('.', 'p')


def calculate_sic_matrix(
    accel_g: np.ndarray,
    *,
    sample_rate_hz: float,
    window_ms_list: list[int],
    exponents: list[float],
    column_prefix: str,
) -> dict[str, float]:
    """
    SIC (HIC-style) computed from *CFC-filtered acceleration sensor data*.

    Discrete-time approximation (piecewise-constant per sample), matching the TS HIC reference:
      For an interval of N samples:
        deltaT = N * dt
        mean   = (1/N) * sum(|a[k]|)
        SIC    = deltaT * (mean ** exponent)

    We compute:
      SIC_T = max over all intervals with duration <= T.

    Inputs:
      accel_g: acceleration in g (signed allowed). We use abs() like HIC.
      sample_rate_hz: constant sample rate (post-resample).
    """
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError('sample_rate_hz must be > 0.')
    if not window_ms_list:
        raise ValueError('window_ms_list must be non-empty.')
    if not exponents:
        raise ValueError('exponents must be non-empty.')

    a = np.asarray(accel_g, dtype=float)
    if a.size == 0:
        raise ValueError('accel_g must be non-empty.')
    if np.any(~np.isfinite(a)):
        bad = int(np.argmax(~np.isfinite(a)))
        raise ValueError(f'accel_g contains non-finite value at index {bad}: {a[bad]}')

    # HIC convention: magnitude
    a = np.abs(a)

    n = int(a.size)
    dt = 1.0 / float(sample_rate_hz)

    max_window_ms = int(max(int(w) for w in window_ms_list))
    max_n_all = max(1, int(np.floor((max_window_ms / 1000.0) * float(sample_rate_hz))))
    max_n_all = min(max_n_all, n)

    exps = np.asarray([float(e) for e in exponents], dtype=float)
    if np.any(~np.isfinite(exps)) or np.any(exps <= 0):
        raise ValueError('All exponents must be finite and > 0.')

    # Prefix sums: prefix[i] = sum(a[0:i])
    prefix = np.empty(n + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(a, out=prefix[1:])

    # max_by_n[N-1, j] = max SIC over all intervals of exactly N samples, exponent j
    max_by_n = np.zeros((max_n_all, exps.size), dtype=float)

    for N in range(1, max_n_all + 1):
        sums = prefix[N:] - prefix[:-N]  # length n-N+1
        mean = sums / float(N)
        delta_t = float(N) * dt

        vals = delta_t * np.power(mean[:, None], exps[None, :])
        max_by_n[N - 1, :] = np.max(vals, axis=0)

    # cummax over N so we can answer "<= window" queries
    cummax = np.maximum.accumulate(max_by_n, axis=0)

    out: dict[str, float] = {}
    for window_ms in window_ms_list:
        max_n = max(1, int(np.floor((float(window_ms) / 1000.0) * float(sample_rate_hz))))
        max_n = min(max_n, max_n_all)
        row = cummax[max_n - 1, :]

        for j, e in enumerate(exps.tolist()):
            col = f'{column_prefix}_w{int(window_ms)}_e{_exp_to_token(e)}'
            out[col] = float(row[j])

    return out
