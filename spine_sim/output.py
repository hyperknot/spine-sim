"""Output utilities for simulation results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_timeseries_csv(
    path: Path,
    time_s: np.ndarray,
    base_accel_g: np.ndarray,
    node_names: list[str],
    elem_names: list[str],
    y: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    forces_n: np.ndarray,
    *,
    strain_rate_per_s: np.ndarray | None = None,
    k_dynamic_n_per_m: np.ndarray | None = None,
) -> None:
    """Write simulation timeseries to CSV.

    Optional extra outputs:
      - strain_rate_per_s: (T, N_elem) written as eps_<element>_per_s
      - k_dynamic_n_per_m: (T, N_elem) written as k_<element>_MN_per_m
    """
    headers = ['time_s', 'base_accel_g']
    headers += [f'y_{n}_mm' for n in node_names]
    headers += [f'v_{n}_mps' for n in node_names]
    headers += [f'a_{n}_mps2' for n in node_names]
    headers += [f'F_{e}_kN' for e in elem_names]

    if strain_rate_per_s is not None:
        headers += [f'eps_{e}_per_s' for e in elem_names]
    if k_dynamic_n_per_m is not None:
        headers += [f'k_{e}_MN_per_m' for e in elem_names]

    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(time_s.size):
            row = [f'{time_s[i]:.6f}', f'{base_accel_g[i]:.6f}']
            row += [f'{y[i, j] * 1000.0:.6f}' for j in range(y.shape[1])]
            row += [f'{v[i, j]:.6f}' for j in range(v.shape[1])]
            row += [f'{a[i, j]:.6f}' for j in range(a.shape[1])]
            row += [f'{forces_n[i, j] / 1000.0:.6f}' for j in range(forces_n.shape[1])]

            if strain_rate_per_s is not None:
                row += [f'{strain_rate_per_s[i, j]:.6f}' for j in range(strain_rate_per_s.shape[1])]
            if k_dynamic_n_per_m is not None:
                row += [
                    f'{(k_dynamic_n_per_m[i, j] / 1.0e6):.6f}'
                    for j in range(k_dynamic_n_per_m.shape[1])
                ]

            w.writerow(row)
