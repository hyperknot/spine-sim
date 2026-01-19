"""Output utilities for simulation results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_timeseries_csv(
    path: Path, time_s: np.ndarray, base_accel_g: np.ndarray,
    node_names: list[str], elem_names: list[str],
    y: np.ndarray, v: np.ndarray, a: np.ndarray, forces_n: np.ndarray,
) -> None:
    """Write simulation timeseries to CSV.

    Args:
        path: Output file path.
        time_s: Time array in seconds.
        base_accel_g: Base acceleration in g.
        node_names: List of node names.
        elem_names: List of element names.
        y: Displacement array (n_steps, n_nodes) in meters.
        v: Velocity array (n_steps, n_nodes) in m/s.
        a: Acceleration array (n_steps, n_nodes) in m/s^2.
        forces_n: Element forces array (n_steps, n_elems) in Newtons.
    """
    headers = ["time_s", "base_accel_g"]
    headers += [f"y_{n}_mm" for n in node_names]
    headers += [f"v_{n}_mps" for n in node_names]
    headers += [f"a_{n}_mps2" for n in node_names]
    headers += [f"F_{e}_kN" for e in elem_names]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(time_s.size):
            row = [f"{time_s[i]:.6f}", f"{base_accel_g[i]:.6f}"]
            row += [f"{y[i, j] * 1000.0:.6f}" for j in range(y.shape[1])]
            row += [f"{v[i, j]:.6f}" for j in range(v.shape[1])]
            row += [f"{a[i, j]:.6f}" for j in range(a.shape[1])]
            row += [f"{forces_n[i, j] / 1000.0:.6f}" for j in range(forces_n.shape[1])]
            w.writerow(row)
