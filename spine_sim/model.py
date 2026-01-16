from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


G0 = 9.80665


@dataclass
class SpineModel:
    node_names: list[str]
    masses_kg: np.ndarray  # shape (N,)
    element_names: list[str]  # buttocks + disc elements
    k_elem: np.ndarray  # shape (N,) includes buttocks at index 0
    c_elem: np.ndarray  # shape (N,) includes buttocks at index 0

    def size(self) -> int:
        return len(self.node_names)

    def build_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.size()
        M = np.diag(self.masses_kg)
        K = np.zeros((n, n), dtype=float)
        C = np.zeros((n, n), dtype=float)

        # Buttocks element (base to pelvis, node 0)
        K[0, 0] += self.k_elem[0]
        C[0, 0] += self.c_elem[0]

        # Disc elements between nodes
        for e in range(1, n):
            k = self.k_elem[e]
            c = self.c_elem[e]
            i = e - 1  # lower
            j = e  # upper
            K[i, i] += k
            K[j, j] += k
            K[i, j] -= k
            K[j, i] -= k

            C[i, i] += c
            C[j, j] += c
            C[i, j] -= c
            C[j, i] -= c

        return M, C, K


@dataclass
class SimulationResult:
    time_s: np.ndarray
    base_accel_g: np.ndarray
    y: np.ndarray  # shape (T, N)
    v: np.ndarray  # shape (T, N)
    a: np.ndarray  # shape (T, N)
    element_forces_n: np.ndarray  # shape (T, N) buttocks + discs


def newmark_linear(
    model: SpineModel,
    time_s: np.ndarray,
    base_accel_g: np.ndarray,
    y0: np.ndarray,
    v0: np.ndarray,
) -> SimulationResult:
    n = model.size()
    t = time_s
    dt = float(np.median(np.diff(t)))
    base_accel_mps2 = base_accel_g * G0

    M, C, K = model.build_matrices()
    M_inv = np.linalg.inv(M)

    # External forcing: -M*(g + a_base)
    forcing = -model.masses_kg[:, None] * (G0 + base_accel_mps2[None, :])

    # Newmark constants
    beta = 0.25
    gamma = 0.5

    a0c = 1.0 / (beta * dt * dt)
    a1c = 1.0 / (beta * dt)
    a2c = (1.0 / (2.0 * beta)) - 1.0
    a3c = gamma / (beta * dt)
    a4c = (gamma / beta) - 1.0
    a5c = dt * ((gamma / (2.0 * beta)) - 1.0)

    # Effective stiffness
    K_eff = K + a3c * C + a0c * M
    K_eff_inv = np.linalg.inv(K_eff)

    y = np.zeros((t.size, n), dtype=float)
    v = np.zeros((t.size, n), dtype=float)
    a = np.zeros((t.size, n), dtype=float)

    y[0] = y0
    v[0] = v0
    a[0] = M_inv @ (forcing[:, 0] - C @ v0 - K @ y0)

    for i in range(t.size - 1):
        f_eff = (
            forcing[:, i + 1]
            + M @ (a0c * y[i] + a1c * v[i] + a2c * a[i])
            + C @ (a3c * y[i] + a4c * v[i] + a5c * a[i])
        )
        y[i + 1] = K_eff_inv @ f_eff
        a[i + 1] = a0c * (y[i + 1] - y[i]) - a1c * v[i] - a2c * a[i]
        v[i + 1] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a[i + 1])

    # Element forces (compression positive)
    elem_forces = np.zeros((t.size, n), dtype=float)

    # Buttocks element
    elem_forces[:, 0] = -model.k_elem[0] * y[:, 0] - model.c_elem[0] * v[:, 0]

    # Disc elements
    for e in range(1, n):
        lower = e - 1
        upper = e
        dy = y[:, upper] - y[:, lower]
        dv = v[:, upper] - v[:, lower]
        elem_forces[:, e] = -model.k_elem[e] * dy - model.c_elem[e] * dv

    return SimulationResult(
        time_s=t,
        base_accel_g=base_accel_g,
        y=y,
        v=v,
        a=a,
        element_forces_n=elem_forces,
    )


def initial_state_static(model: SpineModel, base_accel_g0: float) -> tuple[np.ndarray, np.ndarray]:
    n = model.size()
    M, _, K = model.build_matrices()

    # Static equilibrium: K*y0 = -M*(g + a_base)
    rhs = -model.masses_kg * (G0 + base_accel_g0 * G0)
    y0 = np.linalg.solve(K, rhs)
    v0 = np.zeros(n, dtype=float)
    return y0, v0
