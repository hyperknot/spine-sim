from __future__ import annotations

from dataclasses import dataclass

import numpy as np


G0 = 9.80665


@dataclass
class SpineModel:
    node_names: list[str]
    masses_kg: np.ndarray  # shape (N,)
    element_names: list[str]  # buttocks + disc elements
    k_elem: np.ndarray  # shape (N,) equilibrium stiffness
    c_elem: np.ndarray  # shape (N,) Kelvin-Voigt damper

    # Nonlinear parameters
    compression_ref_m: np.ndarray  # shape (N,)
    compression_k_mult: np.ndarray  # shape (N,)
    tension_k_mult: np.ndarray  # shape (N,)

    # Contact-like behavior
    compression_only: np.ndarray  # shape (N,)
    damping_compression_only: np.ndarray  # shape (N,)
    gap_m: np.ndarray  # shape (N,)

    # Maxwell branches for rate dependence
    maxwell_k: np.ndarray  # shape (N_elem, B)
    maxwell_tau_s: np.ndarray  # shape (N_elem, B)
    maxwell_compression_only: np.ndarray  # shape (N_elem,)

    # Optional explicit polynomial terms for compression:
    # F_eq(x) = k*x + k2*x^2 + k3*x^3, for x >= 0 (compression)
    # Units:
    #   k2: N/m^2
    #   k3: N/m^3
    poly_k2: np.ndarray | None = None  # shape (N_elem,)
    poly_k3: np.ndarray | None = None  # shape (N_elem,)

    # Optional compression limit / stop stiffness (bottom-out)
    compression_limit_m: np.ndarray | None = None  # shape (N_elem,)
    compression_stop_k: np.ndarray | None = None  # shape (N_elem,)

    def size(self) -> int:
        return len(self.node_names)

    def n_elems(self) -> int:
        return len(self.element_names)

    def n_maxwell(self) -> int:
        if self.maxwell_k.size == 0:
            return 0
        return int(self.maxwell_k.shape[1])

    def build_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearized matrices using only k_elem/c_elem."""
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
            i = e - 1
            j = e
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
    element_forces_n: np.ndarray  # shape (T, N_elem)
    maxwell_state_n: np.ndarray  # shape (T, N_elem, B)


def _k3_from_multiplier(k_lin: float, x_ref: float, k_mult: float) -> float:
    if x_ref <= 0.0 or k_mult <= 1.0:
        return 0.0
    return (k_mult - 1.0) * k_lin / (3.0 * x_ref * x_ref)


def _maxwell_branch_update(
    *,
    k_mx: float,
    tau_s: float,
    dt: float,
    relv_mps: float,
    s_prev: float,
    in_compression: bool,
    close_only: bool = True,
) -> tuple[float, float]:
    """
    Discrete Maxwell branch in force form:

        dF/dt + F/tau = k * xdot,

    where xdot is compression rate. We define compression x = max(-(ext+gap), 0),
    so xdot = -relv.

    If close_only=True, we clamp branch force to be >= 0 (no tensile Maxwell force).
    """
    if k_mx <= 0.0 or tau_s <= 0.0:
        return 0.0, 0.0

    if not in_compression:
        return 0.0, 0.0

    denom = 1.0 + dt / tau_s

    xdot = -relv_mps
    s_new = (s_prev + dt * k_mx * xdot) / denom

    ds_drelv = -(dt * k_mx) / denom

    # Optional compression-only behavior: do not allow branch to go tensile.
    if close_only and s_new < 0.0:
        s_new = 0.0
        ds_drelv = 0.0

    return s_new, ds_drelv


def _element_force_upper_and_partials(
    model: SpineModel,
    e_idx: int,
    ext_m: float,
    rel_v_mps: float,
    *,
    dt: float,
    s_prev: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    k_lin = float(model.k_elem[e_idx])
    c_lin = float(model.c_elem[e_idx])

    gap = float(model.gap_m[e_idx])
    ext_eff = ext_m + gap

    x = max(-ext_eff, 0.0)
    in_compression = x > 0.0

    # Use explicit ZWT-style polynomial coefficients if provided;
    # otherwise fall back to the older "multiplier at reference compression" cubic.
    k2 = 0.0
    if model.poly_k2 is not None:
        k2 = float(model.poly_k2[e_idx])

    if model.poly_k3 is not None:
        k3 = float(model.poly_k3[e_idx])
    else:
        k3 = _k3_from_multiplier(
            k_lin,
            float(model.compression_ref_m[e_idx]),
            float(model.compression_k_mult[e_idx]),
        )

    limit_m = None
    stop_k = 0.0
    if model.compression_limit_m is not None:
        limit_m = float(model.compression_limit_m[e_idx])
    if model.compression_stop_k is not None:
        stop_k = float(model.compression_stop_k[e_idx])

    B = model.n_maxwell()
    s_new = np.zeros(B, dtype=float)
    dF_drelv_mx = 0.0

    if B > 0:
        for b in range(B):
            s_b, ds_drelv_b = _maxwell_branch_update(
                k_mx=float(model.maxwell_k[e_idx, b]),
                tau_s=float(model.maxwell_tau_s[e_idx, b]),
                dt=dt,
                relv_mps=rel_v_mps,
                s_prev=float(s_prev[b]) if s_prev.size else 0.0,
                in_compression=in_compression,
                close_only=bool(model.maxwell_compression_only[e_idx]),
            )
            s_new[b] = s_b
            dF_drelv_mx += ds_drelv_b

    use_damp = True
    if model.damping_compression_only[e_idx]:
        use_damp = in_compression and (rel_v_mps < 0.0)

    if in_compression:
        f_s = k_lin * x + k2 * x * x + k3 * x * x * x
        dF_dx = k_lin + 2.0 * k2 * x + 3.0 * k3 * x * x

        if limit_m is not None and limit_m > 0.0 and stop_k > 0.0 and x > limit_m:
            x_excess = x - limit_m
            f_s += stop_k * x_excess
            dF_dx += stop_k

        f_d = (-c_lin * rel_v_mps) if use_damp else 0.0
        f_mx = float(np.sum(s_new))
        f = f_s + f_d + f_mx

        dF_dext = -dF_dx
        dF_drelv = (-(c_lin) if use_damp else 0.0) + dF_drelv_mx
        return f, dF_dext, dF_drelv, s_new

    if model.compression_only[e_idx]:
        return 0.0, 0.0, 0.0, s_new

    k_t = k_lin * float(model.tension_k_mult[e_idx])
    f_s = -k_t * ext_eff
    f_d = -c_lin * rel_v_mps
    f = f_s + f_d

    dF_dext = -k_t
    dF_drelv = -c_lin
    return f, dF_dext, dF_drelv, s_new


def _assemble_element_forces_and_tangent(
    model: SpineModel,
    y: np.ndarray,
    v: np.ndarray,
    dv_dy_coeff: float,
    *,
    dt: float,
    s_prev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = model.size()
    ne = model.n_elems()
    B = model.n_maxwell()

    f_node = np.zeros(n, dtype=float)
    dfdy = np.zeros((n, n), dtype=float)
    elem_forces = np.zeros(ne, dtype=float)
    s_next = np.zeros((ne, B), dtype=float)

    # Element 0: buttocks
    ext0 = y[0] - 0.0
    relv0 = v[0] - 0.0
    F0, dF_dext0, dF_drelv0, s0 = _element_force_upper_and_partials(
        model,
        0,
        ext0,
        relv0,
        dt=dt,
        s_prev=s_prev[0] if s_prev.size else np.zeros(B),
    )
    s_next[0] = s0
    elem_forces[0] = F0
    f_node[0] += F0

    k_eff0 = dF_dext0 + dF_drelv0 * dv_dy_coeff
    dfdy[0, 0] += k_eff0

    # Disc elements
    for e in range(1, n):
        i = e - 1
        j = e
        ext = y[j] - y[i]
        relv = v[j] - v[i]

        Fup, dF_dext, dF_drelv, s_e = _element_force_upper_and_partials(
            model,
            e,
            ext,
            relv,
            dt=dt,
            s_prev=s_prev[e] if s_prev.size else np.zeros(B),
        )
        s_next[e] = s_e
        elem_forces[e] = Fup

        f_node[j] += Fup
        f_node[i] -= Fup

        k_eff = dF_dext + dF_drelv * dv_dy_coeff

        dfdy[j, j] += k_eff
        dfdy[j, i] -= k_eff
        dfdy[i, j] -= k_eff
        dfdy[i, i] += k_eff

    return f_node, elem_forces, dfdy, s_next


# newmark_nonlinear and initial_state_static unchanged below
def newmark_nonlinear(
    model: SpineModel,
    time_s: np.ndarray,
    base_accel_g: np.ndarray,
    y0: np.ndarray,
    v0: np.ndarray,
    s0: np.ndarray | None = None,
    *,
    max_newton_iter: int = 25,
    newton_tol: float = 1e-9,
) -> SimulationResult:
    n = model.size()
    ne = model.n_elems()
    B = model.n_maxwell()

    t = np.asarray(time_s, dtype=float)
    base_accel_g = np.asarray(base_accel_g, dtype=float)

    if t.size < 2:
        raise ValueError("Need at least 2 time samples for integration.")

    dt = float(np.median(np.diff(t)))
    base_accel_mps2 = base_accel_g * G0

    M = np.diag(model.masses_kg)

    f_base = -model.masses_kg[:, None] * (G0 + base_accel_mps2[None, :])

    beta = 0.25
    gamma = 0.5

    a0c = 1.0 / (beta * dt * dt)
    a1c = 1.0 / (beta * dt)
    a2c = (1.0 / (2.0 * beta)) - 1.0
    dv_dy_coeff = gamma / (beta * dt)

    y = np.zeros((t.size, n), dtype=float)
    v = np.zeros((t.size, n), dtype=float)
    a = np.zeros((t.size, n), dtype=float)
    elem_forces = np.zeros((t.size, ne), dtype=float)

    s_hist = np.zeros((t.size, ne, B), dtype=float)
    if s0 is None:
        s0 = np.zeros((ne, B), dtype=float)
    else:
        s0 = np.asarray(s0, dtype=float)
        if s0.shape != (ne, B):
            raise ValueError(f"s0 must have shape {(ne, B)}, got {s0.shape}.")

    y[0] = y0
    v[0] = v0
    s_hist[0] = s0

    f_elem0, elem0, _, s_next0 = _assemble_element_forces_and_tangent(
        model,
        y0,
        v0,
        dv_dy_coeff=0.0,
        dt=dt,
        s_prev=s_hist[0],
    )
    elem_forces[0] = elem0
    s_hist[0] = s_next0
    a[0] = np.linalg.solve(M, f_elem0 + f_base[:, 0])

    for k in range(t.size - 1):
        y_n = y[k].copy()
        v_n = v[k].copy()
        a_n = a[k].copy()
        s_prev = s_hist[k].copy()

        y_guess = y_n + dt * v_n + (0.5 - beta) * dt * dt * a_n

        s_step = s_prev.copy()

        for it in range(max_newton_iter):
            a_guess = a0c * (y_guess - y_n) - a1c * v_n - a2c * a_n
            v_guess = v_n + dt * ((1.0 - gamma) * a_n + gamma * a_guess)

            f_elem, elem_f, dfdy, s_next = _assemble_element_forces_and_tangent(
                model,
                y_guess,
                v_guess,
                dv_dy_coeff=dv_dy_coeff,
                dt=dt,
                s_prev=s_prev,
            )
            s_step = s_next

            r = (M @ a_guess) - f_elem - f_base[:, k + 1]

            J = (M * a0c) - dfdy

            try:
                dy = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dy = np.linalg.lstsq(J, -r, rcond=None)[0]

            y_guess = y_guess + 0.8 * dy

            if float(np.linalg.norm(dy)) < newton_tol:
                elem_forces[k + 1] = elem_f
                s_hist[k + 1] = s_step
                break

            if it == max_newton_iter - 1:
                elem_forces[k + 1] = elem_f
                s_hist[k + 1] = s_step

        a[k + 1] = a0c * (y_guess - y_n) - a1c * v_n - a2c * a_n
        v[k + 1] = v_n + dt * ((1.0 - gamma) * a_n + gamma * a[k + 1])
        y[k + 1] = y_guess

    return SimulationResult(
        time_s=t,
        base_accel_g=base_accel_g,
        y=y,
        v=v,
        a=a,
        element_forces_n=elem_forces,
        maxwell_state_n=s_hist,
    )


def initial_state_static(model: SpineModel, base_accel_g0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonlinear static equilibrium (Maxwell branches assumed relaxed)."""
    n = model.size()
    ne = model.n_elems()
    B = model.n_maxwell()

    _, _, K_lin = model.build_matrices()

    f_base = -model.masses_kg * (G0 + base_accel_g0 * G0)

    try:
        y = np.linalg.solve(K_lin, f_base)
    except np.linalg.LinAlgError:
        y = np.zeros(n, dtype=float)

    v = np.zeros(n, dtype=float)
    s = np.zeros((ne, B), dtype=float)

    for _ in range(60):
        f_elem, _, dfdy, _ = _assemble_element_forces_and_tangent(
            model, y, v, dv_dy_coeff=0.0, dt=1.0, s_prev=s
        )
        r = f_elem + f_base

        if float(np.linalg.norm(r)) < 1e-8:
            break

        J = dfdy
        try:
            dy = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dy = np.linalg.lstsq(J, -r, rcond=None)[0]

        y = y + 0.8 * dy
        if float(np.linalg.norm(dy)) < 1e-10:
            break

    return y, v, s
