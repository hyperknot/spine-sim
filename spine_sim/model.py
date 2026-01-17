from __future__ import annotations

from dataclasses import dataclass

import numpy as np


G0 = 9.80665


@dataclass
class SpineModel:
    node_names: list[str]
    masses_kg: np.ndarray  # shape (N,)
    element_names: list[str]  # buttocks + disc elements
    k_elem: np.ndarray  # shape (N,) includes buttocks at index 0 (linear baseline)
    c_elem: np.ndarray  # shape (N,) includes buttocks at index 0 (linear baseline)

    # Nonlinear parameters (simple cubic stiffening in compression)
    # For each element e:
    # - if compression_ref_m[e] <= 0 or compression_k_mult[e] <= 1 => nonlinear disabled for that element
    compression_ref_m: np.ndarray  # shape (N,)
    compression_k_mult: np.ndarray  # shape (N,)
    tension_k_mult: np.ndarray  # shape (N,) (for discs; buttocks typically 0 or unused)

    # Contact-like behavior
    compression_only: np.ndarray  # shape (N,) (buttocks True)
    damping_compression_only: np.ndarray  # shape (N,) (buttocks True)
    gap_m: np.ndarray  # shape (N,) gap before compression engages (usually 0)

    def size(self) -> int:
        return len(self.node_names)

    def build_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearized matrices using k_elem/c_elem.
        Useful for initial guesses and diagnostics.
        """
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
    element_forces_n: np.ndarray  # shape (T, N) buttocks + discs (force on UPPER node; buttocks on pelvis)


def _k3_from_multiplier(k_lin: float, x_ref: float, k_mult: float) -> float:
    """
    Choose k3 in F = k*x + k3*x^3 so that tangent stiffness doubles (or more)
    at x_ref:
        dF/dx = k + 3*k3*x^2
        dF/dx(x_ref) = k_mult * k
    => k3 = (k_mult - 1)*k / (3*x_ref^2)
    """
    if x_ref <= 0.0 or k_mult <= 1.0:
        return 0.0
    return (k_mult - 1.0) * k_lin / (3.0 * x_ref * x_ref)


def _element_force_upper_and_partials(
    model: SpineModel,
    e_idx: int,
    ext_m: float,
    rel_v_mps: float,
) -> tuple[float, float, float]:
    """
    Returns:
      F_upper (N): force applied to the upper node (compression positive)
      dF_dext: partial wrt extension ext (m)
      dF_drelv: partial wrt relative velocity (m/s)
    """
    k_lin = float(model.k_elem[e_idx])
    c_lin = float(model.c_elem[e_idx])

    gap = float(model.gap_m[e_idx])
    ext_eff = ext_m + gap

    # Compression amount x > 0 when ext_eff < 0
    x = max(-ext_eff, 0.0)

    k3 = _k3_from_multiplier(k_lin, float(model.compression_ref_m[e_idx]), float(model.compression_k_mult[e_idx]))

    in_compression = x > 0.0

    # Damping handling
    use_damp = True
    if model.damping_compression_only[e_idx]:
        # only damp during closing (more compression): rel_v < 0
        use_damp = in_compression and (rel_v_mps < 0.0)

    if in_compression:
        # Spring: F = k*x + k3*x^3
        f_s = k_lin * x + k3 * x * x * x

        # Damping: F_d = -c*rel_v  (so rel_v<0 gives +F, i.e. resist closing)
        f_d = (-c_lin * rel_v_mps) if use_damp else 0.0

        f = f_s + f_d

        # Partials
        dF_dx = k_lin + 3.0 * k3 * x * x
        dF_dext = -dF_dx  # x = -ext_eff (when in compression)
        dF_drelv = (-c_lin) if use_damp else 0.0
        return f, dF_dext, dF_drelv

    # Not in compression: tension side
    if model.compression_only[e_idx]:
        return 0.0, 0.0, 0.0

    k_t = k_lin * float(model.tension_k_mult[e_idx])
    f_s = -k_t * ext_eff
    f_d = -c_lin * rel_v_mps
    f = f_s + f_d

    dF_dext = -k_t
    dF_drelv = -c_lin
    return f, dF_dext, dF_drelv


def _assemble_element_forces_and_tangent(
    model: SpineModel,
    y: np.ndarray,
    v: np.ndarray,
    dv_dy_coeff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build:
      f_elem (N): nodal force vector (actual forces from elements on nodes)
      elem_forces (N): per-element "force on upper node" for output (buttocks is pelvis force)
      dfdy (NxN): derivative of nodal forces wrt y (includes velocity coupling via dv_dy_coeff)
                 where v = v_const + dv_dy_coeff * y in the Newton step context.
    """
    n = model.size()
    f = np.zeros(n, dtype=float)
    dfdy = np.zeros((n, n), dtype=float)
    elem_forces = np.zeros(n, dtype=float)

    # Element 0: buttocks (base -> node 0)
    ext0 = y[0] - 0.0
    relv0 = v[0] - 0.0
    F0, dF_dext0, dF_drelv0 = _element_force_upper_and_partials(model, 0, ext0, relv0)
    elem_forces[0] = F0
    f[0] += F0

    # Include v(y) coupling
    k_eff0 = dF_dext0 + dF_drelv0 * dv_dy_coeff
    dfdy[0, 0] += k_eff0

    # Disc elements: e=1..n-1 between nodes (e-1) and e
    for e in range(1, n):
        i = e - 1
        j = e
        ext = y[j] - y[i]
        relv = v[j] - v[i]

        Fup, dF_dext, dF_drelv = _element_force_upper_and_partials(model, e, ext, relv)
        elem_forces[e] = Fup

        # Nodal forces (equal and opposite)
        f[j] += Fup
        f[i] -= Fup

        k_eff = dF_dext + dF_drelv * dv_dy_coeff

        # Upper row
        dfdy[j, j] += k_eff
        dfdy[j, i] -= k_eff
        # Lower row
        dfdy[i, j] -= k_eff
        dfdy[i, i] += k_eff

    return f, elem_forces, dfdy


def newmark_nonlinear(
    model: SpineModel,
    time_s: np.ndarray,
    base_accel_g: np.ndarray,
    y0: np.ndarray,
    v0: np.ndarray,
    *,
    max_newton_iter: int = 25,
    newton_tol: float = 1e-9,
) -> SimulationResult:
    n = model.size()
    t = np.asarray(time_s, dtype=float)
    base_accel_g = np.asarray(base_accel_g, dtype=float)

    if t.size < 2:
        raise ValueError("Need at least 2 time samples for integration.")

    dt = float(np.median(np.diff(t)))
    base_accel_mps2 = base_accel_g * G0

    M = np.diag(model.masses_kg)

    # External forcing: -m*(g + a_base)
    f_base = -model.masses_kg[:, None] * (G0 + base_accel_mps2[None, :])

    # Newmark constants (average acceleration)
    beta = 0.25
    gamma = 0.5

    a0c = 1.0 / (beta * dt * dt)
    a1c = 1.0 / (beta * dt)
    a2c = (1.0 / (2.0 * beta)) - 1.0
    dv_dy_coeff = gamma / (beta * dt)  # = a3c

    y = np.zeros((t.size, n), dtype=float)
    v = np.zeros((t.size, n), dtype=float)
    a = np.zeros((t.size, n), dtype=float)
    elem_forces = np.zeros((t.size, n), dtype=float)

    y[0] = y0
    v[0] = v0

    # Initial acceleration from dynamics: M a = f_elem + f_base
    f_elem0, elem0, _ = _assemble_element_forces_and_tangent(model, y0, v0, dv_dy_coeff=0.0)
    elem_forces[0] = elem0
    a[0] = np.linalg.solve(M, f_elem0 + f_base[:, 0])

    for k in range(t.size - 1):
        y_n = y[k].copy()
        v_n = v[k].copy()
        a_n = a[k].copy()

        # Predictor (standard Newmark)
        y_guess = y_n + dt * v_n + (0.5 - beta) * dt * dt * a_n

        # Newton solve for y_{n+1}
        for it in range(max_newton_iter):
            a_guess = a0c * (y_guess - y_n) - a1c * v_n - a2c * a_n
            v_guess = v_n + dt * ((1.0 - gamma) * a_n + gamma * a_guess)

            f_elem, elem_f, dfdy = _assemble_element_forces_and_tangent(
                model, y_guess, v_guess, dv_dy_coeff=dv_dy_coeff
            )

            # Residual: M a - f_elem - f_base = 0
            r = (M @ a_guess) - f_elem - f_base[:, k + 1]

            # Jacobian: d/dy (M a) - df_elem/dy
            J = (M * a0c) - dfdy

            try:
                dy = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dy = np.linalg.lstsq(J, -r, rcond=None)[0]

            y_guess = y_guess + 0.8 * dy  # mild damping helps near contact transitions

            if float(np.linalg.norm(dy)) < newton_tol:
                elem_forces[k + 1] = elem_f
                break

            if it == max_newton_iter - 1:
                # keep last iterate; still store element forces for output
                elem_forces[k + 1] = elem_f

        # Finalize step
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
    )


def initial_state_static(model: SpineModel, base_accel_g0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Nonlinear static equilibrium:
      f_elem(y, 0) + f_base = 0
    """
    n = model.size()
    M, _, K_lin = model.build_matrices()

    f_base = -model.masses_kg * (G0 + base_accel_g0 * G0)

    # Linear guess
    try:
        y = np.linalg.solve(K_lin, f_base)
    except np.linalg.LinAlgError:
        y = np.zeros(n, dtype=float)

    v = np.zeros(n, dtype=float)

    # Newton iterations (static => dv/dy=0)
    for _ in range(60):
        f_elem, _, dfdy = _assemble_element_forces_and_tangent(model, y, v, dv_dy_coeff=0.0)
        r = f_elem + f_base  # want 0

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

    return y, v
