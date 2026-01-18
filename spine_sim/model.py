from __future__ import annotations

from dataclasses import dataclass

import numpy as np


G0 = 9.80665


@dataclass
class SpineModel:
    node_names: list[str]
    masses_kg: np.ndarray  # shape (N,)
    element_names: list[str]  # buttocks + disc elements
    k_elem: np.ndarray  # shape (N,) includes buttocks at index 0 (equilibrium stiffness)
    c_elem: np.ndarray  # shape (N,) includes buttocks at index 0 (Kelvin-Voigt damper)

    # Nonlinear parameters (cubic stiffening in compression) for the equilibrium spring:
    # Choose k3 in F = k*x + k3*x^3 so that tangent stiffness is k_mult*k at x_ref.
    compression_ref_m: np.ndarray  # shape (N,)
    compression_k_mult: np.ndarray  # shape (N,)
    tension_k_mult: np.ndarray  # shape (N,) (used when not compression_only)

    # Contact-like behavior
    compression_only: np.ndarray  # shape (N,) (buttocks True)
    damping_compression_only: np.ndarray  # shape (N,) (buttocks True)
    gap_m: np.ndarray  # shape (N,) gap before compression engages

    # --- NEW: ZWT/Generalized-Maxwell-like parallel Maxwell branches (rate dependence) ---
    # For each element e and branch b:
    #   s_dot = k_mx[e,b] * x_dot - s / tau[e,b],
    # where x is compression amount (x = max(-(ext+gap), 0)).
    #
    # Force contribution from each branch is +s (compression positive).
    maxwell_k: np.ndarray  # shape (N_elem, B)
    maxwell_tau_s: np.ndarray  # shape (N_elem, B)
    maxwell_compression_only: np.ndarray  # shape (N_elem,)

    def size(self) -> int:
        return len(self.node_names)

    def n_elems(self) -> int:
        return len(self.element_names)

    def n_maxwell(self) -> int:
        if self.maxwell_k.size == 0:
            return 0
        return int(self.maxwell_k.shape[1])

    def build_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearized matrices using only k_elem/c_elem (equilibrium + Kelvin damping).
        Maxwell branches are stateful and are not included in this linearization.
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
    element_forces_n: np.ndarray  # shape (T, N_elem) buttocks + discs (force on UPPER node; buttocks on pelvis)
    maxwell_state_n: np.ndarray  # shape (T, N_elem, B) internal Maxwell forces (N)


def _k3_from_multiplier(k_lin: float, x_ref: float, k_mult: float) -> float:
    """
    Choose k3 in F = k*x + k3*x^3 so that tangent stiffness is k_mult*k at x_ref:
      dF/dx = k + 3*k3*x^2
      dF/dx(x_ref) = k_mult*k
    => k3 = (k_mult - 1)*k / (3*x_ref^2)
    """
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
    Backward-Euler update for Maxwell branch force s.

    We define compression amount x = -ext_eff (in compression).
    Then x_dot = -relv.
      - Closing (relv < 0): x_dot > 0, branch builds compressive force.
      - Opening (relv > 0): x_dot < 0, but we optionally do 'close_only' to avoid creating tension.

    Returns:
      s_new: updated branch force (N, compression positive)
      ds_drelv: partial derivative wrt relv (N per (m/s))
    """
    if k_mx <= 0.0 or tau_s <= 0.0:
        return 0.0, 0.0

    if not in_compression:
        # If the element is not engaged in compression, do not transmit branch force.
        # (We also discard residual branch force to avoid re-contact “memory” artifacts.)
        return 0.0, 0.0

    denom = 1.0 + dt / tau_s

    if close_only and relv_mps >= 0.0:
        # No further compression; only relax internal force.
        s_new = s_prev / denom
        return s_new, 0.0

    # Closing: x_dot = -relv (positive)
    xdot = -relv_mps
    s_new = (s_prev + dt * k_mx * xdot) / denom

    # s_new depends on relv through xdot = -relv
    ds_drelv = -(dt * k_mx) / denom
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
    """
    Returns:
      F_upper (N): force applied to the upper node (compression positive)
      dF_dext: partial wrt extension ext (m)
      dF_drelv: partial wrt relative velocity (m/s)
      s_new: Maxwell internal branch forces for this element (shape (B,))
    """
    k_lin = float(model.k_elem[e_idx])
    c_lin = float(model.c_elem[e_idx])

    gap = float(model.gap_m[e_idx])
    ext_eff = ext_m + gap

    # Compression amount x > 0 when ext_eff < 0
    x = max(-ext_eff, 0.0)
    in_compression = x > 0.0

    k3 = _k3_from_multiplier(
        k_lin,
        float(model.compression_ref_m[e_idx]),
        float(model.compression_k_mult[e_idx]),
    )

    # --- Maxwell branches (rate dependence) ---
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

    # --- Damping handling (Kelvin damper) ---
    use_damp = True
    if model.damping_compression_only[e_idx]:
        # only damp during closing (more compression): rel_v < 0
        use_damp = in_compression and (rel_v_mps < 0.0)

    if in_compression:
        # Equilibrium spring: F = k*x + k3*x^3
        f_s = k_lin * x + k3 * x * x * x

        # Kelvin damping: F_d = -c*rel_v  (rel_v<0 gives +F, resist closing)
        f_d = (-c_lin * rel_v_mps) if use_damp else 0.0

        # Maxwell branches contribute additional compressive force
        f_mx = float(np.sum(s_new))

        f = f_s + f_d + f_mx

        # Partials
        dF_dx = k_lin + 3.0 * k3 * x * x
        dF_dext = -dF_dx  # x = -ext_eff (when in compression)

        dF_drelv = (-(c_lin) if use_damp else 0.0) + dF_drelv_mx
        return f, dF_dext, dF_drelv, s_new

    # Not in compression: tension side
    if model.compression_only[e_idx]:
        # No force if "contact-only".
        return 0.0, 0.0, 0.0, s_new

    k_t = k_lin * float(model.tension_k_mult[e_idx])
    f_s = -k_t * ext_eff
    f_d = -c_lin * rel_v_mps
    # Maxwell is not used in tension in this simplified impact-oriented model.
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
    """
    Build:
      f_node (N): nodal force vector from elements
      elem_forces (N_elem): per-element "force on upper node" (buttocks is pelvis force)
      dfdy (NxN): derivative of nodal forces wrt y (includes velocity coupling via dv_dy_coeff)
      s_next (N_elem, B): next Maxwell branch states (forces), used by the integrator
    """
    n = model.size()
    ne = model.n_elems()
    B = model.n_maxwell()

    f_node = np.zeros(n, dtype=float)
    dfdy = np.zeros((n, n), dtype=float)
    elem_forces = np.zeros(ne, dtype=float)
    s_next = np.zeros((ne, B), dtype=float)

    # Element 0: buttocks (base -> node 0)
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

    # Include v(y) coupling
    k_eff0 = dF_dext0 + dF_drelv0 * dv_dy_coeff
    dfdy[0, 0] += k_eff0

    # Disc elements: e=1..(n-1) between nodes (e-1) and e
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

        # Nodal forces (equal and opposite)
        f_node[j] += Fup
        f_node[i] -= Fup

        k_eff = dF_dext + dF_drelv * dv_dy_coeff

        # Upper row
        dfdy[j, j] += k_eff
        dfdy[j, i] -= k_eff
        # Lower row
        dfdy[i, j] -= k_eff
        dfdy[i, i] += k_eff

    return f_node, elem_forces, dfdy, s_next


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

    # Initial acceleration from dynamics: M a = f_elem + f_base
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

        # Predictor (standard Newmark)
        y_guess = y_n + dt * v_n + (0.5 - beta) * dt * dt * a_n

        s_step = s_prev.copy()

        # Newton solve for y_{n+1}
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
                s_hist[k + 1] = s_step
                break

            if it == max_newton_iter - 1:
                elem_forces[k + 1] = elem_f
                s_hist[k + 1] = s_step

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
        maxwell_state_n=s_hist,
    )


def initial_state_static(model: SpineModel, base_accel_g0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Nonlinear static equilibrium (Maxwell branches assumed relaxed => 0):
      f_elem(y, 0) + f_base = 0
    """
    n = model.size()
    ne = model.n_elems()
    B = model.n_maxwell()

    _, _, K_lin = model.build_matrices()

    f_base = -model.masses_kg * (G0 + base_accel_g0 * G0)

    # Linear guess
    try:
        y = np.linalg.solve(K_lin, f_base)
    except np.linalg.LinAlgError:
        y = np.zeros(n, dtype=float)

    v = np.zeros(n, dtype=float)
    s = np.zeros((ne, B), dtype=float)

    # Newton iterations (static => dv/dy=0)
    for _ in range(60):
        f_elem, _, dfdy, _ = _assemble_element_forces_and_tangent(
            model, y, v, dv_dy_coeff=0.0, dt=1.0, s_prev=s
        )
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

    return y, v, s
