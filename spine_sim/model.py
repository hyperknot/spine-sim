from __future__ import annotations

from dataclasses import dataclass

import numpy as np


G0 = 9.80665


def kemper_k_n_per_m(strain_rate_per_s: float) -> float:
    """
    Kemper 2013:
      k = 57.328 * eps_dot + 2019.1
    with:
      k in N/mm, eps_dot in 1/s.

    Convert to N/m by multiplying by 1000.
    """
    k_n_per_mm = 57.328 * float(strain_rate_per_s) + 2019.1
    return 1000.0 * k_n_per_mm


@dataclass
class SpineModel:
    node_names: list[str]
    masses_kg: np.ndarray  # (N,)
    element_names: list[str]  # buttocks + discs
    k0_elem_n_per_m: np.ndarray  # (N_elem,) baseline stiffness (buttocks placeholder at index 0)
    c_elem_ns_per_m: np.ndarray  # (N_elem,)

    # Global disc strain-rate model parameters
    disc_height_m: float
    tension_k_mult: float

    # Buttocks bilinear contact parameters (gap removed; always 0)
    buttocks_k1_n_per_m: float
    buttocks_k2_n_per_m: float
    buttocks_bottom_out_force_n: float
    buttocks_c_ns_per_m: float

    # Kemper scaling and stability knobs
    kemper_normalize_to_eps_per_s: float
    strain_rate_smoothing_tau_s: float
    warn_over_eps_per_s: float

    def size(self) -> int:
        return len(self.node_names)

    def n_elems(self) -> int:
        return len(self.element_names)

    def buttocks_bottom_out_compression_m(self) -> float:
        """
        Compute the compression at which the buttocks bilinear spring changes from k1 to k2,
        based on the configured bottom-out force threshold.
        """
        if self.buttocks_bottom_out_force_n <= 0.0:
            return 0.0
        return self.buttocks_bottom_out_force_n / self.buttocks_k1_n_per_m


@dataclass
class SimulationResult:
    time_s: np.ndarray
    base_accel_g: np.ndarray
    y: np.ndarray  # (T, N)
    v: np.ndarray  # (T, N)
    a: np.ndarray  # (T, N)
    element_forces_n: np.ndarray  # (T, N_elem)

    # Diagnostics (optional but always computed now)
    strain_rate_per_s: np.ndarray  # (T, N_elem)
    k_dynamic_n_per_m: np.ndarray  # (T, N_elem)


def _alpha_lp(dt: float, tau: float) -> float:
    if tau <= 0.0:
        return 1.0
    return float(dt / (tau + dt))


def _disc_strain_rate_per_s(model: SpineModel, relv_mps: float) -> float:
    """
    eps_dot ≈ compression_rate / h0, where compression_rate = max(-relv, 0).
    """
    if model.disc_height_m <= 0.0:
        return 0.0
    compression_rate = max(-float(relv_mps), 0.0)
    return compression_rate / model.disc_height_m


def _disc_k_multiplier(model: SpineModel, eps_dot: float) -> float:
    """
    Apply Kemper as a multiplier on baseline stiffness distribution:
      s(eps) = kK(eps) / kK(eps_norm)

    eps_norm is configurable; default 0 1/s means baseline stiffness corresponds to quasi-static.
    """
    eps_norm = float(model.kemper_normalize_to_eps_per_s)
    k_norm = kemper_k_n_per_m(eps_norm)
    if k_norm <= 0.0:
        return 1.0
    return kemper_k_n_per_m(eps_dot) / k_norm


def _buttocks_force_and_partials(
    model: SpineModel,
    ext_m: float,
    relv_mps: float,
) -> tuple[float, float, float, float]:
    """
    Buttocks element (base-to-pelvis) with:
      - compression-only contact (gap removed; always 0)
      - bilinear spring with bottom-out specified by force threshold
      - damping: contact-only (active while in contact). Damping acts for both
        closing and opening, but the total contact force is clamped to be
        non-negative (the contact cannot "pull" in tension).

    Returns:
      F, dF_dext, dF_drelv, compression_x_m
    """
    # compression x = max(-ext, 0)
    x = max(-float(ext_m), 0.0)
    in_contact = x > 0.0

    if not in_contact:
        return 0.0, 0.0, 0.0, 0.0

    k1 = float(model.buttocks_k1_n_per_m)
    k2 = float(model.buttocks_k2_n_per_m)
    F0 = float(model.buttocks_bottom_out_force_n)

    # Bottom-out compression inferred from bottom-out force threshold
    x0 = 0.0
    if F0 > 0.0:
        x0 = F0 / k1

    # Bilinear spring (hard kink)
    if F0 > 0.0 and x > x0:
        f_s = F0 + k2 * (x - x0)
        dF_dx = k2
    else:
        f_s = k1 * x
        dF_dx = k1

    # Damping: active in contact for both closing and opening
    c = float(model.buttocks_c_ns_per_m)
    f_d = (-c * float(relv_mps)) if (c > 0.0) else 0.0

    f = f_s + f_d

    # Contact cannot pull: clamp to non-negative force
    if f <= 0.0:
        return 0.0, 0.0, 0.0, x

    dF_dext = -dF_dx
    dF_drelv = (-c if c > 0.0 else 0.0)
    return f, dF_dext, dF_drelv, x


def _disc_force_and_partials(
    *,
    k_n_per_m: float,
    c_ns_per_m: float,
    tension_k_mult: float,
    ext_m: float,
    relv_mps: float,
) -> tuple[float, float, float, float]:
    """
    Disc Kelvin-Voigt with:
      - compression stiffness = k
      - tension stiffness = k * tension_k_mult
      - damping symmetric (always active)

    Returns:
      F, dF_dext, dF_drelv, eps_raw_sign_convention_dummy
    """
    ext = float(ext_m)

    if ext < 0.0:
        # compression
        x = -ext
        f_s = float(k_n_per_m) * x
        dF_dext = -float(k_n_per_m)
    else:
        # tension
        k_t = float(k_n_per_m) * float(tension_k_mult)
        f_s = -k_t * ext
        dF_dext = -k_t

    f_d = -float(c_ns_per_m) * float(relv_mps)
    f = f_s + f_d

    dF_drelv = -float(c_ns_per_m)
    return f, dF_dext, dF_drelv, 0.0


def _assemble_forces_and_tangent(
    model: SpineModel,
    y: np.ndarray,
    v: np.ndarray,
    dv_dy_coeff: float,
    *,
    k_disc_step: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble nodal internal force vector f_node (positive upward sign convention as in original code),
    element forces, tangent df/dy, and diagnostic arrays for this evaluation.
    """
    n = model.size()
    ne = model.n_elems()

    f_node = np.zeros(n, dtype=float)
    dfdy = np.zeros((n, n), dtype=float)
    elem_forces = np.zeros(ne, dtype=float)

    # Diagnostics
    eps_raw = np.zeros(ne, dtype=float)
    x_butt = np.zeros(ne, dtype=float)

    # Element 0: buttocks (base to pelvis node 0)
    ext0 = float(y[0] - 0.0)
    relv0 = float(v[0] - 0.0)
    F0, dF_dext0, dF_drelv0, x0 = _buttocks_force_and_partials(model, ext0, relv0)

    elem_forces[0] = F0
    x_butt[0] = x0
    f_node[0] += F0

    k_eff0 = dF_dext0 + dF_drelv0 * dv_dy_coeff
    dfdy[0, 0] += k_eff0

    # Disc elements: between adjacent nodes
    for e in range(1, n):
        i = e - 1
        j = e

        ext = float(y[j] - y[i])
        relv = float(v[j] - v[i])

        eps_raw[e] = _disc_strain_rate_per_s(model, relv)

        F, dF_dext, dF_drelv, _ = _disc_force_and_partials(
            k_n_per_m=float(k_disc_step[e]),
            c_ns_per_m=float(model.c_elem_ns_per_m[e]),
            tension_k_mult=float(model.tension_k_mult),
            ext_m=ext,
            relv_mps=relv,
        )

        elem_forces[e] = F
        f_node[j] += F
        f_node[i] -= F

        k_eff = dF_dext + dF_drelv * dv_dy_coeff

        dfdy[j, j] += k_eff
        dfdy[j, i] -= k_eff
        dfdy[i, j] -= k_eff
        dfdy[i, i] += k_eff

    return f_node, elem_forces, dfdy, eps_raw, x_butt


def initial_state_static(model: SpineModel, base_accel_g0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a static equilibrium under constant base acceleration (typically 0g),
    using a simple Newton solve on internal forces.

    Note: discs are evaluated at their baseline (rate multiplier = 1) since v=0.
    """
    n = model.size()
    y = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)

    masses = np.asarray(model.masses_kg, dtype=float)
    f_base = -masses * (G0 + float(base_accel_g0) * G0)

    # Baseline k for discs (no rate effect needed for static)
    k_disc = model.k0_elem_n_per_m.copy()
    k_disc[0] = 0.0  # buttocks handled separately

    for _ in range(80):
        f_int, _elem, dfdy, _eps_raw, _x = _assemble_forces_and_tangent(
            model,
            y,
            v,
            dv_dy_coeff=0.0,
            k_disc_step=k_disc,
        )
        r = f_int + f_base  # want r=0
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
    """
    Nonlinear Newmark-beta integration with per-timestep frozen disc stiffness.
    """
    n = model.size()
    ne = model.n_elems()

    t = np.asarray(time_s, dtype=float)
    a_g = np.asarray(base_accel_g, dtype=float)

    if t.size < 2:
        raise ValueError('Need at least 2 time samples for integration.')

    dt = float(np.median(np.diff(t)))
    masses = np.asarray(model.masses_kg, dtype=float)
    M = np.diag(masses)

    beta = 0.25
    gamma = 0.5

    a0c = 1.0 / (beta * dt * dt)
    a1c = 1.0 / (beta * dt)
    a2c = (1.0 / (2.0 * beta)) - 1.0
    dv_dy_coeff = gamma / (beta * dt)

    base_accel_mps2 = a_g * G0

    # State histories
    y = np.zeros((t.size, n), dtype=float)
    v = np.zeros((t.size, n), dtype=float)
    a = np.zeros((t.size, n), dtype=float)
    elem_forces = np.zeros((t.size, ne), dtype=float)
    eps_hist = np.zeros((t.size, ne), dtype=float)
    k_hist = np.zeros((t.size, ne), dtype=float)

    y[0] = np.asarray(y0, dtype=float)
    v[0] = np.asarray(v0, dtype=float)

    # Smoothed strain rate state per element (disc elements only; index 0 unused)
    eps_smooth = np.zeros(ne, dtype=float)
    alpha = _alpha_lp(dt, float(model.strain_rate_smoothing_tau_s))

    # Initial baseline disc stiffness (rate multiplier = 1 at eps_norm)
    k_disc_step = model.k0_elem_n_per_m.copy()
    k_disc_step[0] = 0.0

    # Initial accel
    f_base0 = -masses * (G0 + base_accel_mps2[0])
    f_int0, elem0, _dfdy0, eps_raw0, _x0 = _assemble_forces_and_tangent(
        model,
        y[0],
        v[0],
        dv_dy_coeff=0.0,
        k_disc_step=k_disc_step,
    )
    elem_forces[0] = elem0
    eps_hist[0] = eps_raw0
    k_hist[0] = k_disc_step
    a[0] = np.linalg.solve(M, f_int0 + f_base0)

    for k in range(t.size - 1):
        y_n = y[k].copy()
        v_n = v[k].copy()
        a_n = a[k].copy()

        f_base = -masses * (G0 + base_accel_mps2[k + 1])

        # Newmark predictor
        y_guess = y_n + dt * v_n + (0.5 - beta) * dt * dt * a_n

        # We'll freeze disc stiffness for this step after first iteration
        k_disc_frozen = None
        eps_smooth_step = eps_smooth.copy()

        elem_f_last = None
        eps_raw_last = None

        for it in range(max_newton_iter):
            a_guess = a0c * (y_guess - y_n) - a1c * v_n - a2c * a_n
            v_guess = v_n + dt * ((1.0 - gamma) * a_n + gamma * a_guess)

            # Compute strain rate from current kinematics
            eps_raw = np.zeros(ne, dtype=float)
            for e in range(1, n):
                relv = float(v_guess[e] - v_guess[e - 1])
                eps_raw[e] = _disc_strain_rate_per_s(model, relv)

            if it == 0:
                # Update smoothing ONCE per timestep, then freeze k
                eps_smooth_step = eps_smooth + alpha * (eps_raw - eps_smooth)

                # Build frozen k for discs based on smoothed eps
                k_disc_frozen = model.k0_elem_n_per_m.copy()
                k_disc_frozen[0] = 0.0
                for e in range(1, ne):
                    s = _disc_k_multiplier(model, float(eps_smooth_step[e]))
                    k_disc_frozen[e] = float(k_disc_frozen[e]) * float(s)

            assert k_disc_frozen is not None

            f_int, elem_f, dfdy, _eps_raw_eval, _x = _assemble_forces_and_tangent(
                model,
                y_guess,
                v_guess,
                dv_dy_coeff=dv_dy_coeff,
                k_disc_step=k_disc_frozen,
            )
            elem_f_last = elem_f
            eps_raw_last = eps_raw

            r = (M @ a_guess) - f_int - f_base
            J = (M * a0c) - dfdy

            try:
                dy = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dy = np.linalg.lstsq(J, -r, rcond=None)[0]

            y_guess = y_guess + 0.8 * dy

            if float(np.linalg.norm(dy)) < newton_tol:
                break

        # Commit step
        a_next = a0c * (y_guess - y_n) - a1c * v_n - a2c * a_n
        v_next = v_n + dt * ((1.0 - gamma) * a_n + gamma * a_next)
        y_next = y_guess

        y[k + 1] = y_next
        v[k + 1] = v_next
        a[k + 1] = a_next

        if elem_f_last is not None:
            elem_forces[k + 1] = elem_f_last

        # Update smoothing state for next step
        eps_smooth = eps_smooth_step.copy()

        # Store diagnostics (use eps_raw_last from last iteration)
        if eps_raw_last is None:
            eps_raw_last = np.zeros(ne, dtype=float)
        eps_hist[k + 1] = eps_raw_last
        k_hist[k + 1] = k_disc_frozen if k_disc_frozen is not None else k_disc_step

    return SimulationResult(
        time_s=t,
        base_accel_g=a_g,
        y=y,
        v=v,
        a=a,
        element_forces_n=elem_forces,
        strain_rate_per_s=eps_hist,
        k_dynamic_n_per_m=k_hist,
    )
