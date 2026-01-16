#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

r"""
1D Human Spine + Buttocks + Viscera Simulation (base excitation)
===============================================================

Purpose
-------
This script implements a *simple but structured* 1D axial (vertical) biodynamic model
for evaluating spinal loads under caudo-cephalad (vertical) impacts, aimed at
paragliding harness back protectors and related underbody-like loading studies.

It is intentionally NOT a full finite element model. It is a multi-segment lumped
mass-spring-damper chain (spine + viscera) driven by prescribed seat/base acceleration.

Primary objective (this version)
--------------------------------
Study *rate-of-onset / jerk* effects (G/s) on spine loading, while keeping reference
inputs fixed (impact speed and available stroke) to match the user's reference table.

A key lesson from Raj (2019) and Yoganandan (2021) is that "pulse shape matters":
rise time / onset rate can change internal load transfer and injury risk, even at
similar gross conditions.

However, in a purely linear 1D axial chain, the *peak compressive force* at a disc can
be relatively insensitive to sub-millisecond changes in input shape because the spine
system acts as a low-pass filter. Therefore, this version adds *additional, jerk-sensitive
diagnostics* so that the simulation can report *what jerk changes*, even when a single
peak-force metric is flat:

Added jerk-sensitive outputs (impact window)
--------------------------------------------
For T12–L1 and for whole-spine summaries, we now report, during the impact window:

- Peak compressive force: max(F_comp).
- Peak compressive loading rate: max(dF_comp/dt).
- Peak compressive "early-window" force and loading rate (default first 5 ms of impact).
  *This is where rise-time differences are most visible.*
- Peak absolute accelerations and jerks (da/dt) at key nodes (Pelvis, T12, Head).
- A 1D "Raj-inspired" axial-only injury proxy:
    I_axial(t) = sum over discs of max(0, compression_deformation),
  reported as peak(mm) during the impact window and during the early-window.

We still keep the full-duration metrics (for later comparison), but note:
it is physically possible for the maximum internal compressive force to occur after the
input pulse ends (ring-down overshoot). This is not "impossible" in a dynamic system.

Reference-values matching (analytic mode)
-----------------------------------------
Analytic mode matches the user's reference input semantics:

- impact speed: 5.7 m/s (explicit input)
- stroke: 10.5 cm = 0.105 m (explicit input)
- max allowed G: e.g. 16/24/32 (explicit input)
- total mass: 75 kg (default)

Given (v0, stroke, max_g), we compute the minimum jerk (G/s) required to stop within the
stroke, using the jerk-limited triangular/trapezoidal profile math (physics.ts port).

Time step and automatic dt refinement (critical)
------------------------------------------------
In the 16 G / 0.105 m / 5.7 m/s case, the minimum jerk is ~31k G/s and the time-to-peak
is ~0.5 ms. A dt of 1.0 ms cannot resolve that ramp.

This version implements *automatic dt refinement* for analytic mode, with improvements:

1) Input peak capture:
   sampled_peak_g ~= analytic_peak_g within tolerance.

2) Output convergence checks (impact window):
   peak force, peak dF/dt, and early-window peak force and early-window peak dF/dt
   must converge under dt halving.

3) Minimum samples per ramp-up requirement:
   dt must satisfy t_ramp / dt >= N (default N=20).
   This forces dt to be small enough to resolve the jerk-limited rise.

Auto-dt history is stored in sim.json and also printed in --debug output.

Coordinate and sign convention
------------------------------
Axis: +z is upward.

Seat/base acceleration a_seat(t) is inertial acceleration (m/s^2) of the seat interface:
- -1 G during freefall (a_seat = -g).
- 0 G at rest (a_seat = 0).
- positive during impact deceleration.

The solver uses relative coordinates:
  u = x - x_seat,
so the seat reference is 0 in the equations of motion.

Forcing in relative coordinates:
  M u_ddot + C u_dot + K u = f_ext(t),
with:
  f_ext(t) = m * (-g - a_seat(t)).

Seat coupling (pelvis-seat)
---------------------------
We use two unilateral elements in parallel:
- Buttocks tissue: compression-only.
- Harness strap: tension-only.

Important fix:
- The pelvis-seat coupling is always active through exactly one path (buttocks OR strap).
- We do NOT allow a "none engaged" state by default because it can artificially decouple the
  pelvis from the input and suppress jerk transmission.
- We enforce mutual exclusivity with a small deadband and hysteresis to prevent chatter.

Limitations
-----------
- 1D axial only (no shear, no rotation, no posture geometry).
- Linear springs/dampers (no explicit viscoelastic constitutive law beyond Kelvin-Voigt).
- Parameter values were originally tuned/used for vibration contexts in the cited literature;
  millisecond impacts can excite frequency ranges beyond those validation regimes.

References (as provided)
------------------------
- Kitazaki & Griffin (1997). Modal analysis of whole-body vertical vibration.
- Raj & Krishnapillai (2019). Improved spinal injury parameter model for underbody impulses.
- Yoganandan et al. (2021). Thoracolumbar tolerance to caudo-cephalad loading (pulse shape matters).

Usage examples
--------------
Analytic (reference-like inputs):
  uv run spine-sim.py --mode analytic --impact-speed-mps 5.7 --stroke-m 0.105 --max-g 16 --debug

CSV mode:
  uv run spine-sim.py --mode csv --csv path/to/file.csv --resample-dt 0.0001 --debug
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib
import numpy as np


matplotlib.use('Agg')  # ensure no GUI windows/dialogs
import matplotlib.pyplot as plt


# -------------------------
# Constants and utilities
# -------------------------

G = 9.80665  # m/s^2

EPS = 1e-12


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def lin_interp_resample(t: np.ndarray, y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample (t,y) to uniform dt using linear interpolation.
    Assumes t is strictly increasing.
    """
    t0 = float(t[0])
    t1 = float(t[-1])
    n = int(math.floor((t1 - t0) / dt)) + 1
    t_u = t0 + dt * np.arange(n, dtype=np.float64)
    y_u = np.interp(t_u, t, y).astype(np.float64)
    return t_u, y_u


def trapz_integrate(
    t: np.ndarray, a: np.ndarray, v0: float = 0.0, x0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate acceleration to get velocity and position using trapezoidal integration.
    Used for CSV visualization. The solver uses a(t) directly (base excitation).
    """
    v = np.zeros_like(a, dtype=np.float64)
    x = np.zeros_like(a, dtype=np.float64)
    v[0] = v0
    x[0] = x0
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        v[i] = v[i - 1] + 0.5 * (a[i - 1] + a[i]) * dt
        x[i] = x[i - 1] + 0.5 * (v[i - 1] + v[i]) * dt
    return v, x


def finite_difference(y: np.ndarray, dt: float) -> np.ndarray:
    """
    Simple first-order finite difference aligned to y[1:]:
      dy[i] ~ (y[i] - y[i-1]) / dt, for i>=1, dy[0]=0
    """
    dy = np.zeros_like(y, dtype=np.float64)
    if len(y) <= 1 or dt <= 0:
        return dy
    dy[1:] = (y[1:] - y[:-1]) / dt
    return dy


# -----------------------------------
# Jerk-limited pulse (physics.ts port)
# -----------------------------------


@dataclass(frozen=True)
class ProfileResult:
    profile_type: Literal['triangular', 'trapezoidal']
    v0: float
    jerk_g: float
    max_g: float
    jerk: float  # m/s^3
    peak_a: float  # m/s^2
    t1: float
    t2: float
    total_time: float
    stop_distance: float


def compute_profile(v0: float, jerk_g: float, max_g: float) -> ProfileResult:
    """
    Jerk-limited stopping profile.

    Inputs:
    - v0: impact speed magnitude (m/s), positive.
    - jerk_g: jerk limit in G/s, positive.
    - max_g: peak acceleration cap in G, positive.

    Output:
    - triangular if the cap is not reached
    - trapezoidal if the cap is reached with a plateau t2
    """
    v0 = max(float(v0), 0.0)
    jerk_g = max(float(jerk_g), 0.0)
    max_g = max(float(max_g), 0.0)

    if v0 <= 0:
        raise ValueError('v0 must be > 0 m/s')
    if jerk_g <= 0:
        raise ValueError('jerk_g must be > 0 G/s')
    if max_g <= 0:
        raise ValueError('max_g must be > 0 G')

    jerk = jerk_g * G
    a_limit = max_g * G

    a_tri = math.sqrt(jerk * v0)

    if a_tri <= a_limit + EPS:
        profile_type = 'triangular'
        peak_a = a_tri
        t1 = peak_a / jerk
        t2 = 0.0
    else:
        profile_type = 'trapezoidal'
        peak_a = a_limit
        t1 = peak_a / jerk
        t2 = (v0 - (peak_a * peak_a) / jerk) / peak_a
        if t2 < 0.0:
            t2 = 0.0

    total_time = 2.0 * t1 + t2

    v1 = v0 - 0.5 * jerk * t1 * t1
    s1 = v0 * t1 - (jerk * t1 * t1 * t1) / 6.0
    s2 = v1 * t2 - 0.5 * peak_a * t2 * t2
    s3 = (jerk * t1 * t1 * t1) / 6.0
    stop_distance = s1 + s2 + s3

    return ProfileResult(
        profile_type=profile_type,
        v0=v0,
        jerk_g=jerk_g,
        max_g=max_g,
        jerk=jerk,
        peak_a=peak_a,
        t1=t1,
        t2=t2,
        total_time=total_time,
        stop_distance=stop_distance,
    )


def compute_min_jerk(v0: float, target_stroke: float, max_g: float) -> tuple[float, ProfileResult]:
    """
    Find minimum jerk (G/s) such that stop_distance <= target_stroke, given v0 and max_g.
    Binary-search jerk to match stop_distance ~ target_stroke.
    """
    v0 = float(v0)
    target_stroke = float(target_stroke)
    max_g = float(max_g)

    if v0 <= 0 or target_stroke <= 0 or max_g <= 0:
        raise ValueError('v0, target_stroke, max_g must be positive')

    VERY_LARGE_JERK_G = 1e9

    best = compute_profile(v0=v0, jerk_g=VERY_LARGE_JERK_G, max_g=max_g)
    if target_stroke < best.stop_distance - 1e-9:
        raise ValueError(
            f'Impossible: target stroke {target_stroke:.6f} m < theoretical minimum {best.stop_distance:.6f} m '
            f'even with extremely high jerk, at max_g={max_g:.3f}.'
        )

    low = 1.0
    high = 1.0
    r_high = compute_profile(v0=v0, jerk_g=high, max_g=max_g)
    while r_high.stop_distance > target_stroke:
        high *= 2.0
        if high > 1e9:
            break
        r_high = compute_profile(v0=v0, jerk_g=high, max_g=max_g)

    if r_high.stop_distance > target_stroke:
        raise ValueError(
            'Could not bracket a jerk solution: target stroke too small or numerical issue.'
        )

    best_res = r_high
    for _ in range(120):
        mid = 0.5 * (low + high)
        r_mid = compute_profile(v0=v0, jerk_g=mid, max_g=max_g)
        best_res = r_mid
        if abs(r_mid.stop_distance - target_stroke) < 1e-9:
            return mid, r_mid
        if r_mid.stop_distance > target_stroke:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high), best_res


def profile_kinematics_at_time(pr: ProfileResult, t: float) -> tuple[float, float, float]:
    """
    Exact kinematics within the jerk-limited stopping profile.

    Returns (a, v, x):
    - a(t): acceleration magnitude along motion direction (m/s^2), >=0
    - v(t): remaining speed magnitude (m/s), decreases from v0 to 0
    - x(t): stopping displacement (m), increases from 0 to stopDistance
    """
    t = float(t)
    if t <= 0.0:
        return 0.0, pr.v0, 0.0
    if t >= pr.total_time:
        return 0.0, 0.0, pr.stop_distance

    j = pr.jerk
    A = pr.peak_a
    t1 = pr.t1
    t2 = pr.t2

    if t <= t1 + EPS:
        a = j * t
        v = pr.v0 - 0.5 * j * t * t
        x = pr.v0 * t - (j * t * t * t) / 6.0
        return a, v, x

    v1 = pr.v0 - 0.5 * j * t1 * t1
    s1 = pr.v0 * t1 - (j * t1 * t1 * t1) / 6.0

    if t <= t1 + t2 + EPS:
        tau = t - t1
        a = A
        v = v1 - A * tau
        x = s1 + v1 * tau - 0.5 * A * tau * tau
        return a, v, x

    sigma = t - t1 - t2
    a = max(A - j * sigma, 0.0)
    v2 = v1 - A * t2
    s2 = v1 * t2 - 0.5 * A * t2 * t2
    s2_end = s1 + s2

    v = v2 - A * sigma + 0.5 * j * sigma * sigma
    x = s2_end + v2 * sigma - 0.5 * A * sigma * sigma + (j * sigma * sigma * sigma) / 6.0
    return a, v, x


# -------------------------
# Model data (papers)
# -------------------------

SPINE_LEVELS_BOTTOM_TO_TOP = [
    'Pelvis',
    'L5',
    'L4',
    'L3',
    'L2',
    'L1',
    'T12',
    'T11',
    'T10',
    'T9',
    'T8',
    'T7',
    'T6',
    'T5',
    'T4',
    'T3',
    'T2',
    'T1',
    'C7',
    'C6',
    'C5',
    'C4',
    'C3',
    'C2',
    'C1',
    'Head',
]

SPINE_MASS_KG = {
    'Head': 4.5,
    'C1': 0.815,
    'C2': 0.815,
    'C3': 0.815,
    'C4': 0.815,
    'C5': 0.815,
    'C6': 0.900,
    'C7': 1.200,
    'T1': 2.114,
    'T2': 1.829,
    'T3': 1.915,
    'T4': 1.819,
    'T5': 1.930,
    'T6': 1.948,
    'T7': 1.308,
    'T8': 1.326,
    'T9': 1.417,
    'T10': 1.352,
    'T11': 0.3184,
    'T12': 0.3329,
    'L1': 0.2842,
    'L2': 0.3420,
    'L3': 0.4325,
    'L4': 0.5621,
    'L5': 0.4659,
    'Pelvis': 16.879,
}

DISC_AXIAL_K_N_PER_M = {
    'Head-C1': 0.55e6,
    'C1-C2': 0.30e6,
    'C2-C3': 0.70e6,
    'C3-C4': 0.76e6,
    'C4-C5': 0.794e6,
    'C5-C6': 0.967e6,
    'C6-C7': 1.014e6,
    'C7-T1': 1.334e6,
    'T1-T2': 0.70e6,
    'T2-T3': 1.20e6,
    'T3-T4': 1.50e6,
    'T4-T5': 2.10e6,
    'T5-T6': 1.90e6,
    'T6-T7': 1.80e6,
    'T7-T8': 1.50e6,
    'T8-T9': 1.50e6,
    'T9-T10': 1.50e6,
    'T10-T11': 1.50e6,
    'T11-T12': 1.50e6,
    'T12-L1': 1.80e6,
    'L1-L2': 2.13e6,
    'L2-L3': 2.00e6,
    'L3-L4': 2.00e6,
    'L4-L5': 1.87e6,
    'L5-Pelvis': 1.47e6,
}

VISCERA_LEVELS = ['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']
VISCERA_MASS_KG = {
    'T11': 1.282,
    'T12': 1.341,
    'L1': 1.676,
    'L2': 1.670,
    'L3': 1.720,
    'L4': 1.625,
    'L5': 1.774,
}

VISCERA_K_N_PER_M = {
    'T10-T11': 2.86e4,
    'T11-T12': 2.62e4,
    'T12-L1': 2.42e4,
    'L1-L2': 2.24e4,
    'L2-L3': 1.91e4,
    'L3-L4': 1.64e4,
    'L4-L5': 1.68e4,
    'L5-Pelvis': 1.29e4,
}

BUTTOCKS_K_N_PER_M = 8.8425e4
BUTTOCKS_C_NS_PER_M = 1700.0


@dataclass(frozen=True)
class Node:
    node_id: str
    kind: Literal['spine', 'viscera']
    mass_kg: float
    dof: int


@dataclass(frozen=True)
class Element:
    """
    A 2-node axial element: spring + damper.

    Signed internal force convention used for output:
      compression-positive: F_signed = -(k*delta + c*delta_dot).
    """

    elem_id: str
    kind: Literal['disc', 'viscera', 'buttocks', 'strap']
    node_i: str
    node_j: str
    k: float
    c: float
    unilateral_compression_only: bool = False
    unilateral_tension_only: bool = False


@dataclass(frozen=True)
class ModelConfig:
    body_mass_target_kg: float = 75.0
    disc_damping_ns_per_m: float = 1200.0
    viscera_damping_ns_per_m: float = 1200.0
    thoracolumbar_disc_damping_mult: float = 5.0
    thoracolumbar_from_level: str = 'T10'
    thoracolumbar_to_level: str = 'L5'
    buttocks_k: float = BUTTOCKS_K_N_PER_M
    buttocks_c: float = BUTTOCKS_C_NS_PER_M
    strap_k: float = 2.0e5
    strap_c: float = 1200.0


def build_model(
    cfg: ModelConfig,
) -> tuple[list[Node], list[Element], dict[str, int], dict[str, int]]:
    """
    Build nodes and elements, and scale masses to cfg.body_mass_target_kg.
    """
    spine_nodes: list[Node] = []
    dof = 0
    spine_index: dict[str, int] = {}
    for lvl in SPINE_LEVELS_BOTTOM_TO_TOP:
        m = SPINE_MASS_KG[lvl]
        spine_index[lvl] = dof
        spine_nodes.append(Node(node_id=lvl, kind='spine', mass_kg=m, dof=dof))
        dof += 1

    viscera_nodes: list[Node] = []
    viscera_index: dict[str, int] = {}
    for lvl in VISCERA_LEVELS:
        m = VISCERA_MASS_KG[lvl]
        node_id = f'V_{lvl}'
        viscera_index[lvl] = dof
        viscera_nodes.append(Node(node_id=node_id, kind='viscera', mass_kg=m, dof=dof))
        dof += 1

    nodes = spine_nodes + viscera_nodes

    total_mass = sum(n.mass_kg for n in nodes)
    if total_mass <= 0:
        raise ValueError('Total model mass is non-positive.')
    scale = cfg.body_mass_target_kg / total_mass

    nodes_scaled: list[Node] = []
    for n in nodes:
        nodes_scaled.append(
            Node(node_id=n.node_id, kind=n.kind, mass_kg=n.mass_kg * scale, dof=n.dof)
        )

    def in_thoracolumbar_disc(elem_name: str) -> bool:
        a, b = elem_name.split('-')
        order = {lvl: i for i, lvl in enumerate(SPINE_LEVELS_BOTTOM_TO_TOP)}
        lo = cfg.thoracolumbar_to_level
        hi = cfg.thoracolumbar_from_level
        if a not in order or b not in order or lo not in order or hi not in order:
            return False
        i_a = order[a]
        i_b = order[b]
        i_lo = order[lo]
        i_hi = order[hi]
        return (min(i_a, i_b) >= i_lo) and (max(i_a, i_b) <= i_hi)

    elements: list[Element] = []

    for i in range(len(SPINE_LEVELS_BOTTOM_TO_TOP) - 1):
        lower = SPINE_LEVELS_BOTTOM_TO_TOP[i]
        upper = SPINE_LEVELS_BOTTOM_TO_TOP[i + 1]

        disc_key = f'{upper}-{lower}'
        if disc_key not in DISC_AXIAL_K_N_PER_M:
            disc_key2 = f'{lower}-{upper}'
            if disc_key2 in DISC_AXIAL_K_N_PER_M:
                disc_key = disc_key2
            else:
                raise KeyError(f'Missing disc stiffness for segment {upper}-{lower}')

        k = DISC_AXIAL_K_N_PER_M[disc_key]
        c = cfg.disc_damping_ns_per_m
        if in_thoracolumbar_disc(disc_key):
            c *= cfg.thoracolumbar_disc_damping_mult

        elements.append(
            Element(
                elem_id=f'DISC_{disc_key}',
                kind='disc',
                node_i=upper,
                node_j=lower,
                k=k,
                c=c,
            )
        )

    # Pelvis-seat coupling
    elements.append(
        Element(
            elem_id='BUTTOCKS_Pelvis-SEAT',
            kind='buttocks',
            node_i='Pelvis',
            node_j='SEAT',
            k=cfg.buttocks_k,
            c=cfg.buttocks_c,
            unilateral_compression_only=True,
        )
    )
    elements.append(
        Element(
            elem_id='STRAP_Pelvis-SEAT',
            kind='strap',
            node_i='Pelvis',
            node_j='SEAT',
            k=cfg.strap_k,
            c=cfg.strap_c,
            unilateral_tension_only=True,
        )
    )

    def viscera_node_id(level: str) -> str:
        return f'V_{level}'

    viscera_chain = ['T10'] + VISCERA_LEVELS + ['Pelvis']
    for i in range(len(viscera_chain) - 1):
        a = viscera_chain[i]
        b = viscera_chain[i + 1]
        node_a = a if a in ('T10', 'Pelvis') else viscera_node_id(a)
        node_b = b if b in ('T10', 'Pelvis') else viscera_node_id(b)

        k_key = f'{a}-{b}'
        if k_key not in VISCERA_K_N_PER_M:
            raise KeyError(f'Missing viscera stiffness for segment {k_key}')

        elements.append(
            Element(
                elem_id=f'VISCERA_{k_key}',
                kind='viscera',
                node_i=node_a,
                node_j=node_b,
                k=VISCERA_K_N_PER_M[k_key],
                c=cfg.viscera_damping_ns_per_m,
            )
        )

    def assert_all_nodes_connected(nodes_: list[Node], elements_: list[Element]) -> None:
        connected: set[str] = set()
        for e in elements_:
            if e.node_i != 'SEAT':
                connected.add(e.node_i)
            if e.node_j != 'SEAT':
                connected.add(e.node_j)
        unconnected = [n.node_id for n in nodes_ if n.node_id not in connected]
        if unconnected:
            raise RuntimeError(f'Unconnected DOFs (no elements attached): {unconnected}')

    assert_all_nodes_connected(nodes_scaled, elements)
    return nodes_scaled, elements, spine_index, viscera_index


# -------------------------
# Input: seat acceleration
# -------------------------


@dataclass(frozen=True)
class SeatKinematics:
    t: np.ndarray  # s
    a: np.ndarray  # m/s^2
    v: np.ndarray  # m/s (visualization)
    x: np.ndarray  # m (visualization)
    segments: list[dict[str, Any]]


def build_seat_kinematics_analytic(
    *,
    dt: float,
    pre_freefall_s: float,
    impact_speed_mps: float,
    target_stroke_m: float,
    max_g: float,
    post_time_s: float,
) -> tuple[SeatKinematics, ProfileResult]:
    """
    Analytic seat/base kinematics centered on impact.

    Segments:
    - pre_freefall: a = -g
    - impact_decel: jerk-limited stopping pulse computed from (impact_speed, stroke, max_g)
    - post: a = 0

    Solver uses a(t) directly; v(t), x(t) are for plotting/JSON.
    """
    if dt <= 0:
        raise ValueError('dt must be positive.')
    if pre_freefall_s < 0 or post_time_s < 0:
        raise ValueError('pre_freefall_s and post_time_s must be non-negative.')
    if impact_speed_mps <= 0:
        raise ValueError('impact_speed_mps must be positive.')
    if target_stroke_m <= 0:
        raise ValueError('target_stroke_m must be positive.')
    if max_g <= 0:
        raise ValueError('max_g must be positive.')

    v0 = float(impact_speed_mps)
    jerk_g, pr = compute_min_jerk(v0=v0, target_stroke=target_stroke_m, max_g=max_g)

    t0 = 0.0
    t_pre_end = pre_freefall_s
    t_impact_end = t_pre_end + pr.total_time
    t_end = t_impact_end + post_time_s

    n = int(math.floor((t_end - t0) / dt)) + 1
    t = t0 + dt * np.arange(n, dtype=np.float64)

    a = np.zeros_like(t)
    v = np.zeros_like(t)
    x = np.zeros_like(t)

    segments: list[dict[str, Any]] = [
        {
            'name': 'pre_freefall',
            't_start': t0,
            't_end': t_pre_end,
            'description': 'Short freefall baseline: a = -g.',
        },
        {
            'name': 'impact_decel',
            't_start': t_pre_end,
            't_end': t_impact_end,
            'description': 'Jerk-limited decel pulse (min-jerk for given v0, stroke, max_g).',
            'impact_speed_mps': v0,
            'target_stroke_m': target_stroke_m,
            'max_g': max_g,
            'jerk_g': jerk_g,
            'profile_type': pr.profile_type,
            't1': pr.t1,
            't2': pr.t2,
            'total_time': pr.total_time,
            'computed_stop_distance_m': pr.stop_distance,
            'peak_g': pr.peak_a / G,
        },
        {
            'name': 'post',
            't_start': t_impact_end,
            't_end': t_end,
            'description': 'Seat held at rest: a = 0.',
        },
    ]

    # Visualization continuity: enforce v(impact_start) = -v0, x(impact_start)=0.
    v_init = -v0 + G * pre_freefall_s
    x_init = -v_init * pre_freefall_s + 0.5 * G * pre_freefall_s * pre_freefall_s

    for i, ti in enumerate(t):
        ti = float(ti)
        if ti < t_pre_end - EPS:
            tau = ti - t0
            a[i] = -G
            v[i] = v_init - G * tau
            x[i] = x_init + v_init * tau - 0.5 * G * tau * tau
        elif ti < t_impact_end - EPS:
            tau = ti - t_pre_end
            a_prof, v_prof, x_prof = profile_kinematics_at_time(pr, tau)
            a[i] = +a_prof
            v[i] = -v_prof
            x[i] = -x_prof
        else:
            a[i] = 0.0
            v[i] = 0.0
            x[i] = -pr.stop_distance

    return SeatKinematics(t=t, a=a, v=v, x=x, segments=segments), pr


def read_seat_accel_csv(
    *,
    csv_path: str,
    time_col: str = 'time0',
    accel_col: str = 'accel',
    resample_dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read CSV with columns (time_col, accel_col).
    accel is in G units, inertial acceleration, +up convention.
    Returns (t, a_mps2).
    """
    times: list[float] = []
    accels_g: list[float] = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV has no header row.')
        fields = {name.strip(): name for name in reader.fieldnames}
        if time_col not in fields and 'time' in fields:
            time_col = 'time'
        if time_col not in fields:
            raise ValueError(f"CSV missing time column '{time_col}'. Found: {reader.fieldnames}")
        if accel_col not in fields:
            raise ValueError(f"CSV missing accel column '{accel_col}'. Found: {reader.fieldnames}")

        for row in reader:
            times.append(float(row[fields[time_col]]))
            accels_g.append(float(row[fields[accel_col]]))

    t = np.asarray(times, dtype=np.float64)
    a_g = np.asarray(accels_g, dtype=np.float64)

    idx = np.argsort(t)
    t = t[idx]
    a_g = a_g[idx]

    a = a_g * G

    if resample_dt is not None:
        t, a = lin_interp_resample(t, a, resample_dt)

    return t, a


def build_seat_kinematics_from_csv(*, csv_path: str, resample_dt: float | None) -> SeatKinematics:
    """
    Convert CSV acceleration to seat kinematics (v, x) by integration for plotting/JSON.
    Solver uses a(t) directly.
    """
    t, a = read_seat_accel_csv(csv_path=csv_path, resample_dt=resample_dt)
    v, x = trapz_integrate(t, a, v0=0.0, x0=0.0)
    segments = [
        {
            'name': 'csv_input',
            't_start': float(t[0]),
            't_end': float(t[-1]),
            'description': 'Seat/base motion from CSV accelerometer signal.',
            'csv_path': csv_path,
            'resample_dt': resample_dt,
        }
    ]
    return SeatKinematics(t=t, a=a, v=v, x=x, segments=segments)


# -------------------------
# Simulation (Newmark-beta)
# -------------------------


@dataclass(frozen=True)
class SimulationConfig:
    dt: float
    beta: float = 1.0 / 4.0
    gamma: float = 1.0 / 2.0
    max_contact_iterations: int = 30
    seat_contact_deadband_m: float = 1e-6
    initial_condition: Literal['rest', 'freefall'] = 'freefall'


@dataclass
class SimulationResult:
    t: np.ndarray
    seat: SeatKinematics
    nodes: list[Node]
    elements: list[Element]
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    elem_deformation: np.ndarray
    elem_deformation_rate: np.ndarray
    elem_force_signed: np.ndarray
    elem_force_comp: np.ndarray
    seat_coupling_state: np.ndarray  # -1 buttocks, +1 strap
    meta: dict[str, Any]


def assemble_linear_mck(
    nodes: list[Node], elements: list[Element]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Assemble linear M, C, K for all DOFs excluding any element connected to SEAT.
    """
    n = len(nodes)
    node_id_to_dof = {nd.node_id: nd.dof for nd in nodes}

    M = np.zeros((n, n), dtype=np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    K = np.zeros((n, n), dtype=np.float64)

    for nd in nodes:
        M[nd.dof, nd.dof] = nd.mass_kg

    def add_2node(node_i: str, node_j: str, k: float, c: float) -> None:
        i = node_id_to_dof[node_i]
        j = node_id_to_dof[node_j]
        K[i, i] += k
        K[i, j] -= k
        K[j, i] -= k
        K[j, j] += k

        C[i, i] += c
        C[i, j] -= c
        C[j, i] -= c
        C[j, j] += c

    for e in elements:
        if e.node_i == 'SEAT' or e.node_j == 'SEAT':
            continue
        add_2node(e.node_i, e.node_j, e.k, e.c)

    return M, C, K, node_id_to_dof


def solve_static_equilibrium(
    *, K_lin: np.ndarray, nodes: list[Node], node_id_to_dof: dict[str, int], buttocks_elem: Element
) -> np.ndarray:
    """
    Static equilibrium in relative coordinates for a seated-at-rest initial condition:
      (K_lin + K_buttocks) u0 = f_g, where f_g = m*(-g).
    """
    n = K_lin.shape[0]
    f = np.zeros(n, dtype=np.float64)
    for nd in nodes:
        f[nd.dof] += nd.mass_kg * (-G)

    pelvis_dof = node_id_to_dof['Pelvis']
    K = K_lin.copy()
    K[pelvis_dof, pelvis_dof] += buttocks_elem.k
    return np.linalg.solve(K, f)


def run_simulation(
    *,
    model_cfg: ModelConfig,
    sim_cfg: SimulationConfig,
    seat: SeatKinematics,
    profile: ProfileResult | None,
) -> SimulationResult:
    nodes, elements, spine_index, viscera_index = build_model(model_cfg)
    buttocks = next(e for e in elements if e.kind == 'buttocks')
    strap = next(e for e in elements if e.kind == 'strap')

    M, C_lin, K_lin, node_id_to_dof = assemble_linear_mck(nodes, elements)

    t = seat.t
    n_steps = len(t)
    n_dof = len(nodes)
    n_elem = len(elements)
    pelvis_dof = node_id_to_dof['Pelvis']

    if sim_cfg.initial_condition == 'rest':
        q0 = solve_static_equilibrium(
            K_lin=K_lin, nodes=nodes, node_id_to_dof=node_id_to_dof, buttocks_elem=buttocks
        )
        v0 = np.zeros(n_dof, dtype=np.float64)
        state0 = -1
    else:
        # Start weightless with no relative deformation and no relative velocity.
        q0 = np.zeros(n_dof, dtype=np.float64)
        v0 = np.zeros(n_dof, dtype=np.float64)
        # Choose buttocks as the default active path at delta≈0 to keep coupling continuous.
        state0 = -1

    q = np.zeros((n_steps, n_dof), dtype=np.float64)
    v = np.zeros((n_steps, n_dof), dtype=np.float64)
    a = np.zeros((n_steps, n_dof), dtype=np.float64)
    q[0, :] = q0
    v[0, :] = v0

    elem_def = np.zeros((n_steps, n_elem), dtype=np.float64)
    elem_defdot = np.zeros((n_steps, n_elem), dtype=np.float64)
    elem_force_signed = np.zeros((n_steps, n_elem), dtype=np.float64)
    elem_force_comp = np.zeros((n_steps, n_elem), dtype=np.float64)

    seat_state = np.zeros(n_steps, dtype=np.int8)
    seat_state[0] = int(state0)

    beta = sim_cfg.beta
    gamma = sim_cfg.gamma
    dt = sim_cfg.dt
    eps_contact = float(sim_cfg.seat_contact_deadband_m)

    def compute_external_force_vector(a_seat: float) -> np.ndarray:
        f_ext = np.zeros(n_dof, dtype=np.float64)
        for nd in nodes:
            f_ext[nd.dof] += nd.mass_kg * (-G - a_seat)
        return f_ext

    def update_seat_coupling_state(prev_state: int, delta: float) -> int:
        """
        Two-state coupling with deadband hysteresis:

        -1: buttocks active (compression path)
        +1: strap active (tension path)

        If |delta| < eps_contact, keep previous state.
        Else choose by sign:
          delta < 0 => buttocks
          delta > 0 => strap
        """
        if abs(delta) < eps_contact:
            return prev_state
        return -1 if delta < 0.0 else +1

    def build_total_CK(*, coupling_state: int) -> tuple[np.ndarray, np.ndarray]:
        C2 = C_lin
        K2 = K_lin
        C2 = C2.copy()
        K2 = K2.copy()

        if coupling_state == -1:
            C2[pelvis_dof, pelvis_dof] += buttocks.c
            K2[pelvis_dof, pelvis_dof] += buttocks.k
        elif coupling_state == +1:
            C2[pelvis_dof, pelvis_dof] += strap.c
            K2[pelvis_dof, pelvis_dof] += strap.k
        else:
            raise AssertionError('Invalid coupling_state.')
        return C2, K2

    # Initial acceleration
    C0, K0 = build_total_CK(coupling_state=int(seat_state[0]))
    f0 = compute_external_force_vector(float(seat.a[0]))
    a[0, :] = np.linalg.solve(M, f0 - C0 @ v[0, :] - K0 @ q[0, :])

    # Newmark constants
    a0c = 1.0 / (beta * dt * dt)
    a1c = gamma / (beta * dt)
    a2c = 1.0 / (beta * dt)
    a3c = (1.0 / (2.0 * beta)) - 1.0
    a4c = (gamma / beta) - 1.0
    a5c = dt * ((gamma / (2.0 * beta)) - 1.0)

    for n in range(n_steps - 1):
        a_seat_n1 = float(seat.a[n + 1])

        qn = q[n, :]
        vn = v[n, :]
        an = a[n, :]

        state_guess = int(seat_state[n])

        qn1 = None
        vn1 = None
        an1 = None
        state_n1 = None

        for _it in range(sim_cfg.max_contact_iterations):
            C_tot, K_tot = build_total_CK(coupling_state=state_guess)
            f_ext = compute_external_force_vector(a_seat_n1)

            K_eff = K_tot + a0c * M + a1c * C_tot
            p_eff = (
                f_ext
                + M @ (a0c * qn + a2c * vn + a3c * an)
                + C_tot @ (a1c * qn + a4c * vn + a5c * an)
            )

            qn1_try = np.linalg.solve(K_eff, p_eff)

            an1_try = a0c * (qn1_try - qn) - a2c * vn - a3c * an
            vn1_try = vn + dt * ((1.0 - gamma) * an + gamma * an1_try)

            delta = float(qn1_try[pelvis_dof])
            state_try = update_seat_coupling_state(state_guess, delta)

            if state_try == state_guess:
                qn1, vn1, an1 = qn1_try, vn1_try, an1_try
                state_n1 = state_try
                break

            state_guess = state_try
        else:
            qn1, vn1, an1 = qn1_try, vn1_try, an1_try
            state_n1 = state_guess

        q[n + 1, :] = qn1
        v[n + 1, :] = vn1
        a[n + 1, :] = an1
        seat_state[n + 1] = int(state_n1)

    # Element states (deformation, deformation_rate, force)
    node_id_to_dof = {nd.node_id: nd.dof for nd in nodes}
    for ti in range(n_steps):
        for ei, e in enumerate(elements):
            if e.kind in ('buttocks', 'strap'):
                dof_i = node_id_to_dof[e.node_i]
                qi = float(q[ti, dof_i])
                vi = float(v[ti, dof_i])
                delta = qi
                deltadot = vi

                st = int(seat_state[ti])
                active = (e.kind == 'buttocks' and st == -1) or (e.kind == 'strap' and st == +1)

                if not active:
                    f_signed = 0.0
                    f_comp = 0.0
                else:
                    f_signed = -(e.k * delta + e.c * deltadot)
                    f_comp = max(0.0, f_signed)

                elem_def[ti, ei] = delta
                elem_defdot[ti, ei] = deltadot
                elem_force_signed[ti, ei] = f_signed
                elem_force_comp[ti, ei] = f_comp
                continue

            dof_i = node_id_to_dof[e.node_i]
            dof_j = node_id_to_dof[e.node_j]
            delta = float(q[ti, dof_i] - q[ti, dof_j])
            deltadot = float(v[ti, dof_i] - v[ti, dof_j])

            f_signed = -(e.k * delta + e.c * deltadot)
            f_comp = max(0.0, f_signed)

            elem_def[ti, ei] = delta
            elem_defdot[ti, ei] = deltadot
            elem_force_signed[ti, ei] = f_signed
            elem_force_comp[ti, ei] = f_comp

    meta: dict[str, Any] = {
        'g_mps2': G,
        'sign_convention': {
            'axis': '+up',
            'coordinates': 'relative_to_seat: u = x - x_seat (seat reference is 0 in dynamics)',
            'seat_accel_units': 'm/s^2 (also saved as G)',
            'freefall': '-1 G',
            'rest': '0 G',
            'impact_decel': '+G',
            'disc_force_signed': 'compression-positive',
        },
        'initial_condition': sim_cfg.initial_condition,
        'seat_coupling': {
            'states': {'buttocks': -1, 'strap': +1},
            'deadband_m': eps_contact,
            'note': 'Always exactly one path active (buttocks OR strap). No "none" state.',
        },
        'mass_scaling': {
            'target_total_mass_kg': model_cfg.body_mass_target_kg,
            'note': 'All lumped masses (spine + viscera) scaled uniformly to reach target total mass.',
        },
        'damping': {
            'disc_base_ns_per_m': model_cfg.disc_damping_ns_per_m,
            'viscera_ns_per_m': model_cfg.viscera_damping_ns_per_m,
            'thoracolumbar_multiplier': model_cfg.thoracolumbar_disc_damping_mult,
            'thoracolumbar_region': f'{model_cfg.thoracolumbar_from_level}..{model_cfg.thoracolumbar_to_level}',
        },
        'buttocks': {'k_n_per_m': model_cfg.buttocks_k, 'c_ns_per_m': model_cfg.buttocks_c},
        'strap': {'k_n_per_m': model_cfg.strap_k, 'c_ns_per_m': model_cfg.strap_c},
        'input_mode': seat.segments[0]['name'],
        'segments': seat.segments,
        'profile': None
        if profile is None
        else {
            'profile_type': profile.profile_type,
            'v0_mps': profile.v0,
            'jerk_g_per_s': profile.jerk_g,
            'max_g': profile.max_g,
            'peak_g': profile.peak_a / G,
            't1_s': profile.t1,
            't2_s': profile.t2,
            'total_time_s': profile.total_time,
            'stop_distance_m': profile.stop_distance,
        },
    }

    return SimulationResult(
        t=t,
        seat=seat,
        nodes=nodes,
        elements=elements,
        q=q,
        v=v,
        a=a,
        elem_deformation=elem_def,
        elem_deformation_rate=elem_defdot,
        elem_force_signed=elem_force_signed,
        elem_force_comp=elem_force_comp,
        seat_coupling_state=seat_state,
        meta=meta,
    )


# -------------------------
# Postprocessing & metrics
# -------------------------


def element_index_by_id(elements: list[Element]) -> dict[str, int]:
    return {e.elem_id: i for i, e in enumerate(elements)}


def node_index_by_id(nodes: list[Node]) -> dict[str, int]:
    return {n.node_id: n.dof for n in nodes}


def find_segment(seat: SeatKinematics, name: str) -> dict[str, Any] | None:
    return next((s for s in seat.segments if s.get('name') == name), None)


def window_indices(t: np.ndarray, t0: float, t1: float) -> tuple[int, int]:
    i0 = int(np.searchsorted(t, t0, side='left'))
    i1 = int(np.searchsorted(t, t1, side='right')) - 1
    i0 = max(0, min(i0, len(t) - 1))
    i1 = max(0, min(i1, len(t) - 1))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1


@dataclass(frozen=True)
class WindowMetric:
    peak: float
    t_peak: float
    peak_rate: float
    t_peak_rate: float


def compute_peak_and_rate(t: np.ndarray, y: np.ndarray, i0: int, i1: int) -> WindowMetric:
    """
    Compute:
    - peak: max(y) in window
    - peak_rate: max(dy/dt) in window (positive-going rate)
    """
    if i1 < i0:
        return WindowMetric(float('nan'), float('nan'), float('nan'), float('nan'))
    tw = t[i0 : i1 + 1]
    yw = y[i0 : i1 + 1]
    if len(yw) == 0:
        return WindowMetric(float('nan'), float('nan'), float('nan'), float('nan'))

    k_peak = int(np.argmax(yw))
    peak = float(yw[k_peak])
    t_peak = float(tw[k_peak])

    if len(yw) >= 2:
        dt = float(np.median(np.diff(tw)))
        dydt = np.diff(yw) / dt
        k_rate = int(np.argmax(dydt))
        peak_rate = float(dydt[k_rate])
        t_peak_rate = float(tw[k_rate + 1])
    else:
        peak_rate = float('nan')
        t_peak_rate = float('nan')

    return WindowMetric(peak=peak, t_peak=t_peak, peak_rate=peak_rate, t_peak_rate=t_peak_rate)


def compute_disc_compression_mm(sim: SimulationResult, disc_elem_indices: list[int]) -> np.ndarray:
    """
    Axial-only "compression sum" proxy (Raj-inspired but without rotation/shear):
      I_axial(t) = sum_i max(0, -delta_i),
    where delta_i is disc deformation (u_upper - u_lower).
    """
    delta = sim.elem_deformation[:, disc_elem_indices]  # (n_steps, n_discs)
    comp = np.maximum(0.0, -delta)  # m
    I_m = np.sum(comp, axis=1)
    return I_m * 1000.0  # mm


def compute_metrics(sim: SimulationResult, early_ms: float) -> dict[str, Any]:
    """
    Compute a bundle of jerk/shape-sensitive metrics for reporting and auto-dt convergence.

    early_ms: window length starting at impact start.
    """
    t = sim.t
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float('nan')

    seg_impact = find_segment(sim.seat, 'impact_decel')
    seg_post = find_segment(sim.seat, 'post')

    elem_idx = element_index_by_id(sim.elements)
    node_idx = node_index_by_id(sim.nodes)

    t12l1_id = 'DISC_T12-L1'
    if t12l1_id not in elem_idx:
        t12l1_id = 'DISC_L1-T12'
    idx_t12l1 = elem_idx[t12l1_id]

    f_t12 = sim.elem_force_comp[:, idx_t12l1]  # N

    # Absolute accelerations of nodes (inertial): a_abs = u_ddot + a_seat
    a_abs = sim.a + sim.seat.a[:, None]  # (n_steps, n_dof)
    a_abs_g = a_abs / G
    j_abs_gps = finite_difference(a_abs_g, dt)  # G/s

    pelvis_dof = node_idx['Pelvis']
    t12_dof = node_idx['T12']
    head_dof = node_idx['Head']

    # Seat jerk (for reference)
    seat_a_g = sim.seat.a / G
    seat_j_gps = finite_difference(seat_a_g, dt)

    # Disc list for whole-spine axial compression proxy (all disc elements)
    disc_elem_indices = [i for i, e in enumerate(sim.elements) if e.kind == 'disc']
    I_axial_mm = compute_disc_compression_mm(sim, disc_elem_indices)

    metrics: dict[str, Any] = {
        'dt_s': dt,
        'seat_peak_g': float(np.max(seat_a_g)),
        'seat_peak_jerk_gps': float(np.max(seat_j_gps)),
        't12l1_elem_id': t12l1_id,
    }

    # Full sim (exclude i=0 to avoid trivial initial-state artifacts)
    metrics['t12l1_peak_full_n'] = float(np.max(f_t12[1:])) if len(f_t12) > 1 else float(f_t12[0])

    # Impact and post windows
    if seg_impact is not None:
        t0 = float(seg_impact['t_start'])
        t1 = float(seg_impact['t_end'])
        i0, i1 = window_indices(t, t0, t1)
        w_force = compute_peak_and_rate(t, f_t12, i0, i1)

        metrics['impact_t0'] = t0
        metrics['impact_t1'] = t1
        metrics['t12l1_peak_impact_n'] = w_force.peak
        metrics['t12l1_t_peak_impact_s'] = w_force.t_peak
        metrics['t12l1_peak_dFdt_impact_nps'] = w_force.peak_rate
        metrics['t12l1_t_peak_dFdt_impact_s'] = w_force.t_peak_rate

        # Early window
        early_s = max(0.0, float(early_ms) / 1000.0)
        te = min(t1, t0 + early_s)
        ie0, ie1 = window_indices(t, t0, te)
        w_early = compute_peak_and_rate(t, f_t12, ie0, ie1)
        metrics['early_ms'] = float(early_ms)
        metrics['t12l1_peak_early_n'] = w_early.peak
        metrics['t12l1_peak_dFdt_early_nps'] = w_early.peak_rate

        # Node accel/jerk peaks during impact and early windows
        def _node_stats(dof: int, label: str) -> None:
            aa = a_abs_g[:, dof]
            jj = j_abs_gps[:, dof]
            w_a = compute_peak_and_rate(t, np.abs(aa), i0, i1)  # peak magnitude
            w_j = compute_peak_and_rate(t, np.abs(jj), i0, i1)
            w_a_e = compute_peak_and_rate(t, np.abs(aa), ie0, ie1)
            w_j_e = compute_peak_and_rate(t, np.abs(jj), ie0, ie1)

            metrics[f'{label}_peak_abs_a_impact_g'] = w_a.peak
            metrics[f'{label}_peak_abs_j_impact_gps'] = w_j.peak
            metrics[f'{label}_peak_abs_a_early_g'] = w_a_e.peak
            metrics[f'{label}_peak_abs_j_early_gps'] = w_j_e.peak

        _node_stats(pelvis_dof, 'pelvis')
        _node_stats(t12_dof, 't12')
        _node_stats(head_dof, 'head')

        # Injury proxy peaks
        w_I = compute_peak_and_rate(t, I_axial_mm, i0, i1)
        w_Ie = compute_peak_and_rate(t, I_axial_mm, ie0, ie1)
        metrics['I_axial_peak_impact_mm'] = w_I.peak
        metrics['I_axial_peak_early_mm'] = w_Ie.peak

        # Coupling fractions in windows
        st = sim.seat_coupling_state.astype(int)
        metrics['coupling_buttocks_frac_impact'] = float(np.mean(st[i0 : i1 + 1] == -1))
        metrics['coupling_strap_frac_impact'] = float(np.mean(st[i0 : i1 + 1] == +1))
        metrics['coupling_buttocks_frac_early'] = float(np.mean(st[ie0 : ie1 + 1] == -1))
        metrics['coupling_strap_frac_early'] = float(np.mean(st[ie0 : ie1 + 1] == +1))

    if seg_post is not None:
        j0, j1 = window_indices(t, float(seg_post['t_start']), float(seg_post['t_end']))
        w_post = compute_peak_and_rate(t, f_t12, j0, j1)
        metrics['t12l1_peak_post_n'] = w_post.peak
        metrics['t12l1_t_peak_post_s'] = w_post.t_peak

    return metrics


# -------------------------
# Plots & JSON
# -------------------------


def save_plots(outdir: str, sim: SimulationResult, early_ms: float) -> None:
    t = sim.t
    a_seat_g = sim.seat.a / G

    seg_impact = find_segment(sim.seat, 'impact_decel')
    impact_t0 = float(seg_impact['t_start']) if seg_impact else None
    impact_t1 = float(seg_impact['t_end']) if seg_impact else None

    elem_idx = element_index_by_id(sim.elements)
    t12_l1_elem_id = 'DISC_T12-L1'
    if t12_l1_elem_id not in elem_idx:
        t12_l1_elem_id = 'DISC_L1-T12'
    idx_t12l1 = elem_idx[t12_l1_elem_id]

    f_t12l1_kn = sim.elem_force_comp[:, idx_t12l1] / 1000.0

    # Seat accel
    plt.figure(figsize=(12, 4))
    plt.plot(t, a_seat_g, linewidth=1.2, label='a_seat [G]')
    plt.axhline(0.0, color='k', linewidth=0.8)
    plt.axhline(-1.0, color='k', linewidth=0.8, linestyle='--', label='-1 G')
    if impact_t0 is not None and impact_t1 is not None:
        plt.axvline(impact_t0, color='b', linewidth=1.0, linestyle='--')
        plt.axvline(impact_t1, color='b', linewidth=1.0, linestyle='--')
    plt.title('Seat/Protector Interface Acceleration (input) [G]')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [G]')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'seat_accel.png'), dpi=160)
    plt.close()

    # T12-L1 force full
    plt.figure(figsize=(12, 4))
    plt.plot(t, f_t12l1_kn, linewidth=1.2, label='T12–L1 comp [kN]')
    if impact_t0 is not None and impact_t1 is not None:
        plt.axvline(impact_t0, color='b', linewidth=1.0, linestyle='--')
        plt.axvline(impact_t1, color='b', linewidth=1.0, linestyle='--')
    plt.title('T12–L1 Disc Compressive Force (compression-positive) [kN]')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [kN]')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 't12_l1_force.png'), dpi=160)
    plt.close()

    # Zoom around impact and early window
    if impact_t0 is not None and impact_t1 is not None:
        early_s = float(early_ms) / 1000.0
        t_zoom0 = max(float(t[0]), impact_t0 - 0.010)
        t_zoom1 = min(float(t[-1]), impact_t1 + 0.030)

        plt.figure(figsize=(12, 4))
        plt.plot(t, f_t12l1_kn, linewidth=1.2)
        plt.axvline(impact_t0, color='b', linewidth=1.0, linestyle='--', label='impact start')
        plt.axvline(impact_t1, color='b', linewidth=1.0, linestyle='--', label='impact end')
        plt.axvline(
            impact_t0 + early_s,
            color='g',
            linewidth=1.0,
            linestyle='--',
            label=f'early {early_ms:.1f} ms',
        )
        plt.xlim(t_zoom0, t_zoom1)
        plt.title('T12–L1 Compressive Force Zoom (impact + early window)')
        plt.xlabel('Time [s]')
        plt.ylabel('Force [kN]')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 't12_l1_force_zoom.png'), dpi=160)
        plt.close()

        # dF/dt zoom (kN/s)
        dt = float(np.median(np.diff(t)))
        dfdt_knps = finite_difference(f_t12l1_kn, dt)
        plt.figure(figsize=(12, 4))
        plt.plot(t, dfdt_knps, linewidth=1.2)
        plt.axvline(impact_t0, color='b', linewidth=1.0, linestyle='--')
        plt.axvline(impact_t1, color='b', linewidth=1.0, linestyle='--')
        plt.axvline(impact_t0 + early_s, color='g', linewidth=1.0, linestyle='--')
        plt.xlim(t_zoom0, t_zoom1)
        plt.title('T12–L1 Loading Rate dF/dt Zoom [kN/s]')
        plt.xlabel('Time [s]')
        plt.ylabel('dF/dt [kN/s]')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 't12_l1_dfdt_zoom.png'), dpi=160)
        plt.close()

    # Heatmap of disc compressive forces
    disc_elems = [e for e in sim.elements if e.kind == 'disc']
    order = {lvl: i for i, lvl in enumerate(SPINE_LEVELS_BOTTOM_TO_TOP)}

    def disc_sort_key(e: Element) -> int:
        return order.get(e.node_j, 9999)

    disc_elems_sorted = sorted(disc_elems, key=disc_sort_key)
    disc_indices = [element_index_by_id(sim.elements)[e.elem_id] for e in disc_elems_sorted]
    disc_labels = [e.elem_id.replace('DISC_', '') for e in disc_elems_sorted]
    F = sim.elem_force_comp[:, disc_indices].T / 1000.0

    plt.figure(figsize=(12, 7))
    im = plt.imshow(
        F,
        aspect='auto',
        origin='lower',
        extent=[float(t[0]), float(t[-1]), 0, len(disc_labels) - 1],
        cmap='inferno',
    )
    plt.colorbar(im, label='Compressive Force [kN]')
    plt.yticks(range(len(disc_labels)), disc_labels, fontsize=8)
    plt.title('Spine Disc Compressive Force Heatmap (time vs level)')
    plt.xlabel('Time [s]')
    plt.ylabel('Disc (bottom → top)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'disc_force_heatmap.png'), dpi=160)
    plt.close()


def save_json(outdir: str, sim: SimulationResult, extra: dict[str, Any]) -> None:
    """
    Write a comprehensive JSON suitable for later visualization.
    """
    node_ids = [n.node_id for n in sim.nodes]
    elem_ids = [e.elem_id for e in sim.elements]

    nodes_json = [
        {'id': n.node_id, 'kind': n.kind, 'mass_kg': n.mass_kg, 'dof': n.dof} for n in sim.nodes
    ]
    elems_json = [
        {
            'id': e.elem_id,
            'kind': e.kind,
            'node_i': e.node_i,
            'node_j': e.node_j,
            'k_n_per_m': e.k,
            'c_ns_per_m': e.c,
            'unilateral_compression_only': e.unilateral_compression_only,
            'unilateral_tension_only': e.unilateral_tension_only,
            'signed_force_convention': 'compression-positive: F_signed = -(k*delta + c*delta_dot)',
        }
        for e in sim.elements
    ]

    x_abs = sim.q + sim.seat.x[:, None]
    a_abs = sim.a + sim.seat.a[:, None]

    data = {
        'meta': sim.meta | extra,
        'nodes': nodes_json,
        'elements': elems_json,
        'time_s': sim.t.tolist(),
        'seat': {
            'x_m': sim.seat.x.tolist(),
            'v_mps': sim.seat.v.tolist(),
            'a_mps2': sim.seat.a.tolist(),
            'a_g': (sim.seat.a / G).tolist(),
        },
        'seat_coupling_state': sim.seat_coupling_state.astype(int).tolist(),
        'states': {
            'node_order': node_ids,
            'element_order': elem_ids,
            'nodes': {
                'x_relative_m': sim.q.tolist(),
                'x_absolute_m': x_abs.tolist(),
                'v_relative_mps': sim.v.tolist(),
                'a_relative_mps2': sim.a.tolist(),
                'a_relative_g': (sim.a / G).tolist(),
                'a_absolute_mps2': a_abs.tolist(),
                'a_absolute_g': (a_abs / G).tolist(),
            },
            'elements': {
                'deformation_m': sim.elem_deformation.tolist(),
                'deformation_rate_mps': sim.elem_deformation_rate.tolist(),
                'force_signed_n': sim.elem_force_signed.tolist(),
                'force_compressive_n': sim.elem_force_comp.tolist(),
                'force_compressive_kn': (sim.elem_force_comp / 1000.0).tolist(),
            },
        },
    }

    out_path = os.path.join(outdir, 'sim.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# -------------------------
# Debug printing
# -------------------------


def print_auto_dt_debug(auto_dt_meta: dict[str, Any]) -> None:
    hist = auto_dt_meta.get('history', [])
    cfg = auto_dt_meta.get('config', {})
    reason = auto_dt_meta.get('stop_reason', 'unknown')

    print('\n=== AUTO-DT DEBUG ===')
    print(f'enabled={auto_dt_meta.get("enabled")}')
    if cfg:
        print('[config]')
        for k in [
            'max_iter',
            'min_dt',
            'input_peak_rel_tol',
            'force_rel_tol',
            'dFdt_rel_tol',
            'early_force_rel_tol',
            'early_dFdt_rel_tol',
            'require_consecutive_passes',
            'min_samples_per_ramp',
        ]:
            if k in cfg:
                print(f'  {k}: {cfg[k]}')
        if 'early_ms' in cfg:
            print(f'  early_ms: {cfg["early_ms"]}')

    print(f'stop_reason={reason}')

    if not hist:
        print('no history')
        print('=== END AUTO-DT DEBUG ===\n')
        return

    # Table header
    cols = [
        'dt_s',
        'ramp_samples',
        'sampled_peak_g',
        'input_peak_rel_err',
        'peak_impact_kn',
        'rel_change_peak_impact',
        'peak_dFdt_impact_knps',
        'rel_change_peak_dFdt_impact',
        'peak_early_kn',
        'rel_change_peak_early',
        'peak_dFdt_early_knps',
        'rel_change_peak_dFdt_early',
        'pass_flags',
    ]
    print('\n[history]')
    print(' | '.join(cols))
    print('-' * 140)
    for row in hist:
        vals = []
        for c in cols:
            v = row.get(c, '')
            if isinstance(v, float):
                vals.append(f'{v:.6g}')
            else:
                vals.append(str(v))
        print(' | '.join(vals))
    print('=== END AUTO-DT DEBUG ===\n')


def print_debug_summary(
    sim: SimulationResult,
    profile: ProfileResult | None,
    early_ms: float,
    auto_dt_meta: dict[str, Any] | None,
) -> None:
    t = sim.t
    dt_stats = np.diff(t) if len(t) > 1 else np.array([math.nan])

    print('\n=== DEBUG SUMMARY ===')
    print(f'n_steps={len(t)}, t_start={t[0]:.6f} s, t_end={t[-1]:.6f} s')
    print(
        'dt: '
        f'median={float(np.median(dt_stats)):.6g} s, '
        f'min={float(np.min(dt_stats)):.6g} s, '
        f'max={float(np.max(dt_stats)):.6g} s'
    )

    a_g = sim.seat.a / G
    print('\n[seat]')
    print(f'a_g: min={float(np.min(a_g)):.3f}, max={float(np.max(a_g)):.3f}')
    print(f'v_mps: min={float(np.min(sim.seat.v)):.3f}, max={float(np.max(sim.seat.v)):.3f}')
    print(f'x_m:   min={float(np.min(sim.seat.x)):.3f}, max={float(np.max(sim.seat.x)):.3f}')

    if profile is not None:
        print('\n[profile]')
        print(
            f'type={profile.profile_type}, v0={profile.v0:.3f} m/s, peak={profile.peak_a / G:.3f} G, '
            f'jerk={profile.jerk_g:.3f} G/s, stopDist={profile.stop_distance:.6f} m, '
            f'totalTime={profile.total_time * 1000.0:.3f} ms, t_ramp={profile.t1 * 1000.0:.3f} ms'
        )

    for s in sim.seat.segments:
        print(f'[segments] {s["name"]}: t=[{float(s["t_start"]):.6f}, {float(s["t_end"]):.6f}]')

    st = sim.seat_coupling_state.astype(int)
    butt_frac = float(np.mean(st == -1)) if len(st) else 0.0
    strap_frac = float(np.mean(st == +1)) if len(st) else 0.0
    print('\n[pelvis-seat coupling]')
    print(f'buttocks engaged fraction: {butt_frac * 100.0:.2f}%')
    print(f'strap engaged fraction:    {strap_frac * 100.0:.2f}%')

    m = compute_metrics(sim, early_ms=early_ms)

    print('\n[T12-L1 metrics]')
    print(f'peak impact: {m.get("t12l1_peak_impact_n", float("nan")) / 1000.0:.3f} kN')
    print(
        f'peak early ({early_ms:.1f} ms): {m.get("t12l1_peak_early_n", float("nan")) / 1000.0:.3f} kN'
    )
    print(f'peak post: {m.get("t12l1_peak_post_n", float("nan")) / 1000.0:.3f} kN')
    print(f'peak full: {m.get("t12l1_peak_full_n", float("nan")) / 1000.0:.3f} kN')
    print(
        f'peak dF/dt impact: {m.get("t12l1_peak_dFdt_impact_nps", float("nan")) / 1000.0:.3f} kN/s'
    )
    print(
        f'peak dF/dt early:  {m.get("t12l1_peak_dFdt_early_nps", float("nan")) / 1000.0:.3f} kN/s'
    )

    print('\n[absolute accel/jerk peaks during impact]')
    for label in ['pelvis', 't12', 'head']:
        print(
            f'{label}: '
            f'peak|a|={m.get(label + "_peak_abs_a_impact_g", float("nan")):.3f} G, '
            f'peak|j|={m.get(label + "_peak_abs_j_impact_gps", float("nan")):.1f} G/s; '
            f'early peak|a|={m.get(label + "_peak_abs_a_early_g", float("nan")):.3f} G, '
            f'early peak|j|={m.get(label + "_peak_abs_j_early_gps", float("nan")):.1f} G/s'
        )

    print('\n[injury proxy]')
    print(f'I_axial peak impact: {m.get("I_axial_peak_impact_mm", float("nan")):.3f} mm')
    print(f'I_axial peak early:  {m.get("I_axial_peak_early_mm", float("nan")):.3f} mm')

    if 'coupling_buttocks_frac_impact' in m:
        print('\n[coupling fractions in windows]')
        print(
            f'impact: buttocks={m["coupling_buttocks_frac_impact"] * 100.0:.2f}%, '
            f'strap={m["coupling_strap_frac_impact"] * 100.0:.2f}%'
        )
        print(
            f'early:  buttocks={m["coupling_buttocks_frac_early"] * 100.0:.2f}%, '
            f'strap={m["coupling_strap_frac_early"] * 100.0:.2f}%'
        )

    if auto_dt_meta is not None:
        print_auto_dt_debug(auto_dt_meta)

    print('=== END DEBUG SUMMARY ===\n')


# -------------------------
# Automatic dt refinement
# -------------------------


@dataclass(frozen=True)
class AutoDtConfig:
    enabled: bool = True
    max_iter: int = 16
    min_dt: float = 1e-6

    # Input peak capture
    input_peak_rel_tol: float = 0.005  # 0.5%

    # Convergence tolerances (impact window)
    force_rel_tol: float = 0.02  # 2%
    dFdt_rel_tol: float = 0.05  # 5%

    # Convergence tolerances (early window)
    early_force_rel_tol: float = 0.02
    early_dFdt_rel_tol: float = 0.08

    require_consecutive_passes: int = 2

    # Additional "physics-aware" refinement rule:
    # Ensure t_ramp/dt >= min_samples_per_ramp to resolve jerk-limited ramps.
    min_samples_per_ramp: int = 20

    # Early window definition for jerk sensitivity
    early_ms: float = 5.0


def _relative_change(new: float, old: float) -> float:
    if not math.isfinite(new) or not math.isfinite(old):
        return float('inf')
    denom = max(abs(new), EPS)
    return abs(new - old) / denom


def run_analytic_with_auto_dt(
    *,
    model_cfg: ModelConfig,
    sim_cfg_base: SimulationConfig,
    auto_cfg: AutoDtConfig,
    outdir: str,
    impact_speed_mps: float,
    stroke_m: float,
    max_g: float,
    pre_freefall_s: float,
    post_s: float,
) -> tuple[SimulationResult, ProfileResult, dict[str, Any]]:
    """
    Run analytic mode with optional dt refinement.

    Returns:
      (final_sim, profile, auto_dt_meta)
    """
    dt = float(sim_cfg_base.dt)

    jerk_g_ref, profile_ref = compute_min_jerk(
        v0=impact_speed_mps, target_stroke=stroke_m, max_g=max_g
    )
    analytic_peak_g = profile_ref.peak_a / G

    history: list[dict[str, Any]] = []
    consecutive_passes = 0
    stop_reason = 'unknown'

    last = None

    for it in range(auto_cfg.max_iter):
        seat, profile = build_seat_kinematics_analytic(
            dt=dt,
            pre_freefall_s=pre_freefall_s,
            impact_speed_mps=impact_speed_mps,
            target_stroke_m=stroke_m,
            max_g=max_g,
            post_time_s=post_s,
        )
        sim_cfg = SimulationConfig(
            dt=dt,
            beta=sim_cfg_base.beta,
            gamma=sim_cfg_base.gamma,
            max_contact_iterations=sim_cfg_base.max_contact_iterations,
            seat_contact_deadband_m=sim_cfg_base.seat_contact_deadband_m,
            initial_condition=sim_cfg_base.initial_condition,
        )
        sim = run_simulation(model_cfg=model_cfg, sim_cfg=sim_cfg, seat=seat, profile=profile)
        m = compute_metrics(sim, early_ms=auto_cfg.early_ms)

        sampled_peak_g = float(m['seat_peak_g'])
        input_peak_rel_err = abs(sampled_peak_g - analytic_peak_g) / max(abs(analytic_peak_g), EPS)

        ramp_samples = float(profile.t1 / dt) if dt > 0 else float('inf')

        row: dict[str, Any] = {
            'dt_s': dt,
            'analytic_peak_g': float(analytic_peak_g),
            'sampled_peak_g': sampled_peak_g,
            'input_peak_rel_err': float(input_peak_rel_err),
            'ramp_samples': float(ramp_samples),
            'peak_impact_kn': float(m.get('t12l1_peak_impact_n', float('nan')) / 1000.0),
            'peak_post_kn': float(m.get('t12l1_peak_post_n', float('nan')) / 1000.0),
            'peak_full_kn': float(m.get('t12l1_peak_full_n', float('nan')) / 1000.0),
            'peak_dFdt_impact_knps': float(
                m.get('t12l1_peak_dFdt_impact_nps', float('nan')) / 1000.0
            ),
            'peak_early_kn': float(m.get('t12l1_peak_early_n', float('nan')) / 1000.0),
            'peak_dFdt_early_knps': float(
                m.get('t12l1_peak_dFdt_early_nps', float('nan')) / 1000.0
            ),
        }

        # Convergence checks
        rel_force = float('inf')
        rel_dFdt = float('inf')
        rel_early_force = float('inf')
        rel_early_dFdt = float('inf')

        if last is not None:
            rel_force = _relative_change(
                m.get('t12l1_peak_impact_n', float('nan')),
                last.get('t12l1_peak_impact_n', float('nan')),
            )
            rel_dFdt = _relative_change(
                m.get('t12l1_peak_dFdt_impact_nps', float('nan')),
                last.get('t12l1_peak_dFdt_impact_nps', float('nan')),
            )
            rel_early_force = _relative_change(
                m.get('t12l1_peak_early_n', float('nan')),
                last.get('t12l1_peak_early_n', float('nan')),
            )
            rel_early_dFdt = _relative_change(
                m.get('t12l1_peak_dFdt_early_nps', float('nan')),
                last.get('t12l1_peak_dFdt_early_nps', float('nan')),
            )

            row['rel_change_peak_impact'] = float(rel_force)
            row['rel_change_peak_dFdt_impact'] = float(rel_dFdt)
            row['rel_change_peak_early'] = float(rel_early_force)
            row['rel_change_peak_dFdt_early'] = float(rel_early_dFdt)

        input_ok = input_peak_rel_err <= auto_cfg.input_peak_rel_tol
        ramp_ok = ramp_samples >= float(auto_cfg.min_samples_per_ramp)

        metrics_ok = (
            (rel_force <= auto_cfg.force_rel_tol)
            and (rel_dFdt <= auto_cfg.dFdt_rel_tol)
            and (rel_early_force <= auto_cfg.early_force_rel_tol)
            and (rel_early_dFdt <= auto_cfg.early_dFdt_rel_tol)
        )

        pass_flags = []
        pass_flags.append('input_ok' if input_ok else 'input_fail')
        pass_flags.append('ramp_ok' if ramp_ok else 'ramp_fail')
        if last is None:
            pass_flags.append('no_prev')
        else:
            pass_flags.append('metrics_ok' if metrics_ok else 'metrics_fail')

        row['pass_flags'] = ','.join(pass_flags)
        history.append(row)

        # Decide stop / refine
        if not auto_cfg.enabled:
            stop_reason = 'disabled'
            auto_dt_meta = {
                'enabled': False,
                'config': auto_cfg.__dict__,
                'stop_reason': stop_reason,
                'history': history,
            }
            return sim, profile, auto_dt_meta

        if not ramp_ok:
            consecutive_passes = 0
        elif input_ok and (last is not None) and metrics_ok:
            consecutive_passes += 1
        else:
            consecutive_passes = 0

        if consecutive_passes >= auto_cfg.require_consecutive_passes:
            stop_reason = f'converged_{consecutive_passes}_passes'
            auto_dt_meta = {
                'enabled': True,
                'config': auto_cfg.__dict__,
                'stop_reason': stop_reason,
                'history': history,
            }
            return sim, profile, auto_dt_meta

        if dt / 2.0 < auto_cfg.min_dt:
            stop_reason = 'min_dt_reached'
            auto_dt_meta = {
                'enabled': True,
                'config': auto_cfg.__dict__,
                'stop_reason': stop_reason,
                'history': history,
            }
            return sim, profile, auto_dt_meta

        last = m
        dt *= 0.5

    stop_reason = 'max_iter_reached'
    auto_dt_meta = {
        'enabled': True,
        'config': auto_cfg.__dict__,
        'stop_reason': stop_reason,
        'history': history,
    }
    return sim, profile, auto_dt_meta


# -------------------------
# CLI
# -------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description='1D spine + buttocks + viscera simulation (base excitation).'
    )

    ap.add_argument('--mode', choices=['analytic', 'csv'], required=True)
    ap.add_argument('--outdir', default='out', help='Output directory for PNG and JSON files.')
    ap.add_argument('--debug', action='store_true', help='Print extra debug info.')

    ap.add_argument(
        '--dt',
        type=float,
        default=0.001,
        help='Initial time step [s]. In analytic mode with auto-dt, this is the starting dt.',
    )

    # Analytic mode (reference-like inputs)
    ap.add_argument('--impact-speed-mps', type=float, default=5.7)
    ap.add_argument('--stroke-m', type=float, default=0.105)
    ap.add_argument('--max-g', type=float, default=32.0)
    ap.add_argument('--pre-freefall-s', type=float, default=0.05)
    ap.add_argument('--post-s', type=float, default=0.10)

    ap.add_argument(
        '--early-ms',
        type=float,
        default=5.0,
        help='Early window length starting at impact start [ms] for jerk-sensitive metrics.',
    )

    ap.add_argument(
        '--initial-condition',
        choices=['rest', 'freefall'],
        default='freefall',
        help='Initial condition. freefall is recommended for jerk studies.',
    )

    # Auto-dt
    ap.add_argument('--auto-dt', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--auto-dt-max-iter', type=int, default=16)
    ap.add_argument('--auto-dt-min-dt', type=float, default=1e-6)
    ap.add_argument('--auto-dt-input-peak-rel-tol', type=float, default=0.005)
    ap.add_argument('--auto-dt-force-rel-tol', type=float, default=0.02)
    ap.add_argument('--auto-dt-dfdt-rel-tol', type=float, default=0.05)
    ap.add_argument('--auto-dt-early-force-rel-tol', type=float, default=0.02)
    ap.add_argument('--auto-dt-early-dfdt-rel-tol', type=float, default=0.08)
    ap.add_argument('--auto-dt-consecutive-passes', type=int, default=2)
    ap.add_argument('--auto-dt-min-samples-ramp', type=int, default=20)

    # CSV mode
    ap.add_argument('--csv', type=str, default=None)
    ap.add_argument('--resample-dt', type=float, default=None)

    # Model parameters
    ap.add_argument('--body-mass-kg', type=float, default=75.0)
    ap.add_argument('--disc-damping', type=float, default=1200.0)
    ap.add_argument('--viscera-damping', type=float, default=1200.0)
    ap.add_argument('--thoracolumbar-damp-mult', type=float, default=5.0)

    ap.add_argument(
        '--seat-contact-deadband-mm',
        type=float,
        default=0.001,
        help='Deadband for pelvis-seat coupling hysteresis [mm].',
    )

    ap.add_argument('--strap-k', type=float, default=2.0e5)
    ap.add_argument('--strap-c', type=float, default=1200.0)

    args = ap.parse_args()
    ensure_outdir(args.outdir)

    model_cfg = ModelConfig(
        body_mass_target_kg=args.body_mass_kg,
        disc_damping_ns_per_m=args.disc_damping,
        viscera_damping_ns_per_m=args.viscera_damping,
        thoracolumbar_disc_damping_mult=args.thoracolumbar_damp_mult,
        strap_k=args.strap_k,
        strap_c=args.strap_c,
    )

    seat_deadband_m = float(args.seat_contact_deadband_mm) / 1000.0

    sim_cfg_base = SimulationConfig(
        dt=float(args.dt),
        max_contact_iterations=30,
        seat_contact_deadband_m=seat_deadband_m,
        initial_condition=args.initial_condition,
    )

    early_ms = float(args.early_ms)

    profile = None
    auto_dt_meta: dict[str, Any] | None = None

    if args.mode == 'analytic':
        auto_cfg = AutoDtConfig(
            enabled=bool(args.auto_dt),
            max_iter=int(args.auto_dt_max_iter),
            min_dt=float(args.auto_dt_min_dt),
            input_peak_rel_tol=float(args.auto_dt_input_peak_rel_tol),
            force_rel_tol=float(args.auto_dt_force_rel_tol),
            dFdt_rel_tol=float(args.auto_dt_dfdt_rel_tol),
            early_force_rel_tol=float(args.auto_dt_early_force_rel_tol),
            early_dFdt_rel_tol=float(args.auto_dt_early_dfdt_rel_tol),
            require_consecutive_passes=int(args.auto_dt_consecutive_passes),
            min_samples_per_ramp=int(args.auto_dt_min_samples_ramp),
            early_ms=early_ms,
        )

        sim, profile, auto_dt_meta = run_analytic_with_auto_dt(
            model_cfg=model_cfg,
            sim_cfg_base=sim_cfg_base,
            auto_cfg=auto_cfg,
            outdir=args.outdir,
            impact_speed_mps=float(args.impact_speed_mps),
            stroke_m=float(args.stroke_m),
            max_g=float(args.max_g),
            pre_freefall_s=float(args.pre_freefall_s),
            post_s=float(args.post_s),
        )

    else:
        if not args.csv:
            raise SystemExit('--csv is required in csv mode.')
        seat = build_seat_kinematics_from_csv(csv_path=args.csv, resample_dt=args.resample_dt)

        dts = np.diff(seat.t)
        med_dt = float(np.median(dts)) if len(dts) > 0 else float(args.dt)

        sim_cfg = SimulationConfig(
            dt=med_dt,
            max_contact_iterations=30,
            seat_contact_deadband_m=seat_deadband_m,
            initial_condition=args.initial_condition,
        )

        sim = run_simulation(model_cfg=model_cfg, sim_cfg=sim_cfg, seat=seat, profile=None)

    # Compute reporting metrics
    m = compute_metrics(sim, early_ms=early_ms)

    peak_impact_kn = float(m.get('t12l1_peak_impact_n', float('nan')) / 1000.0)
    peak_post_kn = float(m.get('t12l1_peak_post_n', float('nan')) / 1000.0)
    peak_full_kn = float(m.get('t12l1_peak_full_n', float('nan')) / 1000.0)

    peak_dFdt_impact_knps = float(m.get('t12l1_peak_dFdt_impact_nps', float('nan')) / 1000.0)
    peak_early_kn = float(m.get('t12l1_peak_early_n', float('nan')) / 1000.0)
    peak_dFdt_early_knps = float(m.get('t12l1_peak_dFdt_early_nps', float('nan')) / 1000.0)

    # Save artifacts
    save_plots(args.outdir, sim, early_ms=early_ms)

    extra_meta: dict[str, Any] = {
        'metrics': m,
    }
    if auto_dt_meta is not None:
        extra_meta['auto_dt'] = auto_dt_meta

    save_json(args.outdir, sim, extra=extra_meta)

    # Print main report
    print(f'Output written to: {args.outdir}')
    print(f'Peak T12–L1 compressive force (impact window): {peak_impact_kn:.3f} kN')
    print(f'Peak T12–L1 compressive force (early {early_ms:.1f} ms): {peak_early_kn:.3f} kN')
    print(f'Peak T12–L1 compressive force (post window):   {peak_post_kn:.3f} kN')
    print(f'Peak T12–L1 compressive force (full sim):      {peak_full_kn:.3f} kN')
    print(f'Peak T12–L1 loading rate dF/dt (impact):       {peak_dFdt_impact_knps:.3f} kN/s')
    print(f'Peak T12–L1 loading rate dF/dt (early):        {peak_dFdt_early_knps:.3f} kN/s')
    print(
        f'I_axial peak (impact):                         {m.get("I_axial_peak_impact_mm", float("nan")):.3f} mm'
    )
    print(
        f'I_axial peak (early):                          {m.get("I_axial_peak_early_mm", float("nan")):.3f} mm'
    )

    if profile is not None:
        print(
            f'Analytic impact profile: {profile.profile_type}, '
            f'impactSpeed={profile.v0:.3f} m/s, stroke={profile.stop_distance:.3f} m, '
            f'peak={profile.peak_a / G:.2f} G, jerk={profile.jerk_g:.1f} G/s, '
            f'totalTime={profile.total_time * 1000:.3f} ms, ramp={profile.t1 * 1000:.3f} ms'
        )

    if args.mode == 'analytic' and auto_dt_meta is not None and auto_dt_meta.get('history'):
        dt_final = float(auto_dt_meta['history'][-1]['dt_s'])
        print(f'Auto-dt final dt: {dt_final:.12f} s')
        print(f'Auto-dt stop_reason: {auto_dt_meta.get("stop_reason")}')

    if args.debug:
        print_debug_summary(sim, profile, early_ms=early_ms, auto_dt_meta=auto_dt_meta)


if __name__ == '__main__':
    main()
