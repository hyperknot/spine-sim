# Spine-Sim (Simplified): 1D Axial Spine Impact Simulation (Onset-Rate Focus)

This repo simulates a **1D axial spine compression event** driven by a measured base acceleration time history (accelerometer placed **under the buttocks on the skin surface**). The primary output is the **T12–L1 junction force** time history.

This version has been **deliberately simplified**:
- No ZWT/Maxwell branches.
- No calibration pipelines.
- One CLI command: `simulate-drop`.
- All parameters live in `config.json`.

---

## What this model is (for sharing with an LLM / preserving decisions)

### Goal
Predict how **impact onset-rate** (rise time / jerk) changes the internal spine loading, especially the **early peak T12–L1 force**.

We explicitly do *not* try to match any Yoganandan reference targets in this simplified version.

### Coordinate / sign conventions
- Axis is vertical only (1D).
- **Compression forces are positive**.
- Base acceleration input is in **g**.
- Gravity is included consistently:
  - In freefall, the accelerometer reads about **-1 g**.
  - The base inertial term uses `G0 + base_accel`.

### System topology
A serial chain of masses connected by 1D elements:

```
[Base/Seat Plate] --(buttocks contact)--> pelvis -- L5 -- ... -- T1 -- C7 -- ... -- C1 -- HEAD
```

- Nodes are lumped masses from an OpenSim body-mass JSON (`opensim/fullbody.json`).
- Cervical nodes C1..C7 are explicit in this simplified version. Because OpenSim does not provide separate cervical vertebra masses in this file, we allocate a small configurable mass to each cervical vertebra and subtract it from the head/neck lump mass.

---

## Input signal interpretation (important)
- The input CSV acceleration is assumed to be measured at the **bottom of the buttocks element** (skin surface under the buttocks).
- This acceleration is applied as a **base excitation** to the entire chain via inertial loading.
- The buttocks element is a **contact element** against the base. When there is no contact, it produces **zero force**, allowing “flight” / separation.

---

## Element models

### 1) Buttocks element (contact + bilinear bottom-out)
**Purpose:** represent soft tissue compliance up to a “bottom-out” transition where load transmits much more directly through the pelvis.

Model behavior:
- **Compression-only contact**:
  - If the pelvis separates from the base (tension), buttocks force is **0**.
- **Bilinear spring**:
  - stiffness `k1` up to a bottom-out point,
  - stiffness `k2` after bottom-out (hard kink, no smoothing).
- **Bottom-out is configured in force (kN)**:
  - `bottom_out_force_kN` is the force threshold where the spring switches from `k1` to `k2`.
  - The implied displacement threshold is computed as:
    - `bottom_out_compression_mm = bottom_out_force_N / k1 * 1000`.
- **Damping**:
  - viscous damping `c` is **contact-only** and **closing-only** (it never “pulls” the pelvis toward the base during separation).

### 2) Intervertebral disc (IVD) elements: Kelvin–Voigt with rate-dependent stiffness
Each disc element is:
- spring + dashpot (Kelvin–Voigt),
- constant damping `c = 1200 Ns/m` (global, all IVDs),
- tension stiffness is reduced by `tension_k_mult = 0.1` to represent non-disc tensile structures while keeping the chain connected.

#### Strain rate computation
For each disc element at each timestep:
- Compute relative velocity across the element: `relv = v_upper - v_lower`.
- Compression rate is clamped to compression-only: `compression_rate = max(-relv, 0)`.
- Strain rate:
  - eps_dot ≈ compression_rate / h0
  - h0 is a constant `disc_height_mm` (default 11.3 mm) applied to all discs.

#### Kemper strain-rate stiffness law (used as a multiplier)
Kemper relation:
- kK(eps_dot) = 57.328 * eps_dot + 2019.1 (N/mm), eps_dot in 1/s
- converted internally to N/m.

We apply Kemper as a multiplier on the baseline per-level stiffness distribution:
- s(eps_dot) = kK(eps_dot) / kK(eps_norm)
- k_disc(t) = k0_disc * s(eps_dot_smoothed)

Default `eps_norm = 0 1/s` so baseline stiffness corresponds to quasi-static.

#### Smoothing (numerical stability)
Strain rate is low-pass filtered with a first-order filter:
- time constant `strain_rate_smoothing_tau_ms` (default 2 ms).
- Disc stiffness is updated once per timestep using the smoothed strain rate and then **frozen for the Newton iterations** in that timestep (“frozen stiffness per timestep”).

#### No clamping (but warnings)
We do not clamp strain rate. We **warn** if any element exceeds `warn_over_eps_per_s` (default 73 1/s) and report which elements exceeded it and their maxima.

---

## Numerical integration
- Newmark-beta (beta=0.25, gamma=0.5) with Newton iterations per step.
- The simulation always runs at an internal fixed timestep `solver.dt_internal_s` (default 0.05 ms = 20 kHz).
- The input acceleration is resampled and CFC-filtered, then **interpolated** to the internal time grid.

CPU time is intentionally not optimized; stability and smooth k(t) behavior take priority.

---

## Running the simulation

### Install
Python 3.11+ with `uv` recommended.

```bash
uv sync
```

### Run
```bash
./simulate.py simulate-drop
```

### Input format
Place CSV files in `drops/` (configurable), with columns:
- `time` (or `time0` or `t`) in seconds or milliseconds
- `accel` (or `acceleration`) in g

### Outputs
For each drop file `<name>.csv`:
- `output/drop/<name>/timeseries.csv`
- `output/drop/<name>/displacements.png`
- `output/drop/<name>/forces.png`
- `output/drop/<name>/mixed.png`

And a summary:
- `output/drop/summary.json`

The CSV includes:
- node displacements/velocities/accelerations
- element forces (kN)
- per-element strain rate (1/s)
- per-element dynamic stiffness (MN/m)

---

## Configuration reference (`config.json`)

Key knobs:
- `solver.dt_internal_s`: internal timestep (e.g., 0.00005 for 20 kHz)
- `buttock.k1_n_per_m`, `buttock.k2_n_per_m`, `buttock.bottom_out_force_kN`, `buttock.c_ns_per_m`
- `spine.damping_ns_per_m` (default 1200)
- `spine.disc_height_mm` (default 11.3)
- `spine.tension_k_mult` (default 0.1)
- `spine.kemper.strain_rate_smoothing_tau_ms` (default 2.0)
- `spine.kemper.warn_over_eps_per_s` (default 73)

---

## Known limitations
- 1D axial only: no bending, shear, rotation.
- Cervical masses are approximated (not taken from OpenSim directly).
- Damping is constant despite large literature scatter and frequency dependence.
- No injury criterion validation is included in this simplified version.

---

## References
- Kemper et al. (lumbar IVD stiffness vs strain rate): k = 57.328*eps_dot + 2019.1 (N/mm)
- Izambert et al. 2003 (dynamic stiffness/damping of lumbar IVD, 5–30 Hz)
- Raj/Kitazaki baseline axial stiffness distribution (used as k0 per level)
```

---

## One more thing you should delete/update
### `spine_sim/simulation.py`
It currently re-exports calibration/buttocks modules. You can delete it, or reduce it to nothing. Since `simulate.py` now imports directly from `drop_commands`, it won’t be used.

If you want, here is a minimal safe version:

```python
"""Compatibility module (no longer used in simplified repo)."""
```

---

## Sanity checks you should run
1. `./simulate.py --help` should show only `simulate-drop`.
2. Run on one CSV and confirm:
   - `T12-L1` force output exists.
   - Buttocks force goes to 0 when pelvis separates (free flight).
   - Summary prints implied bottom-out compression (mm) computed from `bottom_out_force_kN`.
   - Warning prints if any disc strain rate exceeds 73 1/s.

---

## Anything else needed?
Two quick follow-ups that affect realism but weren’t explicitly specified:

1. Do you want **buttocks damping to act during opening while still compressed** (i.e., allow damping for both signs as long as contact exists), or keep it “closing-only” as implemented?
   - Current implementation is the usual contact-safe choice (prevents tensile damping forces).

2. Should the **Kemper stiffness multiplier** apply equally in tension, or only in compression?
   - Current implementation applies it to the disc’s base k(t), and tension uses `tension_k_mult * k(t)`.

If you want different behavior for either, say so and I’ll adjust.
