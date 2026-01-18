# Buttocks model (Toen 2012) — project state and spec (LLM-continuable)

## Goal

We want a **universal buttocks model** derived from Toen 2012 (backward fall onto buttocks) that:

1. Reproduces Toen’s conditions reasonably well at 3.5 m/s (paper average peak ground forces).
2. Can be used in **higher-energy paragliding drops (~8 m/s)** without producing anatomically impossible buttocks compression.
3. Uses a **single always-on force law** (no velocity-gated enable/disable bottom-out switch).
4. Is used as a fixed “buttocks layer” before calibrating the spine model against Yoganandan UBB FE base-acceleration pulses.

Key practical constraint: for high-energy cases, buttocks compression must not keep increasing indefinitely. It must saturate near a realistic maximum, implying forces rise instead.

---

## Important modeling choices (agreed)

### What is “buttocks compression”?

Buttocks compression refers to the **soft tissue thickness under the pelvis** in the lumped model:
- In the 2-node Toen drop model: `x_butt = max(-(y_body - y_skin), 0)`.

### Absolute maximum compression anchor

We treat:

- The **rigid floor (400 kN/m) case at 3.5 m/s** as defining the **absolute maximum** buttocks compression regime.

Decision: **Limit = observed peak compression under rigid_400 @ 3.5 m/s after calibration**.
High-energy impacts should not compress substantially beyond this; instead they should generate higher forces.

### No time-to-peak targets

We want to preserve time-to-peak qualitatively, but we do not digitize explicit t_peak targets.
Calibration is **peak-force-only**.

### Calibration floors

For buttocks model calibration, we **ignore soft_59** (and generally the very compliant floors in fitting),
and focus on **firm_95 + rigid_400**.

### Gravity-settled “static sitting” reference

We compute a “gravity-settled” static equilibrium for each floor configuration and report:

- static buttocks compression (mm)
- delta = max_dynamic - static

This is for reporting/intuition only (not used as the cap).

### Sensor / boundary condition for paragliding drops

Paragliding drops use **base acceleration replay** on a rigid excitor plate:
- accelerometer is mounted to the plate (base excitation).
- protectors are not explicitly modeled here; their effect is in the recorded acceleration.

Buttocks can be compression-only and is allowed to lose contact during rebound (not of interest).

---

## Buttocks constitutive law used

We use a Voigt (spring+damper) buttocks element augmented with a **smooth densification term**
implemented via the existing `SpineModel.compression_limit_m` / `compression_stop_k` mechanism.

This densification is **always part of the force law** (no velocity gating).
By tuning `limit_mm`, `stop_k`, and `smoothing_mm`, we can make compression saturate near the limit.

Key tuning caution: smoothing must be small enough (e.g., ~1 mm) so the stop does not affect forces far below the limit.

---

## Calibration algorithm (Toen-based, new)

We calibrate 3 parameters:

- buttocks linear stiffness `k`
- buttocks damping `c`
- buttocks limit `limit_mm`

Fixed shape parameters (typically):
- `stop_k_n_per_m` (large-ish, e.g. 5e6)
- `stop_smoothing_mm` (small, e.g. 1.0)

Residuals:
1. Peak ground force error for `firm_95` at 3.5 m/s (relative error)
2. Peak ground force error for `rigid_400` at 3.5 m/s (relative error)
3. Self-consistency constraint: `max_buttocks_comp_mm(rigid_400) - limit_mm` (scaled by 1 mm)

This enforces: rigid_400 @ 3.5 m/s **reaches** the cap.

Calibration output is stored in `calibration/toen_drop.json` and is loaded by `simulate.py simulate-buttock`.

---

## How to run

### 1) Calibrate buttocks model from Toen (paper averages)

```bash
./simulate.py calibrate-buttock
```

This writes `calibration/toen_drop.json`.

### 2) Run Toen suite simulation

```bash
./simulate.py simulate-buttock
```

Outputs are written to `output/toen_drop/summary.json`.

---

## Environment variable overrides (inside library functions)

You can override the Toen suite without changing `simulate.py` or `config.json`.

### Common overrides

- `SPINE_SIM_TOEN_VELOCITIES_MPS="3.5,8.0"`
- `SPINE_SIM_TOEN_TARGET_SET="avg"` (paper averages)
- `SPINE_SIM_TOEN_DT_S=0.0005`
- `SPINE_SIM_TOEN_DURATION_S=0.15`
- `SPINE_SIM_TOEN_MAX_NEWTON_ITER=10`

### Buttocks model overrides

- `SPINE_SIM_TOEN_BUTTOCKS_K_N_PER_M=180500`
- `SPINE_SIM_TOEN_BUTTOCKS_C_NS_PER_M=3130`
- `SPINE_SIM_TOEN_BUTTOCKS_LIMIT_MM=39.0`
- `SPINE_SIM_TOEN_BUTTOCKS_STOP_K_N_PER_M=5000000`
- `SPINE_SIM_TOEN_BUTTOCKS_STOP_SMOOTHING_MM=1.0`

### Calibration overrides

- `SPINE_SIM_TOEN_CALIB_FLOORS="firm_95,rigid_400"`
- `SPINE_SIM_TOEN_BUTTOCKS_STOP_K_N_PER_M=5000000`
- `SPINE_SIM_TOEN_BUTTOCKS_STOP_SMOOTHING_MM=1.0`

### Example: run 8 m/s without editing config

```bash
SPINE_SIM_TOEN_VELOCITIES_MPS=8.0 ./simulate.py simulate-buttock
```

---

## Current known baseline behavior (before densification)

A purely linear buttocks model (k=180.5 kN/m, c=3.13 kNs/m) produces unrealistic compression at high velocity:

- 3.5 m/s, rigid_400: butt_comp ~39 mm, peak_ground ~9.5 kN (vs target 7.8 kN)
- 8.0 m/s, rigid_400: butt_comp ~86 mm (impossible), peak_ground ~21 kN

Densification is required to prevent unlimited compression.

---

## Next step (not yet implemented here)

Once the buttocks model is calibrated and fixed, we will:
- keep buttocks fixed,
- calibrate spine parameters/scales against Yoganandan UBB FE base-acceleration cases.

This README is meant to preserve the exact agreed spec/state so an LLM can continue from here.
