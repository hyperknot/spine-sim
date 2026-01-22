# Spine-Sim: 1D Axial Spine Impact Simulation

This repo simulates a **1D axial spine compression event** driven by a measured base acceleration time history (accelerometer placed under the buttocks on the skin surface). The primary output is the **T12–L1 junction force** time history.

---

## What this model is

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
[Base/Seat Plate] --(buttocks contact)--> pelvis -- L5 -- ... -- T1 -- HEAD
```

- Nodes are lumped masses from an OpenSim body-mass JSON (`opensim/fullbody.json`).
- The OpenSim file used here provides `head_neck` as a single lump; we treat that as the `HEAD` node (optionally adding helmet mass + recruited arm mass per config).
- Cervical vertebrae are **not** separate nodes in this simplified version.

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
- constant damping `c` (global, all IVDs),
- tension stiffness is reduced by `tension_k_mult` to represent non-disc tensile structures while keeping the chain connected.

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

## Parameter provenance (where the “magic numbers” come from)

This section documents **every constant in `config.json`** and the key **hard-coded constants in code** that affect physics or signal processing. Plot-only cosmetics (line widths, colors, etc.) are not exhaustively listed.

### A) `config.json` parameters

#### `model.*`
- `model.masses_json` (`opensim/fullbody.json`)
  - **Source:** external OpenSim-derived mass data file (repo data). Not a literature constant.
  - **Used in:** `spine_sim/mass.py` via `build_mass_map()`.
- `model.arm_recruitment` (0.5)
  - **Meaning:** fraction of total arm mass that is assumed to load the spine (added at T1).
  - **Source:** **not found in literature** in this repo; it is a modeling assumption knob.
  - **Used in:** `spine_sim/mass.py`.
- `model.helmet_mass_kg` (0.7)
  - **Meaning:** extra mass added to HEAD.
  - **Source:** **not found in literature** in this repo; scenario-dependent assumption.
  - **Used in:** `spine_sim/mass.py`.

#### `solver.*`
- `solver.dt_internal_s` (0.00005 s = 0.05 ms = 20 kHz)
  - **Meaning:** internal integrator timestep.
  - **Source:** **not found in literature**; chosen for numerical stability / temporal resolution.
  - **Used in:** `spine_sim/drop_commands.py` → `_interpolate_to_internal_dt()` → `newmark_nonlinear()`.
- `solver.max_newton_iter` (25)
  - **Meaning:** Newton iterations per time step.
  - **Source:** **not found in literature**; typical nonlinear solve guard.
  - **Used in:** `spine_sim/model.py:newmark_nonlinear()`.
- `solver.newton_tol` (1e-9)
  - **Meaning:** displacement increment norm tolerance used to terminate Newton iterations.
  - **Source:** **not found in literature**; numerical tuning value.
  - **Used in:** `spine_sim/model.py:newmark_nonlinear()`.

#### `drop.*` (input processing and run window)
- `drop.cfc` (1000)
  - **Meaning:** target “CFC” class for low-pass filtering the input acceleration.
  - **Reference (standard):** SAE J211/1 CFC filtering convention (see “References”).
  - **Implementation detail:** code uses `f_design_hz = 2.0775 * cfc` with a forward/backward 2nd-order Butterworth per pass (effective 4th order), in `spine_sim/filters.py`.
  - **Used in:** `spine_sim/input_processing.py` → `cfc_filter()`.
- `drop.style_duration_threshold_ms` (300)
  - **Meaning:** if input duration < threshold => treat as “flat” style (pulse-only) else “drop”.
  - **Source:** **not found in literature**; repo-specific data-format heuristic.
  - **Used in:** `spine_sim/input_processing.py:detect_style()`.
- `drop.sim_duration_ms` (200)
  - **Meaning:** simulation window length after detected hit start (or after pulse start for “flat”).
  - **Source:** **not found in literature**; chosen to capture early peak and keep runtime bounded.
  - **Used in:** `spine_sim/drop_commands.py`, plus plotting uses a separate constant `PLOT_DURATION_MS=200.0`.
- `drop.gravity_settle_ms` (150)
  - **Meaning:** pre-impact “settling” integration time for “flat” inputs (helps start near equilibrium).
  - **Source:** **not found in literature**; numerical convenience.
  - **Used in:** `spine_sim/drop_commands.py` when `style == "flat"`.
- `drop.peak_threshold_g` (5.0)
  - **Meaning:** threshold to find the first impact peak in “drop” style data.
  - **Source:** **not found in literature**; heuristic for event detection.
  - **Used in:** `spine_sim/range.py:find_hit_range()`.
- `drop.freefall_threshold_g` (-0.85)
  - **Meaning:** value below which the signal is treated as “freefall region” for hit bracketing.
  - **Source:** **not found in literature**; heuristic around the expected -1 g freefall reading.
  - **Used in:** `spine_sim/range.py:find_hit_range()`.

#### `buttock.*` (buttocks contact element)
- `buttock.k1_n_per_m` (180500.0 N/m)
  - **Source (literature):** Van Toen et al. report across-subject mean buttock stiffness 180.5 kN/m for backward falls onto buttocks.
  - **Mapping:** 180.5 kN/m → 180500 N/m.
  - **Used in:** `spine_sim/model_components.py` → `SpineModel.buttocks_k1_n_per_m`.
  - **Reference:** Van Toen et al., *Spine*, 2012 (see “References”).
- `buttock.c_ns_per_m` (3130.0 Ns/m)
  - **Source (literature):** Van Toen et al. report across-subject mean buttock damping 3.13 kN·s/m.
  - **Mapping:** 3.13 kN·s/m → 3130 N·s/m.
  - **Used in:** `spine_sim/model_components.py` → `SpineModel.buttocks_c_ns_per_m`.
  - **Reference:** Van Toen et al., *Spine*, 2012.
- `buttock.bottom_out_force_kN` (99997.0 kN)
  - **Meaning:** force threshold where the buttocks spring switches from `k1` to `k2`.
  - **Observed implication:** with k1=180500 N/m, this implies bottom-out compression:
    - x0 = (99997 kN * 1000) / 180500 ≈ 554,000 m (nonsensical physically),
    - so bottom-out is effectively **disabled**.
  - **Source:** **not found in literature**; appears to be a “disable bottom-out” sentinel.
  - **Used in:** `spine_sim/model.py:SpineModel.buttocks_bottom_out_compression_m()` and `_buttocks_force_and_partials()`.
- `buttock.k2_n_per_m` (5000000.0 N/m)
  - **Meaning:** post-bottom-out stiffness.
  - **Source:** **not found in literature** in this repo; a modeling choice.
  - **Used in:** `_buttocks_force_and_partials()`.

#### `spine.*` (disc elements)
- `spine.stiffness_scale` (1.0)
  - **Meaning:** uniform multiplier on all baseline disc stiffness values.
  - **Source:** **not found in literature**; calibration knob.
  - **Used in:** `spine_sim/model_components.py`.
- `spine.disc_height_mm` (11.3 mm)
  - **Source (literature):** Kemper et al. report using an overall average initial disc height of 11.32 mm (combined from their study and Keller et al. disc heights) for strain-rate calculations.
  - **Mapping:** 11.32 mm → default 11.3 mm.
  - **Used in:** `spine_sim/model.py:_disc_strain_rate_per_s()`.
  - **Reference:** Kemper et al. “Stiffness Properties…” (paper in repo as `kemper_2013_stiffness_properties.md`).
- `spine.tension_k_mult` (0.1)
  - **Meaning:** tension stiffness = compression stiffness * tension_k_mult.
  - **Source:** **not found in literature** in this repo; chosen to keep the chain connected while weakening tension response.
  - **Used in:** `spine_sim/model.py:_disc_force_and_partials()`.
- `spine.damping_ns_per_m` (1200.0 Ns/m)
  - **Meaning:** Kelvin–Voigt dashpot coefficient used for all disc elements (symmetric damping).
  - **Literature check:** Van Toen et al. used 237 Ns/m for lumbosacral joint damping (Table 1), citing Izambert et al. (2003). The value 1200 Ns/m is **not traceable** to those cited values in the provided papers.
  - **Source:** **not found in literature** in this repo; modeling/numerical choice.
  - **Used in:** `spine_sim/model_components.py` → `c_elem_ns_per_m[1:]`.

#### `spine.kemper.*` (rate-dependent stiffness scaling)
- `spine.kemper.normalize_to_eps_per_s` (0.0 1/s)
  - **Meaning:** reference strain rate for “baseline stiffness” in the multiplier \(s(\dot{\epsilon}) = k(\dot{\epsilon})/k(\dot{\epsilon}_{norm})\).
  - **Source:** **not found in literature**; a convention to make baseline match quasi-static (0 1/s) within this model.
  - **Used in:** `spine_sim/model.py:_disc_k_multiplier()`.
- `spine.kemper.strain_rate_smoothing_tau_ms` (2.0 ms)
  - **Meaning:** low-pass time constant for strain-rate used to compute disc stiffness per timestep.
  - **Source:** **not found in literature**; numerical stability knob.
  - **Used in:** `spine_sim/model.py:_alpha_lp()` and `newmark_nonlinear()`.
- `spine.kemper.warn_over_eps_per_s` (73.0 1/s)
  - **Meaning:** warning threshold only (no clamp).
  - **Source (literature-adjacent):** Kemper et al. report a mean strain rate around 72.7 1/s for their highest-rate tests.
  - **Used in:** `spine_sim/drop_commands.py` (warning print) and `spine_sim/model_components.py` → `SpineModel.warn_over_eps_per_s`.
  - **Reference:** Kemper et al. (“Stiffness Properties…”).

#### `plotting.*`
- `plotting.buttocks_height_mm` (50.0 mm)
  - **Meaning:** plotting geometry offset (only affects visualization when OpenSim heights aren’t available).
  - **Source:** **not found in literature**; cosmetic/plot layout parameter.
  - **Used in:** `spine_sim/plotting.py` and `spine_sim/drop_commands.py:_get_plotting_config()`.

---

### B) Baseline spine stiffness distribution (`k0`): where those numbers come from

The per-level baseline stiffnesses in `spine_sim/model_components.py`:

```python
k = {
  'HEAD-T1': 1.334e6,
  'T1-T2': 0.7e6,
  ...
  'L5-S1': 1.47e6,
}
```

match the **axial stiffness** values in **Kitazaki & Griffin (1997)** Table 4 (their spinal beams’ axial stiffness, reported as \(N/m \times 10^6\)). In their model, these stiffnesses are part of a vibration-focused FE model, and are described as being derived from prior modeling sources (Belytschko & Privitzer) and other data, then tuned to match observed mode shapes.

**Important interpretation note:** these are not direct cadaver-measured quasi-static disc stiffnesses; they are stiffness parameters used inside a vibration model (still “literature”, but *model-derived*).

Reference: `kitazaki_1996_vibration_modal.md` (J. Sound and Vibration, 1997).

---

### C) Key hard-coded constants in code (physics / numerics / standards)

- `G0 = 9.80665` m/s² (`spine_sim/model.py`)
  - **Source:** standard gravity (by definition / standard constant).
- Newmark parameters `beta=0.25`, `gamma=0.5` (`spine_sim/model.py:newmark_nonlinear`)
  - **Source:** standard “average acceleration” Newmark scheme (common structural dynamics default).
  - **Not a biomechanical literature parameter**; numerical method choice.
- Kemper stiffness-law constants (`spine_sim/model.py:kemper_k_n_per_m`)
  - `57.328` and `2019.1` in `k = 57.328*eps_dot + 2019.1` (N/mm)
  - **Source:** Kemper et al. curve fit (see “References”).
- CFC design multiplier `2.0775` (`spine_sim/filters.py`)
  - **Intended source:** SAE J211/1 CFC convention / implementation choice for mapping “CFC” to a Butterworth design frequency in this repo.
  - If you need strict compliance to a specific standard (SAE vs ISO 6487) we should explicitly validate the frequency response.
- Contact damping “closing-only” logic (`spine_sim/model.py:_buttocks_force_and_partials`)
  - **Source:** numerical contact-safety convention (prevents tensile damping forces); not literature-derived.
- Padding with `-1.0 g` after the extracted window (`spine_sim/input_processing.py`)
  - **Source:** physical assumption for freefall reading (accelerometer reads ~-1 g in freefall), but the *padding strategy* is a repo-specific signal handling choice.
- Various solve-loop constants (all **not found in literature**):
  - static equilibrium Newton loop max 80 iterations, relaxation factor 0.8, tolerances 1e-8 / 1e-10,
  - nonlinear Newmark Newton relaxation 0.8.

---

## Values NOT traced to literature (in this repo)

These values were **not found in the provided papers/standards** and should be treated as **assumptions, heuristics, or numerical tuning**:

### Modeling assumptions
- `model.arm_recruitment = 0.5`
- `model.helmet_mass_kg = 0.7`
- `spine.tension_k_mult = 0.1`
- `spine.damping_ns_per_m = 1200.0` (not matching Izambert-derived 237 Ns/m used by Van Toen et al.)
- `spine.stiffness_scale = 1.0` (calibration knob)
- Buttocks bottom-out parameters as configured:
  - `buttock.bottom_out_force_kN = 99997.0` (effectively disables bottom-out)
  - `buttock.k2_n_per_m = 5000000.0`

### Event detection / signal-processing heuristics
- `drop.style_duration_threshold_ms = 300.0`
- `drop.sim_duration_ms = 200.0`
- `drop.gravity_settle_ms = 150.0`
- `drop.peak_threshold_g = 5.0`
- `drop.freefall_threshold_g = -0.85`

### Numerical/stability knobs
- `solver.dt_internal_s = 0.00005`
- `solver.max_newton_iter = 25`
- `solver.newton_tol = 1e-9`
- `spine.kemper.normalize_to_eps_per_s = 0.0` (normalization convention)
- `spine.kemper.strain_rate_smoothing_tau_ms = 2.0`
- Any Newton relaxation factors / iteration caps inside code

### Plotting-only layout constants
- `plotting.buttocks_height_mm = 50.0`
- `spine_sim/plotting.py:PLOT_DURATION_MS = 200.0`
- `spine_sim/plotting.py:DEFAULT_BUTTOCKS_HEIGHT_MM = 100.0`

---

## Known limitations
- 1D axial only: no bending, shear, rotation.
- Cervical masses are approximated (not taken from OpenSim directly).
- Damping is constant despite large literature scatter and frequency dependence.
- No injury criterion validation is included in this simplified version.

---

## References

1. **Van Toen, C.**, Sran, M. M., Robinovitch, S. N., Cripton, P. A. (2012). *Transmission of Force in the Lumbosacral Spine During Backward Falls*. **Spine**.
   - Used for buttocks stiffness/damping: k ≈ 180.5 kN/m, c ≈ 3.13 kN·s/m.
   - In repo: `toen_2012_backward.md`.

2. **Kitazaki, S.**, Griffin, M. J. (1997). *A modal analysis of whole-body vertical vibration, using a finite element model of the human body*. **Journal of Sound and Vibration**, 200(1), 83–103.
   - Used for baseline axial stiffness distribution per level (Table 4 axial stiffness values).
   - In repo: `kitazaki_1996_vibration_modal.md`.

3. **Kemper, A. R.**, McNally, C., Manoogian, S., McNeely, D., Duma, S. (paper number 07-0471). *Stiffness Properties of Human Lumbar Intervertebral Discs in Compression and the Influence of Strain Rate*.
   - Used for stiffness vs strain-rate law: k = 57.328*eps_dot + 2019.1 (N/mm).
   - Used for representative disc height ≈ 11.32 mm.
   - In repo: `kemper_2013_stiffness_properties.md`.

4. **Izambert, O.**, Mitton, D., Thourot, M., et al. (2003). *Dynamic stiffness and damping of human intervertebral disc using axial oscillatory displacement under a free mass system*. **European Spine Journal**, 12, 562–566.
   - Mentioned because Van Toen et al. cite it for joint damping (237 Ns/m).
   - (Not directly used for the current 1200 Ns/m setting.)

5. **SAE J211/1**. *Instrumentation for Impact Test — Part 1 — Electronic Instrumentation*.
   - Reference standard for “CFC” channel filtering terminology (implementation details should be validated for strict compliance).

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
```

## Values I did *not* find in literature (summary)

From the repo and the provided papers, I could **not** trace the following to a specific literature source:

- `arm_recruitment = 0.5`
- `helmet_mass_kg = 0.7`
- `dt_internal_s = 0.00005`, `max_newton_iter = 25`, `newton_tol = 1e-9`
- Drop/event-detection heuristics: `style_duration_threshold_ms = 300`, `sim_duration_ms = 200`, `gravity_settle_ms = 150`, `peak_threshold_g = 5`, `freefall_threshold_g = -0.85`
- Disc damping `damping_ns_per_m = 1200` (does not match the 237 Ns/m used in Van Toen et al. from Izambert et al.)
- Tension behavior knob `tension_k_mult = 0.1`
- Strain-rate smoothing `strain_rate_smoothing_tau_ms = 2.0`
- Buttocks bottom-out configuration as shipped:
  - `bottom_out_force_kN = 99997` (acts like “disable bottom-out”),
  - `k2_n_per_m = 5e6`
- Plotting layout parameters such as `buttocks_height_mm`, `PLOT_DURATION_MS`, default spacing constants

If you want, I can also add a short “Recommended defaults” subsection that separates *physics parameters* (literature-based) from *engineering/numerical knobs* (timestep, smoothing, thresholds) so readers don’t confuse them.
