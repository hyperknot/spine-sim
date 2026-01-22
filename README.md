# SpineSim — Axial spine loading from paragliding harness drop tests

SpineSim is a research-oriented Python tool that simulates **1D (axial) vertical spine compression** under **base acceleration excitation** (e.g., excitor plate or drop-test sled). It is intended for *exploration* of how **onset-rate / jerk** and **buttocks “bottom-out”** assumptions influence predicted internal spine forces, with a focus on the **thoracolumbar junction force (T12–L1)**.

This repository is not a medically validated injury predictor. It is a transparent, configurable model to support sensitivity analysis and data-driven exploration.

---

## Scientific motivation and intended use

Paragliding harness drop tests can produce short-duration, high-onset vertical accelerations at the seat/harness interface. Two model questions motivate this codebase:

1. **Onset-rate sensitivity:** how do rise time / jerk affect peak internal forces at T12–L1, even when peak acceleration is similar?
2. **Buttocks mechanics sensitivity:** how do buttocks stiffness, damping, and *bottom-out* threshold influence T12–L1 peak force?
3. **Practical engineering output:** provide a reproducible pipeline to turn real-world harness acceleration data into a **peak force (kN)** estimate at the thoracolumbar junction under this model’s assumptions.

---

## Model overview (1D axial chain)

### Topology
The body is modeled as a serial chain of lumped masses connected by 1D viscoelastic elements:

```
[Base / excitor plate] -- buttocks(contact) -- pelvis -- L5 -- ... -- T1 -- HEAD
```

- **Masses** are taken from an OpenSim model (Bruno et al. thoracolumbar spine & rib cage model).
- **Buttocks element** is a unilateral (compression-only) contact element with a bilinear stiffness to represent “bottom-out”.
- **Intervertebral discs (IVDs)** are Kelvin–Voigt (spring + dashpot) elements, with **rate-dependent compression stiffness** via Kemper et al.

### Coordinate and sign conventions
- Motion is **vertical only**.
- **Compression force is positive**.
- Base acceleration input is in **g**.
- Gravity is included. In ideal freefall, an accelerometer reads approximately **-1 g**; the code uses this convention when padding signals.

---

## Input data model

### What the acceleration represents
The input acceleration time history is treated as the **base excitation** applied to the chain (excitor plate / harness interface). Internally, this acts through inertia: each mass experiences base-driven inertial loading.

### CSV format
Input CSV files live in `drops/` and should contain:
- a time column: `time` (or `time0` or `t`) in seconds or milliseconds,
- an acceleration column: `accel` (or `acceleration`) in **g**.

The loader will:
- resample to a uniform time grid (based on median `dt`),
- apply a CFC filter (see below),
- extract a simulation window based on heuristics.

---

## Outputs
For each input file `drops/<name>.csv`, results are written under:

- `output/<name>/timeseries.csv` (time histories)
- `output/<name>/displacements.png`
- `output/<name>/forces.png`
- `output/<name>/mixed.png`
- `output/summary.json` (batch summary)

The timeseries includes (per time sample):
- node displacements/velocities/accelerations,
- element forces (kN),
- per-element strain rate (1/s),
- per-element dynamic stiffness (MN/m).

---

## Running

### Install
Recommended: Python 3.11+.

This repo is commonly run using `uv` (see shebang in `simulate.py`). If you use another environment manager, ensure you have at least:
- numpy
- scipy
- matplotlib

### Run single simulation over all CSV files in `drops/`
```bash
./simulate.py
```

### Batch mode (buttocks sensitivity sweep)
Batch mode runs multiple configurations and writes separate summary JSON files.

```bash
./simulate.py --batch
```

Batch mode varies:
- buttocks stiffness/damping triplets derived from Van Toen et al. (see provenance below),
- bottom-out threshold:
  - `7.0 kN` (enabled),
  - `9999.0 kN` (“unlimited”: effectively disables bottom-out).

---

## Signal processing

### CFC filtering (SAE J211/1-style implementation)
Input acceleration is low-pass filtered using an SAE-style CFC mapping implemented as:

- 2nd-order Butterworth low-pass
- applied forward + backward (`sosfiltfilt`) to obtain zero-phase response

Implementation: `spine_sim/filters.py:cfc_filter()`.

Design frequency mapping used:
- SAE-style: `f_design_hz = 2.0775 * CFC`.

**Reference standard:** SAE J211/1, *Instrumentation for Impact Test — Part 1 — Electronic Instrumentation*.

> Note: ISO 6487 uses a slightly different constant. This repo intentionally implements the SAE-style mapping.

### Event detection and windowing heuristics (non-literature)
The tool supports two “styles” of inputs:

- **flat**: short pulse-like signal (typical excitor plate synthetic inputs)
- **drop**: longer signals containing a freefall region and an impact peak

Heuristic choices (configured in `config.json`, implemented in `spine_sim/input_processing.py` and `spine_sim/range.py`):
- `drop.style_duration_threshold_ms`: if duration < threshold → treat as `flat`, else `drop`.
- `drop.peak_threshold_g`: first sample above this is treated as the start of an “impact peak” region.
- `drop.freefall_threshold_g`: used to bracket the impact by searching outward for “freefall-like” samples.
- `drop.sim_duration_ms`: the simulation window length extracted after the detected start index.
- `drop.gravity_settle_ms`: optional pre-simulation settling time for `flat` style.

Padding behavior (implementation detail, `spine_sim/input_processing.py`):
- if the extracted segment is shorter than `sim_duration_ms`, the signal is padded with **-1.0 g** (freefall convention).

These are application-specific heuristics, intended to be easy to change per dataset.

---

## Element models

### 1) Buttocks element: compression-only + bilinear “bottom-out”
Configured in `config.json` under `buttock.*`, implemented in `spine_sim/model.py:_buttocks_force_and_partials()`.

Behavior:
- **Compression-only contact**: if separation occurs, force is 0.
- **Bilinear spring**:
  - stiffness `k1` up to bottom-out,
  - stiffness `k2` after bottom-out.
- **Damping**: viscous damping in contact, with total contact force clamped to be non-negative (contact cannot “pull”).

Bottom-out is specified as a **force threshold** `bottom_out_force_kN`. The implied bottom-out compression is:

- `x0 = bottom_out_force_N / k1`.

With the default parameters:
- `k1 = 180.5 kN/m`,
- `bottom_out_force = 7.0 kN`,

the implied bottom-out compression is:
- `x0 ≈ 38.8 mm`.

This was selected as a modeling choice consistent with an assumed **~35–40 mm** “available buttocks thickness” before sitting-bone / hard-path engagement.

### 2) Intervertebral discs: Kelvin–Voigt + rate-dependent compression stiffness
Configured in `config.json` under `spine.*`, implemented across:
- `spine_sim/model.py` (force law, Kemper scaling, strain rate),
- `spine_sim/model_components.py` (baseline stiffness distribution).

Each disc element has:
- spring stiffness (compression) scaled by strain rate,
- spring stiffness (tension) reduced by a constant multiplier,
- linear dashpot damping.

#### Strain rate definition (per disc)
Strain rate is computed from relative velocity and a uniform reference disc height `h0`:

- compression rate = `max(-relv, 0)`,
- `eps_dot = compression_rate / h0`.

`h0` is set by `spine.disc_height_mm`.

#### Rate-dependent stiffness: Kemper et al.
Kemper et al. provide a linear fit between strain rate and disc compressive stiffness:

- `k = 57.328 * eps_dot + 2019.1` (N/mm), with `eps_dot` in 1/s.

The code converts this to N/m and uses it as a **multiplier** on a baseline per-level stiffness distribution:

- `k_comp(t) = k0(level) * [kK(eps_dot) / kK(eps_norm)]`.

`eps_norm` defaults to 0 1/s as a normalization convention.

#### Strain-rate smoothing (numerical stabilization)
Strain rate is low-pass filtered with a first-order filter:

- time constant `spine.kemper.strain_rate_smoothing_tau_ms`.

This is an implementation / stability choice, not taken from literature.

---

## Parameter provenance (“magic numbers” audit)

This section documents **physics and signal-processing constants** from:
- `config.json`,
- hard-coded physics constants and literature-derived stiffness tables in code.

### A) External model data: masses and geometry
**`model.masses_json`** (`opensim/fullbody.json`)
- **Source:** OpenSim thoracolumbar spine & rib cage model by Bruno et al.
- **How generated:** `opensim/extract_opensim_masses.py` reads `.osim` and writes a JSON with:
  - per-body masses (kg),
  - body frame heights relative to pelvis (mm).
- **Primary reference:** Bruno et al., 2015. DOI: 10.1115/1.4030408.
- **Model distribution:** SimTK project “Thoracolumbar spine and rib cage model in OpenSim”: https://simtk.org/projects/spine_ribcage.

### B) Buttocks parameters
From `config.json` → `buttock.*`.

- `buttock.k1_n_per_m = 180500.0` N/m
  **Source:** Van Toen et al. report across-subject mean buttock stiffness 180.5 kN/m.
  DOI: 10.1097/BRS.0b013e31823ecae0.

- `buttock.c_ns_per_m = 3130.0` Ns/m
  **Source:** Van Toen et al. report across-subject mean buttock damping 3.13 kN·s/m.
  DOI: 10.1097/BRS.0b013e31823ecae0.

- `buttock.bottom_out_force_kN = 7.0` kN
  **Source:** not found in literature in this repo.
  **Modeling choice:** chosen so that with `k1 = 180.5 kN/m`, bottom-out occurs at ~35–40 mm compression (`~38.8 mm`).

- `buttock.k2_n_per_m = 5000000.0` N/m
  **Source:** not found in literature in this repo.
  **Modeling choice:** “very stiff” post-bottom-out stiffness (~5 kN/mm).

#### Batch-mode buttocks values (`simulate.py`)
Batch mode sweeps buttocks stiffness/damping pairs:

- (305 kN/m, 5.25 kN·s/m)
- (85.2 kN/m, 1.75 kN·s/m)
- (180.5 kN/m, 3.13 kN·s/m)

**Source:** Van Toen et al. Table 2 (subject-specific and across-subject values).
DOI: 10.1097/BRS.0b013e31823ecae0.

Batch mode bottom-out threshold includes:
- `7.0 kN` and `9999.0 kN` (“unlimited”, effectively disables bottom-out).
`9999.0 kN` is a modeling convenience (non-literature).

### C) Spine stiffness distribution (baseline per level)
The baseline per-level axial stiffness dictionary in `spine_sim/model_components.py` is taken from:

- Kitazaki & Griffin, 1997, Table 4 “Axial stiffness (N/m × 10^6)”.
DOI: 10.1006/jsvi.1996.0674.

These values are used as a **baseline distribution** and then modified by the Kemper strain-rate multiplier.

### D) Rate dependence and disc height (Kemper)
From:
- `spine.disc_height_mm = 11.3` mm
  **Source:** Kemper et al. report an overall average initial disc height of 11.32 mm (combined across studies) for strain-rate calculations.

- hard-coded in `spine_sim/model.py:kemper_k_n_per_m()`
  `57.328` and `2019.1` in `k = 57.328*eps_dot + 2019.1` (N/mm)
  **Source:** Kemper et al. linear fit.

- `spine.kemper.warn_over_eps_per_s = 73.0` 1/s
  **Source:** Kemper et al. highest-rate tests are ~72.7 1/s (mean).
  (Used only for warnings; the model does not clamp strain rate.)

### E) Disc damping
From `config.json` → `spine.damping_ns_per_m = 1200.0` Ns/m.

- **Source:** Raj & Krishnapillai (2019) state “all translational damping values 1200 Ns/m”.
  DOI: 10.1002/cnm.3307.

### F) Model-specific mass augmentations (non-literature)
From `config.json` → `model.*`, implemented in `spine_sim/mass.py`.

- `model.arm_recruitment = 0.5`
  **Source:** not found in literature in this repo.
  **Modeling choice:** assumes ~50% of arm mass effectively loads the spine during a drop.

- `model.helmet_mass_kg = 0.7` kg
  **Source:** not found in literature in this repo.
  **Modeling choice:** typical paragliding helmet mass.

### G) Tension behavior (non-literature)
From `config.json` → `spine.tension_k_mult = 0.1`.

- **Source:** not found in literature in this repo.
- **Modeling choice:** reduces tensile stiffness relative to compression while keeping the chain connected.
  (Empirically reported by the author to have <0.1% effect on T12–L1 peak force when changed up to 0.5, for tested conditions.)

### H) Gravity constant
Hard-coded in `spine_sim/model.py`:
- `G0 = 9.80665` m/s²
  **Source:** standard gravity constant (conventional value).

---

## Values not found in literature (explicit list)

These constants are *intentionally* treated as assumptions/heuristics in this repository:

### Modeling assumptions
- `model.arm_recruitment = 0.5` (arm inertial participation)
- `model.helmet_mass_kg = 0.7` (paragliding helmet)
- `buttock.bottom_out_force_kN = 7.0` (derived from assumed 35–40 mm compression limit + k1)
- `buttock.k2_n_per_m = 5e6` (post-bottom-out stiffness)
- `spine.tension_k_mult = 0.1` (tension stiffness multiplier)
- `spine.kemper.normalize_to_eps_per_s` (normalization convention for the multiplier)

### Signal processing & event detection heuristics
- `drop.style_duration_threshold_ms`
- `drop.sim_duration_ms`
- `drop.gravity_settle_ms`
- `drop.peak_threshold_g`
- `drop.freefall_threshold_g`
- “pad with -1.0 g” when extending signals beyond measured data

### Numerical stabilization inside constitutive update
- `spine.kemper.strain_rate_smoothing_tau_ms = 2.0` (chosen for stability/smoothness; not literature)

---

## References

1. Bruno, A. G., Bouxsein, M. L., Anderson, D. E. (2015). *Development and Validation of a Musculoskeletal Model of the Fully Articulated Thoracolumbar Spine and Rib Cage*. Journal of Biomechanical Engineering. DOI: 10.1115/1.4030408.
   OpenSim model distribution: https://simtk.org/projects/spine_ribcage.

2. Van Toen, C., Sran, M. M., Robinovitch, S. N., Cripton, P. A. (2012). *Transmission of force in the lumbosacral spine during backward falls*. Spine. DOI: 10.1097/BRS.0b013e31823ecae0.

3. Kemper, A. R., McNally, C., Manoogian, S., McNeely, D., Duma, S. *Stiffness Properties of Human Lumbar Intervertebral Discs in Compression and the Influence of Strain Rate*. (Referenced in repo as `kemper_2013_stiffness_properties.md`.)

4. Raj, N. R., Krishnapillai, S. (2019). *An improved spinal injury parameter model for underbody impulsive loading scenarios*. DOI: 10.1002/cnm.3307.

5. Kitazaki, S., Griffin, M. J. (1997). *A modal analysis of whole-body vertical vibration, using a finite element model of the human body*. Journal of Sound and Vibration. DOI: 10.1006/jsvi.1996.0674.

6. SAE J211/1. *Instrumentation for Impact Test — Part 1 — Electronic Instrumentation*. (CFC filtering convention; SAE-style mapping used in this repo.)

---

## License
MIT. See `LICENSE`.
