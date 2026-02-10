# SpineSim - Axial spine loading from harness/seat drop tests (1D base excitation)

SpineSim is a research-oriented Python tool that simulates **1D (axial) vertical spine compression** under a **measured base acceleration time history** (e.g., paragliding harness/seat drop tests, seat-drop experiments, or other vertical impacts captured as an accelerometer record).

The primary engineer-facing quantity is the **internal axial force at T12 - L1** (thoracolumbar junction), along with buttocks contact force and compression diagnostics.

This repository is **not** a medically validated injury predictor.

---

## Quickstart

### Install

Install with `uv` (recommended). Exact commands depend on how you manage environments in this repo.

### Run (single mode + profile)

A single run requires both:

- `--mode`: `localized` or `uniform`,
- `--profile`: `sporty`, `avg`, or `soft`.

```bash
./simulate.py <subfolder> --mode localized --profile avg
```

or:

```bash
./simulate.py <subfolder> --mode uniform --profile sporty
```

`<subfolder>` refers to `input/<subfolder>/...` and outputs to `output/<subfolder>/...`.

### Run batch (all profiles × both modes)

```bash
./simulate.py <subfolder> --batch
```

Batch outputs are written under:

- `output/<subfolder>/<profile>-<loc|uni>/...`

---

## Repository layout (high level)

- `input/<subfolder>/*.csv`: acceleration time histories (base excitation).
- `output/<subfolder>/...`: per-run results, plots, and a `summary.csv`.
- `config.json`: model parameters (buttocks **mode/profile are runtime CLI selections**, not stored here).
- `spine_sim/`: model, solver, and command pipeline.

---

## Model lineage (what this model is built from)

This model is intentionally built from a small set of "pillars" that each contribute one major part of the physics. The goal is to keep assumptions explicit and separable for sensitivity testing.

### 1) Rate-dependent disc compression stiffness - Kemper et al.

**Why it matters:** this is the core reason the model reacts differently to "fast-onset" vs "slow-onset" impacts.

- **What Kemper did:** axial compression tests on **human lumbar intervertebral discs** across multiple loading rates, with a fitted relationship between loading rate and compressive stiffness.
- **What we take from it:** a stiffness multiplier behavior so disc compression stiffness **increases when the disc is compressed faster**.

Kemper's fitted relationship (as written in the paper) is:

\[
k = 57.328 \dot{\varepsilon} + 2019.1
\]

(with k in N/mm).

In this README, \(\dot{\varepsilon}\) is **disc compression strain-rate** computed from how fast the element shortens in compression divided by an effective height.

### 2) Baseline stiffness distribution by spinal level - Kitazaki & Griffin

**Why it matters:** Kemper provides "how stiffness changes with compression speed," but you still need a baseline stiffness pattern along the spine.

- **What Kitazaki & Griffin did:** a **2D finite-element** whole-body vibration model, validated by matching measured vibration mode shapes below 10 Hz.
- **What we take from it:** the **axial stiffness values** of spinal "disc-like" elements by level (Table 4), used here as a **slow-loading baseline** distribution along the spine.

**Important cervical detail (because this repo simplifies the neck):**

- Kitazaki Table 4 includes cervical axial stiffnesses for **Head - C1 through C7 - T1**.
- OpenSim provides a combined `head_neck` body, and this model does **not** include explicit C1…C7 nodes.
- Therefore, the model computes the baseline `HEAD - T1` stiffness by combining the 8 Kitazaki cervical joints **in series**.

\[
k*{\text{HEAD-T1,eff}} = \left(\sum*{i=1}^{8}\frac{1}{k_i}\right)^{-1}
\]

This yields an effective baseline `HEAD - T1` stiffness of about **84 kN/m**.

### 3) Seat/buttocks stiffness and damping - Van Toen et al. (effective dynamic fit)

**Why it matters:** in drop tests, the seat/buttocks interface is often the dominant compliance early in the event and strongly affects peak internal forces.

- **What Van Toen et al. did:** developed a verified lumped-parameter model for backward falls onto the buttocks and estimated an effective buttocks stiffness and damping from human subject force - time records.
- **What we take from it:** representative effective buttocks stiffness and damping values as a baseline "effective dynamic" buttocks compliance.

### 4) Thickness-based seated buttocks anatomy - Sonenblum et al.

**Why it matters:** the available soft-tissue thickness under the ischial tuberosity apex in _seated posture_ bounds how far tissue can compress before a hard load path dominates.

- **What Sonenblum et al. did:** upright MRI scans of seated buttocks and quantitative measurements of tissue thickness and contour at and around the ischial tuberosity apex.
- **What we take from it:** measured **seated apex tissue thickness** values (able-bodied cohort) used here as geometry-backed limits for thickness-based hardening ("bottom-out-like") modeling.

### 5) Segment masses - Bruno et al. / OpenSim

**Why it matters:** base excitation loads the chain through inertia, so segment mass distribution matters.

- **What we take from Bruno/OpenSim:** **segment masses** (and relative vertical positions for plotting).
- **What we do not take from OpenSim:** no OpenSim dynamics, no muscle optimization, no joint/muscle forces from the musculoskeletal model.

**Important modeling consequence (neck simplification):**

- OpenSim provides a combined `head_neck` body. This becomes the `HEAD` lump in the chain.
- Cervical vertebrae are not modeled explicitly; their baseline stiffness is represented via the **series-equivalent** `HEAD - T1` stiffness (Kitazaki cervical chain).
- For strain-rate computations, `HEAD - T1` uses a **stacked cervical height** so that collapsing multiple joints into one does not artificially inflate \(\dot{\varepsilon}\).
- `HEAD - T1` damping is also reduced via a series-equivalent approximation (dashpots in series), consistent with the same lumping assumption.

---

## How the model is built (walkthrough)

### 1) Topology: a 1D axial chain

The body is modeled as a serial chain of lumped masses connected by axial elements:

```
[Base / harness interface] -- buttocks(contact) -- pelvis -- L5 -- ... -- T1 -- HEAD
```

- **Nodes (masses):** pelvis, L5…T1, and `HEAD` (OpenSim `head_neck` lump, optionally + helmet mass).
- **Elements:** one buttocks contact element, then one axial "disc-like" element per adjacent node pair.
- **Target output:** internal axial force at **T12 - L1**.

### 2) Input: base acceleration excitation

The input is a measured acceleration time history (in g) interpreted as the **base excitation** at the harness/seat interface.

### 3) Masses + arm recruitment + helmet

- Masses are extracted from Bruno et al.'s OpenSim model.
- A configurable fraction of total arm mass can be "recruited" into the chain and is added at T1 (a shoulder-attachment-level simplification).
- Optional helmet mass is added to `HEAD`.

### 4) Disc-like elements: baseline distribution (Kitazaki) + rate scaling (Kemper)

Each disc-like element behaves like a Kelvin - Voigt element (spring + damper) in the axial direction.

**Baseline stiffness by level (slow-loading baseline):**

- Baseline axial stiffnesses by spinal level come from Kitazaki & Griffin (Table 4).
- Cervical levels are lumped into `HEAD - T1` via a series-equivalent stiffness as described above.

**Compression stiffness increases with strain-rate (Kemper):**

- During compression, stiffness is increased according to Kemper's fitted relationship.
- Only **compression** stiffness is Kemper-scaled.
- **Tension** stiffness is kept constant (pragmatic choice to keep the chain connected without claiming rate-dependent tensile data).

#### Strain-rate computation and disc heights (important)

Strain-rate is computed from compression-only closing speed:

- Compression speed is derived from relative velocity across each element.
- The compression speed is divided by an element-specific effective height to obtain \(\dot{\varepsilon}\).

Config keys:

- `spine.disc_height_mm`: height of one thoracic/lumbar IVD (single-disc height used for all non-neck elements).
- `spine.cervical_disc_height_single_mm`: height of one cervical IVD (single-disc height).

Neck lumping details:

- The element `HEAD - T1` represents 8 Kitazaki cervical joints in series (Head - C1 … C7 - T1).
- For strain-rate only, the effective height is treated as a stacked height:

\[
h\_{\text{eff}}(\text{HEAD-T1}) = 8 \times \text{cervical disc height single}
\]

This prevents the lumped neck element from seeing an artificially high \(\dot{\varepsilon}\) simply because multiple joints were collapsed into one.

### 5) Buttocks element: runtime modes + thickness-based hardening (localized mode)

The buttocks/seat interface is modeled as:

- compression-only contact (no pulling),
- spring + damper response while in contact,
- optional thickness-based hardening (localized mode only).

#### Runtime modes (selected on the CLI)

Two modes are selected at runtime:

- `uniform`: linear spring + compression-only damping, **no thickness hardening**.
- `localized`: thickness-limited "barrier" hardening based on seated apex thickness.

Mode is not stored in `config.json` and must be supplied at runtime:

- `--mode localized` or `--mode uniform`.

#### Buttocks profiles (selected on the CLI)

A buttocks profile is also selected at runtime:

- `--profile sporty`
- `--profile avg`
- `--profile soft`

The profile selects:

- seated apex thickness (Sonenblum),
- effective dynamic stiffness and damping (Van Toen-like scale, per-profile values).

Profiles live in `config.json` under `buttock.profiles.*`.

#### Damping choice: compression-only (avoids "dashpot pulling" on rebound)

For the buttocks element, damping is applied only during contact closing:

- Let \(v\_{\text{rel}}\) be the pelvis velocity relative to base across the buttocks element.
- Closing speed is \(\dot{x} = \max(-v\_{\text{rel}}, 0)\).
- Damping force is \(F_d = c \dot{x}\).

This avoids unphysical tensile damping forces during rebound.

#### Mode `uniform`: linear response

- Spring: \(F_s = k_1 x\).
- Damping: \(F_d = c \dot{x}\).
- Total: \(F = F_s + F_d\).

#### Mode `localized`: seated apex thickness + barrier hardening (no force threshold)

Sonenblum provides a seated, loaded apex thickness under the IT apex, interpreted here as remaining thickness at idle sitting:

- \(h\_{\text{idle}}\) = seated apex thickness (per profile).

The simulator computes an initial seated equilibrium under gravity. Buttocks compression at that equilibrium is used as a reference:

- \(x\_{\text{idle}}\) = buttocks compression at initial equilibrium.

Define:

- \(x*{\text{extra}} = \max(x - x*{\text{idle}}, 0)\),
- remaining thickness \(h = h*{\text{idle}} - x*{\text{extra}}\).

Barrier hardening term:

- \(F*{\text{hard}} = k*{\text{bar}} \frac{x\_{\text{extra}}^2}{\max(h, \epsilon)}\),

where \(\epsilon\) is a small numerical safety constant (not a biomechanical "minimum thickness").

Barrier gain:

- \(k\_{\text{bar}} = k_1 \cdot k2_mult\).

Total spring force:

- \(F*s = k_1 x + F*{\text{hard}}\).

#### Seated initial condition and "idle" reference (important)

- Each simulation starts from a **static equilibrium under gravity** (a seated initial condition).
- In `localized` mode, hardening is defined relative to the seated equilibrium via `x_idle`.
- During the internal static solve (and any optional settling), the model disables barrier hardening so equilibrium can be found without needing `x_idle` first.

---

## Scope limits and parameter assumptions

### Scope limits (what this model does not represent)

- 1D axial only: no bending, shear, spinal curvature effects, or off-axis coupling.
- Not an injury predictor: not medically validated; intended for comparative engineering exploration and sensitivity analysis.
- No muscles / bracing behavior: does not represent active muscle response during impact.
- OpenSim not used for dynamics: OpenSim is only a source of segment masses (and plotting geometry).

### Parameter assumptions (what may affect absolute results)

- Kemper scope: Kemper tested lumbar discs; this model applies the same compression strain-rate scaling across all modeled levels as an engineering approximation.
- Disc heights:
  - Thoracic/lumbar elements use a single configured disc height (`spine.disc_height_mm`).
  - The lumped cervical element uses a stacked height computed from `spine.cervical_disc_height_single_mm × 8`.
- Buttocks thickness anchor is posture/surface dependent:
  - Sonenblum thicknesses are seated on HR45 foam in a standardized posture; different harness geometry may change the true apex thickness at idle.

---

## Input CSV format

Input CSV files are read from:

- `input/<subfolder>/*.csv`

Each file must contain:

- a time column: `time` (or `time0` or `t`) in seconds or milliseconds,
- an acceleration column: `accel` (or `acceleration`) in g.

---

## Outputs (what you get)

Per input file `input/<subfolder>/<name>.csv`, results are written under an output run folder, typically:

- `output/<subfolder>/<profile>-<loc|uni>/<name>/...` (batch), or
- `output/<subfolder>/<name>/...` (depending on how you invoke the run configuration and output subfoldering).

Generated artifacts include:

- `timeseries.csv` (time histories)
- `displacements.png`
- `forces.png`
- `mixed.png`

A summary CSV is written to:

- `output/<subfolder>/summary.csv` (single runs), or
- `output/<subfolder>/<profile>-<loc|uni>/<profile>-<loc|uni>.csv` (batch, one summary per batch variant)

Key engineer-facing summary outputs include:

- peak T12 - L1 force (kN) and time-to-peak,
- peak buttocks force (kN),
- buttocks compression diagnostics:
  - max compression,
  - max extra compression beyond idle,
  - minimum remaining apex thickness (localized mode),
- spine shortening metric (difference between head and pelvis motion in this 1D model).

### Output caching / repeatability note

The simulator writes a copy of the effective configuration into the output folder and may skip recomputation if:

- the saved config matches the current config (including runtime `mode/profile`), and
- the summary CSV already exists.

This is intended to make batch workflows restartable.

---

## References and included paper notes

### Core pillars (used directly in the model)

1. Bruno, A. G., Bouxsein, M. L., Anderson, D. E. (2015). _Development and Validation of a Musculoskeletal Model of the Fully Articulated Thoracolumbar Spine and Rib Cage_. Journal of Biomechanical Engineering. DOI: 10.1115/1.4030408.
   OpenSim model distribution: https://simtk.org/projects/spine_ribcage

2. Van Toen, C., Sran, M. M., Robinovitch, S. N., Cripton, P. A. (2012). _Transmission of force in the lumbosacral spine during backward falls_. Spine. DOI: 10.1097/BRS.0b013e31823ecae0.
   (Repo note copy: `toen_2012_backward.md`.)

3. Kemper, A., McNally, C., Manoogian, S., McNeely, D., Duma, S. (Paper Number 07-0471). _Stiffness Properties of Human Lumbar Intervertebral Discs in Compression and the Influence of Strain Rate_.

4. Kitazaki, S., Griffin, M. J. (1997). _A modal analysis of whole-body vertical vibration, using a finite element model of the human body_. Journal of Sound and Vibration, 200(1), 83 - 103.

5. Sonenblum, S. E., Seol, D., Sprigle, S. H., Cathcart, J. M. (2020). _Seated buttocks anatomy and its impact on biomechanical risk_. Journal of Tissue Viability.
   (Repo note copy: `sonenblum_2020_anatomy.md`.)

### Additional included reading (not required by the simulator, but relevant)

These are checked in as reading notes / copies to support future model upgrades (e.g. more detailed buttocks tissue mechanics, viscoelasticity, FE context):

- Then et al. (2007). _A method for a mechanical characterisation of human gluteal tissue_.
  (Repo note copy: `then_2007_method.md`.)

- Then et al. (2012). _Method for characterizing viscoelasticity of human gluteal tissue_.
  (Repo note copy: `then_2012_viscoelasticity.md`.)

- Tang et al. (2010). _Finite Element Analysis of Contact Pressures between Seat Cushion and Human Buttock-Thigh Tissue_. Engineering, 2, 720 - 726. DOI: 10.4236/eng.2010.29093.
  (Repo note copy: `tang_2010_fe.md`.)

- Wang et al. (2021). _Subcutaneous Fat Thickness Remarkably Influences Contact Pressure and Load Distribution of Buttock in Seated Posture_. Journal of Healthcare Engineering.
  (Repo note copy: `wang_2021_fat.md`.)

---

## License

MIT. See `LICENSE`.
