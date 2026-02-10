# SpineSim - Axial spine loading from paragliding harness drop tests

SpineSim is a research-oriented Python tool that simulates **1D (axial) vertical spine compression** under **measured base acceleration excitation** (e.g., harness/seat drop tests). It is intended for engineering exploration of how **impact onset** and **buttocks "bottom-out"** assumptions influence predicted internal spine forces, with a focus on the **thoracolumbar junction force (T12 - L1)**.

This repository is **not** a medically validated injury predictor.

---

## Model lineage (what this model is built from)

This model is intentionally built from a small set of "pillars" that each contribute one major part of the physics.

### 1) Rate-dependent disc compression stiffness - Kemper et al.

**Why it matters:** this is the core reason the model reacts differently to "fast-onset" vs "slow-onset" impacts.

- **What Kemper did (real-world testing):** axial compression tests on **human lumbar intervertebral discs** across multiple loading rates, with a fitted relationship between loading rate and compressive stiffness.
- **What we take from it:** the fitted stiffness relationship used to make disc compression stiffness **increase when the disc is compressed faster**.

Kemper's fitted relationship (as written in the paper) is:

$$k = 57.328 \dot{\varepsilon} + 2019.1$$

(with $k$ in N/mm).

In this README we will refer to $\dot{\varepsilon}$ as **disc compression strain-rate** (computed from how fast the disc shortens, scaled by an effective disc height).

### 2) Baseline stiffness distribution by spinal level - Kitazaki & Griffin

**Why it matters:** Kemper provides "how stiffness changes with compression speed," but you still need a baseline stiffness pattern along the spine.

- **What Kitazaki & Griffin did:** a **2D finite-element** whole-body vibration model, validated by matching **measured vibration mode shapes below 10 Hz**.
- **What we take from it:** the **axial stiffness values** of the spinal "disc-like" elements by level (their Table 4 axial stiffnesses), used here as a **slow-loading baseline** distribution along the spine.

**Important cervical detail (because this repo simplifies the neck):**

- Kitazaki Table 4 includes cervical axial stiffnesses for **Head - C1 through C7 - T1**.
- OpenSim provides a combined `head_neck` body, and this model does **not** include explicit C1...C7 nodes.
- Therefore the model computes the baseline `HEAD - T1` stiffness by combining the 8 Kitazaki cervical joints **in series**:

$$k_{\text{HEAD - T1,eff}} = \left(\sum_{i=1}^{8}\frac{1}{k_i}\right)^{-1}$$

This yields an effective baseline `HEAD - T1` stiffness of about **84 kN/m**.

### 3) Seat/buttocks stiffness and damping - Van Toen et al. (effective dynamic fit)

**Why it matters:** in drop tests, the seat/buttocks interface is often the dominant compliance early in the event and strongly affects peak internal forces.

- **What Van Toen et al. did:** developed a verified lumped-parameter model for backward falls onto the buttocks and estimated an effective buttocks stiffness and damping from human subject force-time records.
- **What we take from it:** representative effective buttocks stiffness and damping values, used here as the baseline for buttocks compliance.

### 4) Thickness-based seated buttocks anatomy - Sonenblum et al.

**Why it matters:** how much tissue is present under the ischial tuberosities in _seated posture_ bounds how far soft tissue can compress before a hard load path dominates.

- **What Sonenblum et al. did:** upright MRI scans of seated buttocks and quantitative measurements of tissue thickness and contour at and around the ischial tuberosity apex.
- **What we take from it:** measured **seated apex tissue thickness** values (able-bodied cohort) as geometry-backed limits for thickness-based bottom-out modeling.

### 5) Segment masses - Bruno et al. / OpenSim

**Why it matters:** base excitation loads the system through inertia, so segment mass distribution matters.

- **What we take from Bruno/OpenSim:** segment masses only (and relative vertical positions for plotting).
- **What we do not take from OpenSim:** no OpenSim dynamics, no muscle optimization, no joint/muscle forces from the musculoskeletal model.

---

## How the model is built (walkthrough)

### 1) Topology: a 1D axial chain

The body is modeled as a serial chain of lumped masses connected by axial elements:

```
[Base / harness interface] -- buttocks(contact) -- pelvis -- L5 -- ... -- T1 -- HEAD
```

- **Nodes (masses):** pelvis, L5…T1, and HEAD (HEAD is the OpenSim `head_neck` lump, optionally with helmet mass).
- **Elements:** one buttocks contact element, then one axial "disc-like" element per adjacent node pair.
- **Target output:** internal axial force at **T12 - L1**.

### 2) Input: base acceleration excitation

The input is a measured acceleration time history (in g) interpreted as the **base excitation** at the harness/seat interface. Internally, this drives the chain through inertia of the lumped masses.

### 3) Masses (Bruno/OpenSim) + arm recruitment + helmet

- Masses are extracted from Bruno et al.'s OpenSim model.
- The OpenSim model includes `head_neck` as a combined lump; this becomes the `HEAD` node.
- A configurable fraction of total arm mass can be "recruited" into the chain and is added at **T1** (a shoulder-attachment-level simplification).
- Optional helmet mass is added to `HEAD`.

### 4) Disc-like elements: baseline distribution (Kitazaki) + rate scaling (Kemper)

Each disc-like element behaves like a Kelvin - Voigt element (spring + damper) in the axial direction.

**Baseline stiffness by level (slow-loading baseline):**

- The model starts from Kitazaki & Griffin's axial stiffness values by level (their vibration-validated 2D FE work).
- Kitazaki includes cervical segments; in this model they are lumped into `HEAD`, so the top stiffness is represented by `HEAD - T1` as a series-equivalent of Kitazaki's cervical joints.

**Compression stiffness increases with strain-rate (Kemper):**

- During compression, stiffness is increased according to Kemper's fitted relationship.
- Only compression stiffness is Kemper-scaled. Tension stiffness is kept constant (a pragmatic choice to keep the chain connected without claiming rate-dependent tensile data).

#### Strain-rate computation and disc heights (important)

Strain-rate is computed from compression-only closing speed:

- Compression speed is derived from relative velocity across each element.
- The compression speed is divided by an element-specific effective height to obtain $\dot{\varepsilon}$.

Config keys:

- `spine.disc_height_mm` is the height of one thoracic/lumbar IVD (single-disc height used for all non-neck elements).
- `spine.cervical_disc_height_single_mm` is the height of one cervical IVD (single-disc height).

Neck lumping details:

- The element `HEAD - T1` represents 8 Kitazaki cervical joints in series (Head - C1 … C7 - T1).
- For strain-rate only, the effective height is treated as a stacked height:

$$h_{\text{eff}}(\text{HEAD - T1}) = 8 \times \text{cervical disc height single mm}$$

This prevents the lumped neck element from seeing an artificially high $\dot{\varepsilon}$ simply because multiple joints were collapsed into one.

### 5) Buttocks element: two runtime modes + thickness-based bottom-out (localized mode)

The buttocks/seat interface is modeled as:

- compression-only contact (no pulling),
- a spring + damper response while in contact,
- optional thickness-based bottom-out behavior (localized mode only).

#### Runtime modes (selected on the CLI)

This repo supports two modes selected at runtime:

- `uniform`: no bottom-out, linear spring + damper.
- `localized`: thickness-limited barrier hardening based on seated apex thickness.

Mode is not stored in `config.json` and must be supplied at runtime:

- `--mode localized` or `--mode uniform`.

#### Buttocks profiles (selected on the CLI)

The buttocks element also uses a profile selected at runtime:

- `--profile sporty`
- `--profile avg`
- `--profile soft`

The profile selects:

- seated apex thickness (Sonenblum, able-bodied),
- effective dynamic stiffness and damping (Van Toen-like scale, per-profile values).

Profiles live in `config.json` under `buttock.profiles.*`.

#### Damping choice: compression-only (avoids "dashpot pulling" on rebound)

For the buttocks element, damping is applied only during contact closing:

- Let $v_{\text{rel}}$ be the pelvis velocity relative to base across the buttocks element.
- Closing speed is $\dot{x} = \max(-v_{\text{rel}}, 0)$.
- Damping force is $F_d = c \dot{x}$.

This avoids unphysical tensile damping forces during rebound and reduces reliance on force clamping for separation.

#### Mode `uniform`: linear response, no bottom-out

- Spring: $F_s = k_1 x$.
- Damping: $F_d = c \dot{x}$.
- Total: $F = F_s + F_d$.

No thickness hardening is applied.

#### Mode `localized`: seated apex thickness + barrier hardening (no force threshold)

Sonenblum provides a seated, loaded apex thickness under the IT apex. We interpret that as the remaining thickness at idle sitting:

- $h_{\text{idle}}$ = seated apex thickness (per profile).

The simulator also computes its own initial seated equilibrium under gravity. The buttocks compression at that equilibrium is used as a reference:

- $x_{\text{idle}}$ = buttocks compression at initial equilibrium.

Define:

- $x_{\text{extra}} = \max(x - x_{\text{idle}}, 0)$,
- remaining thickness $h = h_{\text{idle}} - x_{\text{extra}}$.

Barrier hardening term:

- $F_{\text{hard}} = k_{\text{bar}} \frac{x_{\text{extra}}^2}{\max(h, \epsilon)}$.

Here $\epsilon$ is a small numerical safety constant (not a biomechanical "minimum thickness").

Barrier gain:

- $k_{\text{bar}} = k_1 \cdot k2\_mult$.

Total spring force:

- $F_s = k_1 x + F_{\text{hard}}$.

---

## Scope limits and parameter assumptions

### Scope limits (what this model does not represent)

- 1D axial only: no bending, shear, spinal curvature effects, or off-axis coupling.
- Not an injury predictor: not medically validated; intended for comparative engineering exploration and sensitivity analysis.
- No muscles / bracing behavior: the model does not attempt to represent active muscle response during impact.
- OpenSim not used for dynamics: OpenSim is only used as a source of segment masses (and plotting geometry).

### Parameter assumptions (what may affect absolute results)

- Kemper scope: Kemper tested lumbar discs; this model applies the same compression strain-rate scaling across all modeled levels as an engineering approximation.
- Disc heights:
  - Thoracic/lumbar elements use a single configured disc height (`spine.disc_height_mm`) as a simplifying assumption.
  - The lumped cervical element uses a stacked height computed from a single cervical disc height (`spine.cervical_disc_height_single_mm`) times 8.
- Buttocks thickness anchor is posture/surface dependent:
  - Sonenblum thicknesses are seated on HR45 foam in a standardized posture; different harness geometry may change the true apex thickness at idle.

---

## Outputs (what you get)

For each input file `input/<name>.csv`, results are written under `output/<name>/`:

- `timeseries.csv` (time histories)
- `displacements.png`
- `forces.png`
- `mixed.png`

A summary is written to:

- `output/<subfolder>/summary.csv` (or a configured name in batch mode).

Key engineer-facing outputs:

- Peak T12 - L1 force (kN) and time-to-peak.
- Peak buttocks force (kN).
- Buttocks compression diagnostics:
  - max compression,
  - max extra compression beyond idle,
  - minimum remaining apex thickness (localized mode).
- Spine shortening metric (difference between head and pelvis motion in this 1D model).

---

## How to run

### Install

Install with uv.

### Run a single mode/profile

Mode and profile are required for a single run:

```bash
./simulate.py <subfolder> --mode localized --profile avg
```

Or:

```bash
./simulate.py <subfolder> --mode uniform --profile sporty
```

### Run batch (always both modes and all 3 profiles)

```bash
./simulate.py <subfolder> --batch
```

Batch outputs are written under `output/<subfolder>/<profile>-<loc|uni>/`.

### Input CSV format (input)

Input CSV files in `input/` should contain:

- a time column: `time` (or `time0` or `t`) in seconds or milliseconds,
- an acceleration column: `accel` (or `acceleration`) in g.

---

## Implementation details (intentionally kept at the bottom)

- Disc strain-rate computation: computed from relative node closing velocity (compression only) divided by an element-specific effective height; it is then used to evaluate Kemper's fitted stiffness relation.
- Disc stiffness application: Kemper scaling is applied to compression stiffness only; tension stiffness is constant.
- Signal processing: the tool can apply a CFC low-pass filter to the input acceleration and then extract a simulation window.
- Numerics: nonlinear Newmark integration with Newton iterations is used; disc compression stiffness is updated stepwise with smoothing for stability.

---

## References (core pillars)

1. Bruno, A. G., Bouxsein, M. L., Anderson, D. E. (2015). _Development and Validation of a Musculoskeletal Model of the Fully Articulated Thoracolumbar Spine and Rib Cage_. Journal of Biomechanical Engineering. DOI: 10.1115/1.4030408.  
   OpenSim model distribution: https://simtk.org/projects/spine_ribcage.

2. Van Toen, C., Sran, M. M., Robinovitch, S. N., Cripton, P. A. (2012). _Transmission of force in the lumbosacral spine during backward falls_. Spine. DOI: 10.1097/BRS.0b013e31823ecae0.

3. Kemper, A., McNally, C., Manoogian, S., McNeely, D., Duma, S. (Paper Number 07-0471). _Stiffness Properties of Human Lumbar Intervertebral Discs in Compression and the Influence of Strain Rate_.

4. Kitazaki, S., Griffin, M. J. (1997). _A modal analysis of whole-body vertical vibration, using a finite element model of the human body_. Journal of Sound and Vibration, 200(1), 83 - 103.

5. Sonenblum, S. E., Seol, D., Sprigle, S. H., Cathcart, J. M. (2020). _Seated buttocks anatomy and its impact on biomechanical risk_. Journal of Tissue Viability.

---

## License

MIT. See `LICENSE`.
