# SpineSim - Axial spine loading from paragliding harness drop tests

SpineSim is a research-oriented Python tool that simulates **1D (axial) vertical spine compression** under **measured base acceleration excitation** (e.g., harness/seat drop tests). It is intended for engineering exploration of how **impact onset** and **buttocks "bottom-out"** assumptions influence predicted internal spine forces, with a focus on the **thoracolumbar junction force (T12 - L1)**.

This repository is **not** a medically validated injury predictor.

---

## Model lineage (what this model is built from)

This model is intentionally built from a small set of "pillars" that each contribute one major part of the physics:

### 1) Rate-dependent disc compression stiffness - Kemper et al.

**Why it matters:** this is the core reason the model reacts differently to "fast-onset" vs "slow-onset" impacts.

- **What Kemper did (real-world testing):** axial compression tests on **human lumbar intervertebral discs** using a **servo-hydraulic material testing system**, across multiple loading rates, and then a fitted relationship between loading rate and compressive stiffness.
- **What we take from it:** the fitted stiffness relationship (Equation 1) used to make disc compression stiffness **increase when the disc is compressed faster**.

Kemper's fitted relationship (as written in the paper) is:

\[
k = 57.328 \dot{\varepsilon} + 2019.1
\]

(with \(k\) in N/mm).

In this README we will refer to \(\dot{\varepsilon}\) as **disc compression speed** (Kemper's variable is a "rate" term computed from how fast the disc shortens, scaled by disc height).

### 2) Baseline stiffness distribution by spinal level - Kitazaki & Griffin

**Why it matters:** Kemper provides "how stiffness changes with compression speed," but you still need a baseline stiffness pattern along the spine.

- **What Kitazaki & Griffin did:** a **2D finite-element** whole-body vibration model, validated by matching **measured vibration mode shapes below 10 Hz**.
- **What we take from it:** the **axial stiffness values** of the spinal "disc-like" elements by level (their Table 4 axial stiffnesses), used here as a **slow-loading baseline** distribution along the spine.

### 3) Seat/buttocks stiffness and damping - Van Toen et al.

**Why it matters:** in drop tests, the seat/buttocks interface is often the dominant compliance early in the event and strongly affects peak internal forces.

- **What Van Toen et al. did:** measured buttocks mechanics in a fall-related loading context.
- **What we take from it:** mean buttocks stiffness and damping values, reused here as a seat-contact element.

### 4) Segment masses - Bruno et al. / OpenSim

**Why it matters:** base excitation loads the system through inertia, so segment mass distribution matters.

- **What we take from Bruno/OpenSim:** **segment masses only** (and relative vertical positions for plotting).
- **What we do not take from OpenSim:** no OpenSim dynamics, no muscle optimization, no joint/muscle forces from the musculoskeletal model.
- **Important modeling consequence:** the OpenSim model provides a combined **`head_neck`** body; **cervical vertebrae are not modeled explicitly** here.

---

## How the model is built (walkthrough)

### 1) Topology: a 1D axial chain

The body is modeled as a serial chain of lumped masses connected by axial elements:

```
[Base / harness interface] -- buttocks(contact) -- pelvis -- L5 -- ... -- T1 -- HEAD
```

- **Nodes (masses):** pelvis, L5…T1, and HEAD (HEAD is the OpenSim `head_neck` lump, optionally with helmet mass).
- **Elements:** one buttocks contact element, then one axial "disc" element per adjacent node pair.
- **Target output:** internal axial force at **T12 - L1**.

### 2) Input: base acceleration excitation

The input is a measured acceleration time history (in g) interpreted as the **base excitation** at the harness/seat interface. Internally, this drives the chain through inertia of the lumped masses.

### 3) Masses (Bruno/OpenSim) + arm recruitment + helmet

- Masses are extracted from Bruno et al.'s OpenSim model.
- The OpenSim model includes **`head_neck`** as a combined lump; this becomes the `HEAD` node.
- A configurable fraction of total arm mass can be "recruited" into the chain and is added at **T1** (a shoulder-attachment-level simplification).
- Optional helmet mass is added to `HEAD`.

### 4) Disc elements: baseline distribution (Kitazaki) + compression-speed scaling (Kemper)

Each disc element behaves like a Kelvin - Voigt element (spring + damper) in the axial direction.

**Baseline stiffness by level (slow-loading baseline):**

- The model starts from Kitazaki & Griffin's **axial stiffness values by level** (their vibration-validated 2D FE work).
- Kitazaki includes cervical segments; in this model they are lumped into `HEAD`, so the top stiffness is represented by **HEAD - T1**.

**Compression stiffness increases with disc compression speed (Kemper):**

- During compression, disc stiffness is increased according to Kemper's fitted relationship (equation shown above).
- **Only compression stiffness is Kemper-scaled.** Tension stiffness is kept constant (a pragmatic choice to keep the chain connected without claiming rate-dependent tensile data).

### 5) Buttocks element: contact + bottom-out

The buttocks/seat interface is modeled as:

- **compression-only contact** (no "pulling" force),
- a spring + damper response while in contact,
- with a **bottom-out** behavior.

**Bottom-out concept (engineering interpretation):**

- Bottom-out represents the point where **buttock tissue reaches maximum compression** and the **sitting bone effectively contacts a hard path**.

**Current implementation (force-threshold form):**

- Bottom-out is currently represented as a **force threshold**.
- The default bottom-out force (7 kN) is chosen so that, with a typical/median buttocks stiffness, it corresponds to a typical/median maximum compressible thickness:
  - "median thickness for a median stiffness person" → about **38 mm** → about **7 kN**.
- In friendlier words: we pick a force level that corresponds to "the tissue has used up its available compression distance."

_Future direction:_ a thickness-based (mm-based) bottom-out parameterization is a natural improvement.

---

## Scope limits and parameter assumptions

### Scope limits (what this model does **not** represent)

- **1D axial only:** no bending, shear, spinal curvature effects, or off-axis coupling.
- **Not an injury predictor:** not medically validated; intended for comparative engineering exploration and sensitivity analysis.
- **No muscles / bracing behavior:** the model does not attempt to represent active muscle response during impact.
- **OpenSim not used for dynamics:** OpenSim is only used as a source of segment masses (and plotting geometry), not as a forward dynamics musculoskeletal simulation.

### Parameter assumptions (what may affect absolute results)

- **Kemper scope:** Kemper tested **lumbar** discs; this model applies the same compression-speed scaling across all modeled levels as an engineering approximation.
  - Rationale: many models effectively lump everything above T12; here we at least keep segments above T12 separate, but we still treat their disc rate-scaling as an approximation.
- **Uniform disc height:** a single disc height (11.3 mm from Kemper's compilation) is used everywhere to relate "disc compression speed" to Kemper's rate variable.
  - This is a global assumption. We expect limited sensitivity for T12 - L1 in typical drops, but this has **not** been formally tested here yet.
  - Further work: explicitly test sensitivity to disc height choices.
- **Buttocks bottom-out threshold is not literature-derived (in this repo):**
  - It is set from a thickness-based engineering assumption (typical available compression distance) and converted to a force threshold using stiffness.
  - Further work: validate bottom-out thickness/force across subjects and seat/harness configurations.
- **Uniform disc damping:** the same damping value is used for all discs.
  - Practical note: varying disc damping from 300 to 6000 Ns/m changed peak T12 - L1 by **≤ 5%** across **40 drops** (in the author's dataset).
- **Cervical lumping:** cervical vertebrae are not modeled explicitly (OpenSim `head_neck` is used as a single lumped mass).
- **Tension response:** tensile stiffness is constant and not Kemper-scaled (compression-only rate scaling).

---

## Outputs (what you get)

For each input file `drops/<name>.csv`, results are written under `output/<name>/`:

- `timeseries.csv` (time histories)
- `displacements.png`
- `forces.png`
- `mixed.png`

A batch summary is written to:

- `output/summary.json`

Key engineer-facing outputs:

- **Peak T12 - L1 force** (kN) and time-to-peak.
- **Peak buttocks force** (kN).
- **Buttocks bottomed-out or not** (based on inferred compression vs bottom-out point).
- **Spine shortening** metric (difference between head and pelvis motion in this 1D model).

---

## How to run

### Install

Install with uv.

### Run over all CSV files in `drops/`

```bash
./simulate.py
```

### Input CSV format (drops)

Input CSV files in `drops/` should contain:

- a time column: `time` (or `time0` or `t`) in seconds or milliseconds,
- an acceleration column: `accel` (or `acceleration`) in g.

---

## Implementation details (intentionally kept at the bottom)

- **Disc compression-speed computation:** internally computed from relative node closing velocity (compression only) and a uniform disc height; it is then used to evaluate Kemper's fitted stiffness relation.
- **Disc stiffness application:** Kemper scaling is applied to **compression stiffness only**; tension stiffness is constant.
- **Signal processing:** the tool can apply a CFC low-pass filter to the input acceleration and then extract a simulation window; details are intentionally treated as implementation-level and are kept in code.
- **Numerics:** nonlinear Newmark integration with Newton iterations is used; disc compression stiffness is updated stepwise with smoothing for stability.

---

## References (core pillars)

1. Bruno, A. G., Bouxsein, M. L., Anderson, D. E. (2015). _Development and Validation of a Musculoskeletal Model of the Fully Articulated Thoracolumbar Spine and Rib Cage_. Journal of Biomechanical Engineering. DOI: 10.1115/1.4030408.
   OpenSim model distribution: https://simtk.org/projects/spine_ribcage.

2. Van Toen, C., Sran, M. M., Robinovitch, S. N., Cripton, P. A. (2012). _Transmission of force in the lumbosacral spine during backward falls_. Spine. DOI: 10.1097/BRS.0b013e31823ecae0.

3. Kemper, A., McNally, C., Manoogian, S., McNeely, D., Duma, S. (Paper Number 07-0471). _Stiffness Properties of Human Lumbar Intervertebral Discs in Compression and the Influence of Strain Rate_.
   (Included in this repo as `kemper_2013_stiffness_properties.md`.)

4. Kitazaki, S., Griffin, M. J. (1997). _A modal analysis of whole-body vertical vibration, using a finite element model of the human body_. Journal of Sound and Vibration, 200(1), 83 - 103.
   (Included in this repo as `kitazaki_1996_vibration_modal.md`.)

---

---

## License

MIT. See `LICENSE`.
