# SpineSim - Axial spine loading from paragliding harness drop tests

SpineSim is a research-oriented Python tool that simulates **1D (axial) vertical spine compression** under **measured base acceleration excitation** (e.g., harness/seat drop tests). It is intended for engineering exploration of how **impact onset** and **buttocks "bottom-out"** assumptions influence predicted internal spine forces, with a focus on the **thoracolumbar junction force (T12 - L1)**.

This repository is **not** a medically validated injury predictor.

---

## Model lineage (what this model is built from)

This model is intentionally built from a small set of "pillars" that each contribute one major part of the physics:

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
- Therefore the model computes the baseline `HEAD - T1` stiffness by combining the 8 Kitazaki cervical joints **in series** (same series-connection idea used when combining multiple discs in Kemper-style analysis):

$$k_{\text{HEAD - T1,eff}} = \left(\sum_{i=1}^{8}\frac{1}{k_i}\right)^{-1}$$

This yields an effective baseline `HEAD - T1` stiffness of about **84 kN/m** (much lower than the single C7 - T1 joint stiffness alone).

### 3) Seat/buttocks stiffness and damping - Van Toen et al.

**Why it matters:** in drop tests, the seat/buttocks interface is often the dominant compliance early in the event and strongly affects peak internal forces.

- **What Van Toen et al. did:** measured buttocks mechanics in a fall-related loading context.
- **What we take from it:** representative buttocks stiffness and damping values, reused here as a seat-contact element.

### 4) Segment masses - Bruno et al. / OpenSim

**Why it matters:** base excitation loads the system through inertia, so segment mass distribution matters.

- **What we take from Bruno/OpenSim:** **segment masses only** (and relative vertical positions for plotting).
- **What we do not take from OpenSim:** no OpenSim dynamics, no muscle optimization, no joint/muscle forces from the musculoskeletal model.
- **Important modeling consequence:** the OpenSim model provides a combined **`head_neck`** body; **cervical vertebrae are not modeled explicitly** here.
  - Their individual stiffnesses (Kitazaki Table 4: Head - C1 through C7 - T1) are combined **in series** to derive the baseline `HEAD - T1` stiffness.
  - The lumped `HEAD - T1` element uses a **stacked cervical height** for strain-rate computation so that Kemper scaling is not artificially inflated by the simplification.
  - The lumped `HEAD - T1` damping is reduced using the same "elements in series" approximation (dashpots in series).

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
- The OpenSim model includes **`head_neck`** as a combined lump; this becomes the `HEAD` node.
- A configurable fraction of total arm mass can be "recruited" into the chain and is added at **T1** (a shoulder-attachment-level simplification).
- Optional helmet mass is added to `HEAD`.

### 4) Disc-like elements: baseline distribution (Kitazaki) + rate scaling (Kemper)

Each disc-like element behaves like a Kelvin - Voigt element (spring + damper) in the axial direction.

**Baseline stiffness by level (slow-loading baseline):**

- The model starts from Kitazaki & Griffin's **axial stiffness values by level** (their vibration-validated 2D FE work).
- Kitazaki includes cervical segments; in this model they are lumped into `HEAD`, so the top stiffness is represented by **HEAD - T1** as a **series-equivalent** of Kitazaki's cervical joints.

**Compression stiffness increases with strain-rate (Kemper):**

- During compression, stiffness is increased according to Kemper's fitted relationship.
- **Only compression stiffness is Kemper-scaled.** Tension stiffness is kept constant (a pragmatic choice to keep the chain connected without claiming rate-dependent tensile data).

#### Strain-rate computation and disc heights (important)

Strain-rate is computed from compression-only closing speed:

- Compression speed is derived from relative velocity across each element.
- The compression speed is divided by an element-specific effective height to obtain $\dot{\varepsilon}$.

Config keys (clarified):

- `spine.disc_height_mm` is the height of **one thoracic/lumbar IVD** (single-disc height used for all non-neck elements).
- `spine.cervical_disc_height_single_mm` is the height of **one cervical IVD** (single-disc height).
  - This repo uses Wang et al. (2024) Table 2 baseline value: **5.6 mm**.
  - Wang also reports example subject-specific disc-height values of **4.7 mm** and **6.3 mm**, which are useful for sensitivity sweeps.

Neck lumping details:

- The element `HEAD - T1` represents 8 Kitazaki cervical joints in series (Head - C1 … C7 - T1).
- For strain-rate only, the effective height is treated as a stacked height:

$$h_{\text{eff}}(\text{HEAD - T1}) = 8 \cdot \texttt{spine.cervical\_disc\_height\_single\_mm}$$

This prevents the lumped neck element from seeing an artificially high $\dot{\varepsilon}$ simply because multiple joints were collapsed into one.

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

- **Kemper scope:** Kemper tested **lumbar** discs; this model applies the same compression strain-rate scaling across all modeled levels as an engineering approximation.
- **Disc heights:**
  - Thoracic/lumbar elements use a single configured disc height (`spine.disc_height_mm`) as a simplifying assumption.
  - The lumped cervical element uses a stacked height computed from a single cervical disc height (`spine.cervical_disc_height_single_mm`) times 8.
- **Buttocks bottom-out threshold is not literature-derived (in this repo):**
  - It is set from a thickness-based engineering assumption (typical available compression distance) and converted to a force threshold using stiffness.
  - Further work: validate bottom-out thickness/force across subjects and seat/harness configurations.
- **Uniform disc damping:** the same damping value is used for most disc-like elements.
  - Practical note: varying disc damping from 300 to 6000 Ns/m changed peak T12 - L1 by ≤ 5% across 40 drops (in the author's dataset).
- **Cervical lumping:** cervical vertebrae are not modeled explicitly (OpenSim `head_neck` is used as a single lumped mass).
  - `HEAD - T1` damping is reduced using a series-equivalent approximation to match the same lumping assumption.
- **Tension response:** tensile stiffness is constant and not Kemper-scaled (compression-only rate scaling).

---

## Outputs (what you get)

For each input file `input/<name>.csv`, results are written under `output/<name>/`:

- `timeseries.csv` (time histories)
- `displacements.png`
- `forces.png`
- `mixed.png`

A batch summary is written to:

- `output/summary.csv` (or a configured name)

Key engineer-facing outputs:

- **Peak T12 - L1 force** (kN) and time-to-peak.
- **Peak buttocks force** (kN).
- **Buttocks bottomed-out or not** (based on inferred compression vs bottom-out point).
- **Spine shortening** metric (difference between head and pelvis motion in this 1D model).

---

## How to run

### Install

Install with uv.

### Run over all CSV files in `input/`

```bash
./simulate.py
```

### Input CSV format (input)

Input CSV files in `input/` should contain:

- a time column: `time` (or `time0` or `t`) in seconds or milliseconds,
- an acceleration column: `accel` (or `acceleration`) in g.

---

## Implementation details (intentionally kept at the bottom)

- **Disc strain-rate computation:** computed from relative node closing velocity (compression only) divided by an element-specific effective height; it is then used to evaluate Kemper's fitted stiffness relation.
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

5. Wang, M. C., Kiapour, A., Massaad, E., Shin, J. H., Yoganandan, N. (2024). _A guide to finite element analysis models of the spine for clinicians_. J Neurosurg Spine, 40, 38 - 44.
   (Included in this repo as `wang_2024_fea_guide.md`.)

---

## License

MIT. See `LICENSE`.
