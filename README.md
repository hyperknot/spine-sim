# Pelvis–Spine 1D Axial Shock Simulation (Base-Excitation From Drop Tests)

This README captures the **full rationale, decisions, assumptions, and pipeline** for the project as discussed.

---

## 0) Goal (What we are building)

We want a **1D axial (no rotation)** dynamic simulation of a **human pelvis + full thoracolumbar spine** (and head/cervical lump) experiencing a **bottom-up vertical shock**.

Primary output:
- **Internal T12–L1 axial compressive force time history** (compression positive).
- Also export full time histories for:
  - node displacements (pelvis, L5…T12…T1, head),
  - element/junction forces (buttocks, L5–S1 … T12–L1 … T1–Head),
  - to support "wave up the spine" visualization in 2D charts.

We explicitly do **not** simulate rotations, bending, wedge injury, shear, etc. (These are real mechanisms but outside the 1D scope.)

---

## 1) Context: Why not use DRI alone?

We discussed DRI (Dynamic Response Index) as a commonly used underbody blast / vertical shock criterion.

Key points:
- Classic DRI is effectively **one SDOF oscillator** with base excitation (not a multi-segment spine model).
- It often fails to reflect posture/bending/mass recruitment effects and may underpredict injury risk.
- Raj (2019) and others criticize DRI for not capturing posture and non-axial injury mechanisms.

We do **not** aim to replicate DRI.
We aim to produce **internal segment-level forces** (especially T12–L1) in a **distributed axial chain**.

---

## 2) Data sources and what each is used for

### 2.1 OpenSim thoracolumbar full body model (Allaire/Anderson/Bruno lineage)
We use OpenSim **only as a parameter source**:
- extract **segment masses** (pelvis, vertebrae, head/upper body).

We do **not** use OpenSim as the simulator.
We do **not** use OpenSim muscles/joints for dynamics in the shock simulation.

### 2.2 Yoganandan 2021 (FE under caudo-cephalad loading)
We use Yoganandan 2021 as the **human reference** for calibration:
- Input: seat/base acceleration pulses (duration 50–200 ms, 11–46 g)
- Output: **thoracolumbar disc "spinal forces"** (compressive plotted negative in their figures)
- Key qualitative trend: shorter rise time/duration → larger transmitted thoracolumbar force.

We **digitized** the 5 acceleration pulses and 5 force time histories and use them as calibration targets.

### 2.3 Kitazaki & Griffin (1997) and Raj (2019)
These provide baseline axial stiffness/damping style values for biodynamic/lumped models.

We decided:
- Use **Raj 2019 Appendix A** (which derives from Kitazaki/Griffin lineage) as **baseline spine axial stiffness distribution** and **baseline buttocks element properties**, then calibrate using a small number of scaling factors.

---

## 3) Experimental setup (our drop tests) and what it means

### 3.1 Dummy and sensor
- Dummy is a **single rigid solid metal torso block**.
- Orientation is fixed (vertical rail system), **no rotation**.
- 1-axis accelerometer is fixed to the dummy.
- Recorded inertial acceleration behaves like:
  - ~0 g at idle/rest
  - ~-1 g during freefall
  - large positive peak during impact (e.g. +40 g)
  - then dips below -1 g (e.g. -1.5 to -2 g) before returning near -1 g freefall until bounce

Sampling:
- 1000 Hz+
- Data is filtered via **CFC 75**, and is smooth.

### 3.2 Interpretation of measured acceleration
Because:
- sensor is on rigid dummy,
- dummy does not rotate,
- dummy+harness move together rigidly (except protector compression zone),

the measured acceleration during contact is effectively the acceleration of the **seat/harness structure** that the pelvis would sit on.

Decision:
- We treat measured drop-test acceleration as a **base excitation** signal for the human model, i.e.:
  - `a_base(t) := a_measured(t)` during the extracted contact window.

### 3.3 Important consequence ("use data as-is" philosophy)
We explicitly decided:
- **DO NOT model the protector/harness response** using springs/foam models.
- Protectors differ (Koroyd, airbag, vented airbag, foam, etc.).
- The point of the drop tests is to record those behaviors directly.

Therefore:
- Each protector is represented by its measured base acceleration during contact, not by a fitted protector constitutive model.

This approach is a **measured-input simulation**:
- It predicts internal human response given the measured seat/harness excitation,
- but does not predict how the protector would behave if a real human replaced the dummy (because the human would alter the protector compression history).

This is acceptable for ranking/comparison across protectors using a standardized dummy-based excitation.

---

## 4) Coordinate and sign conventions (critical decisions)

### 4.1 1D axis and posture
Real harness posture is about 22.5° reclined, but we decided:

- **Assume upright axial loading** for simplicity.
- No cosine projection; treat the axis as aligned with spine.

This makes results an "axial compression estimator," not a bending/shear injury estimator.

### 4.2 Gravity convention
We decided to include gravity consistently and rely on the fact that the data shows:
- idle ~0 g, freefall ~-1 g

This implies the stored signal is treated as **inertial acceleration** (not proper acceleration).

### 4.3 Compression force sign
We want compression forces **positive**.
Yoganandan plots compression as **negative**.
Decision:
- Flip Yoganandan sign so that the calibrated target force peaks are **positive**.
- Use consistent positive compression across all outputs.

---

## 5) Input styles and detection

There are **two input styles** determined by signal duration:

### 5.1 Drop-style (duration ≥ 300 ms)

Paragliding harness drop test standard:
- Starts with ~0 g (idle/rest)
- Drops to ~-1 g (freefall)
- Positive peak during impact
- Possible bounce

Processing:
- Find the first significant peak (> 5 g threshold)
- Expand left/right to freefall boundaries (-0.85 g)
- Bias-correct so freefall baseline equals -1 g
- Initial state: unloaded (y=0, v=0)

### 5.2 Flat-style (duration < 300 ms)

Excitor plate / Yoganandan-style calibration pulses:
- Subject is resting on ground, gravity-settled/pretensioned
- Starts at ~0 g
- Positive peak
- Returns to ~0 g

Processing:
- Use entire signal (no hit extraction)
- Baseline-correct initial samples to 0 g
- Run gravity-settling phase before applying pulse
- Initial state: static equilibrium under gravity

### 5.3 Detection rule

**Simple duration-based detection:**
- Total signal duration < 300 ms → **flat-style**
- Total signal duration ≥ 300 ms → **drop-style**

---

## 6) Model structure: 1D chain (base → buttocks → pelvis → spine → head)

We simulate a serial chain of masses connected by axial viscoelastic elements.

Topology:

1. **Base (seat/harness)**: prescribed acceleration input `a_base(t)`
2. **Buttocks element** (human tissue compliance)
3. **Pelvis mass**
4. Lumbar vertebrae: L5, L4, L3, L2, L1
5. Thoracic vertebrae: T12 … T1
6. **Head + cervical lump** (single mass node)

Outputs:
- Node displacement histories \( y_i(t) \)
- Element forces \( F_{i,i+1}(t) \)
- Primary output: \( F_{T12-L1}(t) \) internal element force time history.

This is **not** an OpenSim joint reaction force.
It is the internal through-force in the T12–L1 axial element.

---

## 7) Mass distribution: what we extract from OpenSim and how we modify it

### 7.1 We use OpenSim male model masses
Decision:
- Use OpenSim's male thoracolumbar full-body model as mass reference.
- Extract masses of bodies; map them to nodes in our chain.

### 7.2 Head/helmet
Decision:
- Add **helmet mass = 0.7 kg** to the head node.

### 7.3 Arms (partial recruitment)
Decision:
- Include arms but with reduced coupling:
  - Apply a recruitment factor (default **0.5**) to total arm mass.
  - Attach that recruited arm mass to the upper body (implemented as added into head/upper node lump).

Rationale:
- In 1D axial-only, any attached mass is recruited immediately; the factor is a pragmatic proxy for partial dynamic coupling.

### 7.4 Legs
Decision:
- Exclude legs from the chain (not recruited in the axial load path).
Rationale:
- We expect legs to hit first in reality; we simplify by not modeling legs as either mass recruitment or parallel load paths.

---

## 8) Element properties (baseline before calibration)

We need:
- buttocks stiffness/damping (human)
- intervertebral axial stiffness/damping distribution (human)

Decision:
- Use Raj 2019 Appendix A baseline values as initial parameter set:
  - axial disc stiffness distribution by level (N/m),
  - buttocks element stiffness/damping.

We lump cervical stiffness into a single equivalent series stiffness between T1 and head.

Damping baseline:
- use a simple constant per disc, with increased damping in thoracolumbar region as in Raj's description (e.g., 5× in T10–L5 region), then calibrate scalars.

---

## 9) Base excitation differences: flat-style vs drop-style

### 9.1 Flat-style pulses start and end at 0 g (inertial)
Interpretation:
- seat is not in freefall; occupant is gravity-settled.
So calibration runs must start from a **gravity-preloaded equilibrium**.

Implementation:
- compute static equilibrium or run a settling stage with `a_base(t)=0` before applying the pulse.

### 9.2 Drop-style segments start/end near -1 g (freefall)
Interpretation:
- base is in freefall before/after contact.
So drop simulations start with **unloaded/freefall initial state**.

Implementation:
- set initial `y0=0`, `v0=0` at the start of the extracted contact window.

---

## 10) Calibration strategy (fit human model to Yoganandan's 5 cases)

We have only 5 reference pulses, so we must keep calibration low-dimensional to avoid overfitting.

Decision:
- Calibrate only **4 global scalars**:

1. `s_k_spine`: multiplies all disc axial stiffnesses \(k_i\) (excluding buttocks)
2. `s_c_spine`: multiplies all disc damping values \(c_i\)
3. `s_k_butt`: multiplies buttocks stiffness
4. `s_c_butt`: multiplies buttocks damping

We calibrate by minimizing error between:
- predicted \(F_{T12-L1}(t)\) and digitized Yoganandan spinal force time history,
across all 5 pulses.

Calibration metrics:
- waveform residual (normalized by per-case peak),
- using least squares on time-aligned force curves.

Sign:
- Yoganandan compression is flipped to positive to match our convention.

---

## 11) Outputs and visualizations

For each simulated run (drop file):

1. **Displacement plot** (with acceleration subplot)
   - Top: displacement (mm) vs time (ms), one line per node
   - Bottom: filtered base acceleration (g) vs time (ms)

2. **Force plot** (with acceleration subplot)
   - Top: force (kN) vs time (ms), one line per element
   - Bottom: filtered base acceleration (g) vs time (ms)

3. **Mixed plot: displacement colored by force** (with acceleration subplot)
   - Top: node displacement curves colored by junction force below
   - Bottom: filtered base acceleration (g) vs time (ms)

4. **Gravity-settling plot** (flat-style only, no acceleration subplot)
   - Shows pre-simulation settling phase

Export:
- `timeseries.csv` per run containing:
  - time, base acceleration (g), all node displacements (mm), all element forces (kN)

---

## 12) Filtering and preprocessing

### CSV format
Input CSV files have at least columns:
- `time0` or `time`
- `accel` or `acceleration`

Example:
```
time0,accel
0.00038,-0.02
0.00076,-0.02
```

Time unit detection:
- If max(time) > 100 → assume milliseconds, else seconds.

### Resampling
We resample to a uniform time grid using:
- median dt of positive time deltas
- linear interpolation

### CFC Filter (SAE J211/1)
We implement CFC filtering:
- CFC → design frequency mapping: `f_design = 2.0775 * CFC`
- 2nd order Butterworth per pass (implemented as 2nd order SOS)
- forward + backward filtering (zero-phase), i.e. `sosfiltfilt`
- no padding (`padtype=None`, `padlen=0`)

Default CFC = 75.

---

## 13) Software structure

### Script 1: OpenSim extraction → JSON
- Reads `.osim`, extracts all body masses.
- Writes JSON with model name, total mass, body masses dict.

### Script 2: Main simulation + calibration
- Reads mass JSON
- Builds chain masses (with helmet + partial arms)
- Builds baseline stiffness/damping from Raj 2019
- Loads digitized calibration cases (accel + force)
- Calibrates the 4 scalars
- Processes all drop CSV files:
  - resample, CFC filter
  - style detection (duration-based)
  - hit-range extract (drop-style) or full signal (flat-style)
  - simulate
  - export `timeseries.csv` + PNG charts
  - save summary JSON with peak T12–L1 force per run

---

## 14) Known limitations (explicit)

1. **Measured-input assumption**:
   - The protector/harness behavior is not predicted; it is injected from dummy-derived acceleration.
   - A real human might change that base acceleration by altering contact mechanics.

2. **1D axial only**:
   - No bending/shear, no wedge injury prediction.
   - Reclined posture effects are ignored by assuming upright axial alignment.

3. **Legs omitted**:
   - No parallel load path via feet.
   - No leg mass recruitment.

4. **Human tissue parameters are uncertain at impact rates**:
   - We calibrate globally to Yoganandan FE trends.
   - This yields a plausible model but not a validated human surrogate for all scenarios.

---

## 15) How to run (high level)

1. Extract OpenSim masses:
   ```bash
   python extract_opensim_masses.py --model path/to/MaleFullBodyModel.osim --out masses.json
   ```

2. Create a `config.json` pointing to:
   - `masses.json`
   - digitized calibration accel/force CSVs
   - drop CSV folder
   - output folder

3. Run simulation:
   ```bash
   python simulate.py
   ```

Outputs will appear in:
- `output/`
  - `calibration_result.json`
  - `summary.json`
  - per-drop folder with:
    - `timeseries.csv`
    - `displacements.png`
    - `forces.png`
    - `mixed.png`
    - `gravity_settling.png` (flat-style only)

---

## 16) Key decisions summary

- Use OpenSim **only** for mass distribution.
- Simulate a 1D axial chain: base → buttocks → pelvis → vertebrae → head.
- Use protector behavior **as measured** (base acceleration), do not model protector mechanics.
- **Detection**: duration < 300 ms → flat-style; duration ≥ 300 ms → drop-style.
- **Drop-style**: extract contact segment via peak/freefall thresholds, start unloaded.
- **Flat-style**: use full signal, baseline to 0 g, start gravity-settled.
- Include gravity; rely on inertial acceleration convention (idle 0 g, freefall -1 g).
- Calibration uses Yoganandan 2021 digitized pulses + force histories.
- Calibrate only 4 global scalars (spine/buttocks stiffness/damping multipliers).
- Compression positive: flip Yoganandan sign.
- Upright axial axis (ignore recline cosine).
- Head included with cervical lump; helmet +0.7 kg.
- Arms included at 50% recruitment; legs excluded.

---

## 17) References

- Yoganandan et al., 2021: FE caudo-cephalad loading; thoracolumbar disc forces and injury risk vs pulse shape.
- Kitazaki & Griffin, 1997: distributed/FE biodynamic model; buttocks compliance critical for resonance behavior.
- Raj & Krishnapillai, 2019: improved injury parameter model; provides baseline stiffness tables.
- Bruno/Allaire/Anderson OpenSim thoracolumbar models: used for anthropometric mass distribution.
