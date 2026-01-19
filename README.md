# Spine-Sim: 1D Axial Spine Impact Simulation

A simulation tool for predicting internal spinal forces during paragliding harness drop tests.

## Overview

This project simulates the axial (vertical) shock response of a human pelvis and thoracolumbar spine during impact. The primary output is the **T12-L1 junction force** time history, which correlates with thoracolumbar injury risk.

### Pipeline

The simulation uses a two-stage calibration approach:

1. **Buttocks Model (Toen 2012)**: Calibrate a viscoelastic buttocks tissue model using backward fall experimental data
2. **Spine Model (Yoganandan 2021)**: Calibrate the full spine model against FE-derived thoracolumbar disc forces
3. **Drop Simulation**: Apply calibrated model to paragliding harness drop test acceleration data

```
Toen Paper Data → calibrate-buttocks → Buttocks Model
                                             ↓
Yoganandan FE Data → calibrate-drop → Full Spine Model
                                             ↓
Drop Test CSV → simulate-drop → T12-L1 Force Output
```

## Installation

Requires Python 3.11+ with uv:

```bash
# Install dependencies
uv sync

# Or use the script directly (uv will handle dependencies)
./simulate.py --help
```

## Usage

### 1. Calibrate Buttocks Model

Calibrate the buttocks tissue model from Toen 2012 paper data:

```bash
./simulate.py calibrate-buttocks
```

This fits three parameters (stiffness, damping, compression limit) to match experimental ground forces on different floor stiffnesses at 3.5 m/s impact velocity.

Output: `calibration/toen_drop.json`

### 2. Simulate Buttocks Model (Optional)

Verify the buttocks model behavior and generate plots:

```bash
./simulate.py simulate-buttocks
```

Output: `output/toen_drop/summary.json` and force/compression plots.

### 3. Calibrate Spine Model

Calibrate the full spine model against Yoganandan FE data:

```bash
./simulate.py calibrate-drop
```

This uses the Toen-calibrated buttocks model and fits spine stiffness scales to match T12-L1 peak forces across 5 acceleration pulse durations (50-200 ms).

Output: `calibration/zwt.json` (or `calibration/maxwell.json`)

### 4. Run Drop Simulations

Process paragliding harness drop test data:

```bash
./simulate.py simulate-drop
```

Input: CSV files in `drops/` with columns `time` (or `time0`) and `accel` (or `acceleration`)

Output: `output/drop/<name>/timeseries.csv` with full time histories

## Scientific Background

### Model Structure

The simulation uses a 1D serial chain of masses connected by viscoelastic elements:

```
[Base/Seat] → [Buttocks] → [Pelvis] → [L5] → ... → [T12] → ... → [T1] → [Head]
```

- **19 nodes**: Pelvis, L5-L1, T12-T1, Head
- **19 elements**: Buttocks + 18 intervertebral disc elements
- **Nonlinear springs**: Polynomial stiffening under compression
- **Maxwell branches**: Rate-dependent viscoelastic response
- **Compression limits**: Densification behavior at large compressions

### Input Styles

The system automatically detects two input styles based on signal duration:

**Drop-style (≥300 ms)**: Paragliding harness drops
- Signal includes freefall (~-1g), impact peak, and post-impact
- Hit extraction isolates the contact window
- Initial state: unloaded

**Flat-style (<300 ms)**: Calibration pulses (Yoganandan-style)
- Subject starts at rest on platform
- Initial state: gravity-settled equilibrium

### Buttocks Model (Toen 2012)

The buttocks model captures soft tissue compliance during seated impact:

- **Source**: Toen et al. 2012 - backward falls onto buttocks at 3.5 m/s
- **Parameters**: Linear spring (k), viscous damper (c), compression limit with densification
- **Calibration targets**: Ground force peaks on floors of varying stiffness (59-400 kN/m)
- **Key constraint**: Maximum compression at rigid floor defines the densification limit

### Spine Model (Yoganandan 2021)

The spine stiffness distribution comes from Raj 2019 (based on Kitazaki & Griffin):

- **Calibration source**: Yoganandan 2021 FE simulations
- **Input**: Base acceleration pulses (11-46g, 50-200ms duration)
- **Target**: T12-L1 disc force peaks (3.3-7.6 kN)
- **Calibration parameters**: Spine stiffness scale (keeps buttocks fixed from Toen)

### Mass Distribution

Body masses are extracted from the OpenSim Male Thoracolumbar Full Body Model:

- Helmet mass added (+0.7 kg)
- Arms at 50% recruitment (partial dynamic coupling)
- Legs excluded (not in axial load path)

## Configuration

Edit `config.json` to customize:

```json
{
  "model": {
    "type": "zwt",           // Model type: "zwt" or "maxwell"
    "masses_json": "opensim/fullbody.json",
    "arm_recruitment": 0.5,  // Fraction of arm mass recruited
    "helmet_mass_kg": 0.7
  },
  "drop": {
    "inputs_dir": "drops",
    "pattern": "*.csv",
    "output_dir": "output/drop",
    "cfc": 75,               // CFC filter frequency
    "sim_duration_ms": 200.0
  },
  "buttock": {
    "target_set": "avg",     // "avg" or "subj3"
    "velocities_mps": [3.5, 8.0]
  }
}
```

## Output Files

### Calibration

- `calibration/toen_drop.json` - Buttocks model parameters
- `calibration/zwt.json` - Spine model calibration scales

### Simulation

- `output/drop/<name>/timeseries.csv` - Full time history:
  - `time_s`, `base_accel_g`
  - `y_<node>_mm` - Displacement per node
  - `v_<node>_mps` - Velocity per node
  - `F_<element>_kN` - Force per element

- `output/drop/summary.json` - Peak values for all runs

### Buttocks Validation

- `output/toen_drop/summary.json` - Results for all floor/velocity combinations
- `output/toen_drop/buttocks_force_compression_v*.png` - Response plots

## Coordinate Conventions

- **Axis**: Vertical (axial) only, aligned with spine
- **Compression**: Positive force values
- **Acceleration**: Inertial acceleration (idle=0g, freefall=-1g)
- **Gravity**: Included consistently

## Known Limitations

1. **1D Axial Only**: No bending, shear, or rotation effects
2. **Measured-Input**: Protector behavior is injected via measured acceleration, not predicted
3. **Human Variability**: Single calibration; no individual variation
4. **Posture**: Assumes upright axial loading (ignores ~22° recline in harness)

## References

- Toen et al., 2012: Buttocks tissue mechanics from backward falls
- Yoganandan et al., 2021: FE caudo-cephalad loading; thoracolumbar disc forces
- Kitazaki & Griffin, 1997: Biodynamic spine model parameters
- Raj & Krishnapillai, 2019: Baseline stiffness distribution
- Bruno/Allaire/Anderson: OpenSim thoracolumbar model for mass distribution

