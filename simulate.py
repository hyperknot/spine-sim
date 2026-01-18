#!/usr/bin/env -S uv run

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spine_sim.calibration import YogCase, apply_calibration, calibrate_to_yoganandan
from spine_sim.filters import cfc_filter
from spine_sim.io import parse_csv_series, resample_to_uniform
from spine_sim.model import SpineModel, initial_state_static, newmark_nonlinear
from spine_sim.plotting import (
    plot_displacement_colored_by_force,
    plot_displacements,
    plot_forces,
    plot_gravity_tensioning,
)
from spine_sim.range import find_first_hit_range
from spine_sim.yoganandan_targets import YOG2021_T12L1_PEAKS_KN, get_case_name_from_filename


# Processing constants
CFC = 75
PEAK_THRESHOLD_G = 5.0
FREE_FALL_THRESHOLD_G = -0.85

# Always simulate this long for drops (ms)
DROP_SIM_DURATION_MS = 200.0

# Default masses JSON (can be overridden via config["model"]["masses_json"])
DEFAULT_MASSES_JSON_PATH = Path(__file__).parent / "opensim" / "fullbody.json"

# Drop inputs
DROPS_DIR = Path(__file__).parent / "drops"
DROPS_PATTERN = "*.csv"

# Yoganandan calibration data directory
YOGANANDAN_DIR = Path(__file__).parent / "yoganandan"

# Yoganandan force sign: -1 because their convention reports compression as negative,
# but our model uses compression as positive
YOGANANDAN_FORCE_SIGN = -1.0


@dataclass
class DropProcessingInfo:
    sample_rate_hz: float
    dt_s: float
    start_idx: int
    end_idx: int
    duration_s: float
    freefall_median_g: float | None
    bias_correction_g: float
    style: str  # "drop" or "yoganandan"


def load_masses_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_mass_map(masses: dict, arm_recruitment: float, helmet_mass: float) -> dict:
    """Build mass map using exact body names from OpenSim fullbody model."""
    b = masses["bodies"]

    arm_mass = (
        b["humerus_R"]
        + b["humerus_L"]
        + b["ulna_R"]
        + b["ulna_L"]
        + b["radius_R"]
        + b["radius_L"]
        + b["hand_R"]
        + b["hand_L"]
    ) * arm_recruitment

    head_total = b["head_neck"] + helmet_mass + arm_mass

    return {
        "pelvis": b["pelvis"],
        "l5": b["lumbar5"],
        "l4": b["lumbar4"],
        "l3": b["lumbar3"],
        "l2": b["lumbar2"],
        "l1": b["lumbar1"],
        "t12": b["thoracic12"],
        "t11": b["thoracic11"],
        "t10": b["thoracic10"],
        "t9": b["thoracic9"],
        "t8": b["thoracic8"],
        "t7": b["thoracic7"],
        "t6": b["thoracic6"],
        "t5": b["thoracic5"],
        "t4": b["thoracic4"],
        "t3": b["thoracic3"],
        "t2": b["thoracic2"],
        "t1": b["thoracic1"],
        "head": head_total,
    }


def _freefall_bias_correct(accel_g: np.ndarray) -> tuple[np.ndarray, float | None, float]:
    """
    Ensure the freefall baseline is -1 g as per README convention.
    We estimate freefall from samples clearly below ~-0.5 g.
    """
    mask = accel_g < -0.5
    if not np.any(mask):
        return accel_g, None, 0.0

    ff_med = float(np.median(accel_g[mask]))
    bias = -1.0 - ff_med
    return accel_g + bias, ff_med, bias


def _baseline_zero_correct(accel_g: np.ndarray) -> tuple[np.ndarray, float]:
    """
    For Yoganandan pulses: baseline is expected ~0 g inertial before pulse.
    """
    n = min(len(accel_g), 200)
    base = float(np.median(accel_g[:n]))
    return accel_g - base, base


def _detect_input_style(time_s: np.ndarray, accel_g: np.ndarray) -> str:
    """
    User-requested rule:
      - if duration < 300 ms AND there are no negative values => yoganandan style
      - else => drop style
    """
    duration_ms = float((time_s[-1] - time_s[0]) * 1000.0) if time_s.size >= 2 else 0.0
    if duration_ms < 300.0 and float(np.min(accel_g)) >= 0.0:
        return "yoganandan"
    return "drop"


def build_spine_model(mass_map: dict, nonlinear_cfg: dict | None = None) -> SpineModel:
    nonlinear_cfg = nonlinear_cfg or {}

    node_names = [
        "pelvis",
        "L5",
        "L4",
        "L3",
        "L2",
        "L1",
        "T12",
        "T11",
        "T10",
        "T9",
        "T8",
        "T7",
        "T6",
        "T5",
        "T4",
        "T3",
        "T2",
        "T1",
        "HEAD",
    ]

    masses = np.array(
        [
            mass_map["pelvis"],
            mass_map["l5"],
            mass_map["l4"],
            mass_map["l3"],
            mass_map["l2"],
            mass_map["l1"],
            mass_map["t12"],
            mass_map["t11"],
            mass_map["t10"],
            mass_map["t9"],
            mass_map["t8"],
            mass_map["t7"],
            mass_map["t6"],
            mass_map["t5"],
            mass_map["t4"],
            mass_map["t3"],
            mass_map["t2"],
            mass_map["t1"],
            mass_map["head"],
        ],
        dtype=float,
    )

    # Raj 2019 axial stiffnesses (N/m) used as equilibrium stiffness baseline.
    k = {
        "head-c1": 0.55e6,
        "c1-c2": 0.3e6,
        "c2-c3": 0.7e6,
        "c3-c4": 0.76e6,
        "c4-c5": 0.794e6,
        "c5-c6": 0.967e6,
        "c6-c7": 1.014e6,
        "c7-t1": 1.334e6,
        "t1-t2": 0.7e6,
        "t2-t3": 1.2e6,
        "t3-t4": 1.5e6,
        "t4-t5": 2.1e6,
        "t5-t6": 1.9e6,
        "t6-t7": 1.8e6,
        "t7-t8": 1.5e6,
        "t8-t9": 1.5e6,
        "t9-t10": 1.5e6,
        "t10-t11": 1.5e6,
        "t11-t12": 1.5e6,
        "t12-l1": 1.8e6,
        "l1-l2": 2.13e6,
        "l2-l3": 2.0e6,
        "l3-l4": 2.0e6,
        "l4-l5": 1.87e6,
        "l5-s1": 1.47e6,
    }

    cerv_keys = ["head-c1", "c1-c2", "c2-c3", "c3-c4", "c4-c5", "c5-c6", "c6-c7", "c7-t1"]
    k_cerv_eq = 1.0 / sum(1.0 / k[key] for key in cerv_keys)

    # Kelvin damping baseline (kept modest because Maxwell branches provide rate effects)
    c_base = float(nonlinear_cfg.get("c_base_ns_per_m", 1200.0))

    def c_disc(name: str) -> float:
        if name in ["t10-t11", "t11-t12", "t12-l1", "l1-l2", "l2-l3", "l3-l4", "l4-l5", "l5-s1"]:
            return 3.0 * c_base
        return c_base

    element_names = [
        "buttocks",
        "L5-S1",
        "L4-L5",
        "L3-L4",
        "L2-L3",
        "L1-L2",
        "T12-L1",
        "T11-T12",
        "T10-T11",
        "T9-T10",
        "T8-T9",
        "T7-T8",
        "T6-T7",
        "T5-T6",
        "T4-T5",
        "T3-T4",
        "T2-T3",
        "T1-T2",
        "T1-HEAD",
    ]

    k_elem = np.array(
        [
            8.8425e4,  # buttocks equilibrium baseline (will stiffen nonlinearly + Maxwell)
            k["l5-s1"],
            k["l4-l5"],
            k["l3-l4"],
            k["l2-l3"],
            k["l1-l2"],
            k["t12-l1"],
            k["t11-t12"],
            k["t10-t11"],
            k["t9-t10"],
            k["t8-t9"],
            k["t7-t8"],
            k["t6-t7"],
            k["t5-t6"],
            k["t4-t5"],
            k["t3-t4"],
            k["t2-t3"],
            k["t1-t2"],
            k_cerv_eq,
        ],
        dtype=float,
    )

    c_elem = np.array(
        [
            1700.0,  # buttocks Kelvin
            c_disc("l5-s1"),
            c_disc("l4-l5"),
            c_disc("l3-l4"),
            c_disc("l2-l3"),
            c_disc("l1-l2"),
            c_disc("t12-l1"),
            c_disc("t11-t12"),
            c_disc("t10-t11"),
            c_disc("t9-t10"),
            c_disc("t8-t9"),
            c_disc("t7-t8"),
            c_disc("t6-t7"),
            c_disc("t5-t6"),
            c_disc("t4-t5"),
            c_disc("t3-t4"),
            c_disc("t2-t3"),
            c_disc("t1-t2"),
            c_base / len(cerv_keys),
        ],
        dtype=float,
    )

    # --- Nonlinear equilibrium spring settings ---
    disc_ref_mm = float(nonlinear_cfg.get("disc_ref_compression_mm", 2.0))
    disc_kmult = float(nonlinear_cfg.get("disc_k_mult_at_ref", 8.0))

    butt_ref_mm = float(nonlinear_cfg.get("buttocks_ref_compression_mm", 25.0))
    butt_kmult = float(nonlinear_cfg.get("buttocks_k_mult_at_ref", 20.0))
    butt_gap_mm = float(nonlinear_cfg.get("buttocks_gap_mm", 0.0))

    compression_ref_m = np.zeros_like(k_elem, dtype=float)
    compression_k_mult = np.ones_like(k_elem, dtype=float)
    tension_k_mult = np.ones_like(k_elem, dtype=float)

    compression_only = np.zeros_like(k_elem, dtype=bool)
    damping_compression_only = np.zeros_like(k_elem, dtype=bool)
    gap_m = np.zeros_like(k_elem, dtype=float)

    # Buttocks element (index 0)
    compression_ref_m[0] = butt_ref_mm / 1000.0
    compression_k_mult[0] = butt_kmult
    compression_only[0] = True
    damping_compression_only[0] = True
    gap_m[0] = butt_gap_mm / 1000.0

    # Spine elements (index 1..)
    compression_ref_m[1:] = disc_ref_mm / 1000.0
    compression_k_mult[1:] = disc_kmult
    tension_k_mult[1:] = 1.0

    # --- Maxwell branches (rate dependence) ---
    # Defaults target the 50–200 ms regime: one branch ~10 ms, one branch ~120 ms.
    mx_k_ratios = nonlinear_cfg.get("maxwell_k_ratios", [1.0, 0.5])
    mx_tau_ms = nonlinear_cfg.get("maxwell_tau_ms", [10.0, 120.0])

    mx_k_ratios = [float(x) for x in mx_k_ratios]
    mx_tau_ms = [float(x) for x in mx_tau_ms]
    B = max(len(mx_k_ratios), len(mx_tau_ms))
    mx_k_ratios = (mx_k_ratios + [0.0] * B)[:B]
    mx_tau_ms = (mx_tau_ms + [0.0] * B)[:B]

    maxwell_k = np.zeros((len(k_elem), B), dtype=float)
    maxwell_tau_s = np.zeros((len(k_elem), B), dtype=float)

    # Apply to all elements as scaled-by-equilibrium baseline
    for e in range(len(k_elem)):
        for b in range(B):
            maxwell_k[e, b] = k_elem[e] * mx_k_ratios[b]
            maxwell_tau_s[e, b] = mx_tau_ms[b] / 1000.0

    maxwell_compression_only = np.zeros(len(k_elem), dtype=bool)
    # For impact modeling we keep Maxwell compression-only everywhere to avoid non-physical tension during rebound
    maxwell_compression_only[:] = True

    return SpineModel(
        node_names=node_names,
        masses_kg=masses,
        element_names=element_names,
        k_elem=k_elem,
        c_elem=c_elem,
        compression_ref_m=compression_ref_m,
        compression_k_mult=compression_k_mult,
        tension_k_mult=tension_k_mult,
        compression_only=compression_only,
        damping_compression_only=damping_compression_only,
        gap_m=gap_m,
        maxwell_k=maxwell_k,
        maxwell_tau_s=maxwell_tau_s,
        maxwell_compression_only=maxwell_compression_only,
    )


def load_yog_cases(cases_config: list[dict]) -> list[YogCase]:
    cases = []
    for c in cases_config:
        accel_series = parse_csv_series(
            YOGANANDAN_DIR / c["accel_csv"],
            time_candidates=["time", "time0", "t"],
            value_candidates=["accel", "acceleration"],
        )
        force_series = parse_csv_series(
            YOGANANDAN_DIR / c["force_csv"],
            time_candidates=["time", "time0", "t"],
            value_candidates=["force", "spinal", "load"],
        )

        accel_series, _ = resample_to_uniform(accel_series)
        force_series, _ = resample_to_uniform(force_series)

        accel_g = np.asarray(accel_series.values, dtype=float)
        accel_g, _baseline = _baseline_zero_correct(accel_g)

        force_n = np.asarray(force_series.values, dtype=float) * 1000.0 * YOGANANDAN_FORCE_SIGN

        cases.append(
            YogCase(
                name=c["name"],
                time_s=np.asarray(accel_series.time_s, dtype=float),
                accel_g=accel_g,
                force_time_s=np.asarray(force_series.time_s, dtype=float),
                force_n=force_n,
            )
        )
    return cases


def process_drop_csv_fixed_duration(
    path: Path,
    *,
    cfc: float,
    peak_g: float,
    freefall_g: float,
    duration_ms: float,
) -> tuple[np.ndarray, np.ndarray, DropProcessingInfo]:
    series = parse_csv_series(
        path,
        time_candidates=["time", "time0", "t"],
        value_candidates=["accel", "acceleration"],
    )
    series, sample_rate = resample_to_uniform(series)
    dt = 1.0 / sample_rate

    accel_raw = series.values
    accel_filtered = np.asarray(cfc_filter(accel_raw, sample_rate, cfc), dtype=float)

    t_all = np.asarray(series.time_s, dtype=float)

    # DEBUG: trace min values at each stage
    duration_ms_raw = float((t_all[-1] - t_all[0]) * 1000.0) if t_all.size >= 2 else 0.0
    print(f"  DEBUG: raw min={np.min(accel_raw):.4f}, max={np.max(accel_raw):.4f}")
    print(f"  DEBUG: filtered min={np.min(accel_filtered):.4f}, max={np.max(accel_filtered):.4f}")
    print(f"  DEBUG: duration={duration_ms_raw:.2f} ms")

    style = _detect_input_style(t_all, accel_filtered)

    if style == "yoganandan":
        # No hit extraction; baseline ~0 g.
        start_idx = 0
        end_idx = min(len(t_all) - 1, int(round((duration_ms / 1000.0) / dt)))
        t_seg = t_all[start_idx : end_idx + 1] - t_all[start_idx]
        a_seg = accel_filtered[start_idx : end_idx + 1]

        print(f"  DEBUG yog: a_seg before baseline_zero_correct min={np.min(a_seg):.4f}, max={np.max(a_seg):.4f}, len={len(a_seg)}")
        a_seg, _base0 = _baseline_zero_correct(a_seg)
        print(f"  DEBUG yog: baseline correction applied: {_base0:.4f}")
        print(f"  DEBUG yog: a_seg after baseline_zero_correct min={np.min(a_seg):.4f}, max={np.max(a_seg):.4f}")

        desired_n = int(round((duration_ms / 1000.0) / dt)) + 1
        if len(t_seg) < desired_n:
            pad_n = desired_n - len(t_seg)
            t_pad = t_seg[-1] + dt * (np.arange(pad_n) + 1)
            a_pad = 0.0 * np.ones(pad_n, dtype=float)
            t_seg = np.concatenate([t_seg, t_pad])
            a_seg = np.concatenate([a_seg, a_pad])
            print(f"  DEBUG yog: after padding min={np.min(a_seg):.4f}, max={np.max(a_seg):.4f}, len={len(a_seg)}")

        info = DropProcessingInfo(
            sample_rate_hz=float(sample_rate),
            dt_s=float(dt),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            duration_s=float(t_seg[-1] - t_seg[0]),
            freefall_median_g=None,
            bias_correction_g=0.0,
            style="yoganandan",
        )
        return t_seg, a_seg, info

    # --- drop-style behavior (as before) ---
    hit = find_first_hit_range(
        accel_filtered.tolist(),
        peak_threshold_g=peak_g,
        free_fall_threshold_g=freefall_g,
    )

    if not hit:
        start_idx = 0
        end_idx = min(len(t_all) - 1, int(round((duration_ms / 1000.0) / dt)))
    else:
        start_idx = hit.start_idx
        end_idx = min(len(t_all) - 1, start_idx + int(round((duration_ms / 1000.0) / dt)))

    t_seg = t_all[start_idx : end_idx + 1] - t_all[start_idx]
    a_seg = accel_filtered[start_idx : end_idx + 1]

    a_seg, ff_med, bias = _freefall_bias_correct(a_seg)

    desired_n = int(round((duration_ms / 1000.0) / dt)) + 1
    if len(t_seg) < desired_n:
        pad_n = desired_n - len(t_seg)
        t_pad = t_seg[-1] + dt * (np.arange(pad_n) + 1)
        a_pad = -1.0 * np.ones(pad_n, dtype=float)  # freefall
        t_seg = np.concatenate([t_seg, t_pad])
        a_seg = np.concatenate([a_seg, a_pad])

    info = DropProcessingInfo(
        sample_rate_hz=float(sample_rate),
        dt_s=float(dt),
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        duration_s=float(t_seg[-1] - t_seg[0]),
        freefall_median_g=ff_med,
        bias_correction_g=float(bias),
        style="drop",
    )
    return t_seg, a_seg, info


def write_timeseries_csv(
    out_path: Path,
    time_s: np.ndarray,
    base_accel_g: np.ndarray,
    node_names: list[str],
    elem_names: list[str],
    y: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    forces_n: np.ndarray,
) -> None:
    import csv

    headers = ["time_s", "base_accel_g"]
    headers += [f"y_{n}_mm" for n in node_names]
    headers += [f"v_{n}_mps" for n in node_names]
    headers += [f"a_{n}_mps2" for n in node_names]
    headers += [f"F_{e}_kN" for e in elem_names]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(time_s.size):
            row = [
                f"{time_s[i]:.6f}",
                f"{base_accel_g[i]:.6f}",
            ]
            row += [f"{(y[i, j] * 1000.0):.6f}" for j in range(y.shape[1])]
            row += [f"{v[i, j]:.6f}" for j in range(v.shape[1])]
            row += [f"{a[i, j]:.6f}" for j in range(a.shape[1])]
            row += [f"{(forces_n[i, j] / 1000.0):.6f}" for j in range(forces_n.shape[1])]
            w.writerow(row)


def main() -> None:
    config_path = Path(__file__).parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    masses_json_path = Path(config.get("model", {}).get("masses_json", str(DEFAULT_MASSES_JSON_PATH)))
    masses_data = load_masses_json(masses_json_path)

    arm_recruitment = float(config["model"].get("arm_recruitment", 0.5))
    helmet_mass = float(config["model"].get("helmet_mass_kg", 0.7))
    mass_map = build_mass_map(masses_data, arm_recruitment=arm_recruitment, helmet_mass=helmet_mass)

    heights_from_model = masses_data.get("heights_relative_to_pelvis_mm", None)

    nonlinear_cfg = config.get("nonlinear", {})
    model = build_spine_model(mass_map, nonlinear_cfg=nonlinear_cfg)

    t12_elem_idx = model.element_names.index("T12-L1")
    butt_elem_idx = model.element_names.index("buttocks")

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calibration (waveform-based; optional)
    calib_cfg = config.get("yoganandan")
    if calib_cfg and "cases" in calib_cfg:
        cases = load_yog_cases(calib_cfg["cases"])
        calib = calibrate_to_yoganandan(model, cases, t12_elem_idx)

        (out_dir / "calibration_result.json").write_text(
            json.dumps(
                {
                    "scales": calib.scales,
                    "success": calib.success,
                    "cost": calib.cost,
                    "residual_norm": calib.residual_norm,
                    "note": "Viscoelastic model: scalars apply to equilibrium k/c and Maxwell k (not tau).",
                },
                indent=2,
            )
        )

        model = apply_calibration(model, calib.scales)

        print("Calibration complete:")
        print(f'  s_k_spine = {calib.scales["s_k_spine"]:.3f}')
        print(f'  s_c_spine = {calib.scales["s_c_spine"]:.3f}')
        print(f'  s_k_butt  = {calib.scales["s_k_butt"]:.3f}')
        print(f'  s_c_butt  = {calib.scales["s_c_butt"]:.3f}')
        print("  Nonlinear/visco settings:")
        print(f'    disc_ref_compression_mm = {nonlinear_cfg.get("disc_ref_compression_mm", 2.0)}')
        print(f'    disc_k_mult_at_ref      = {nonlinear_cfg.get("disc_k_mult_at_ref", 8.0)}')
        print(f'    butt_ref_compression_mm = {nonlinear_cfg.get("buttocks_ref_compression_mm", 25.0)}')
        print(f'    butt_k_mult_at_ref      = {nonlinear_cfg.get("buttocks_k_mult_at_ref", 20.0)}')
        print(f'    maxwell_k_ratios         = {nonlinear_cfg.get("maxwell_k_ratios", [1.0, 0.5])}')
        print(f'    maxwell_tau_ms           = {nonlinear_cfg.get("maxwell_tau_ms", [10.0, 120.0])}')

    drop_files = sorted(DROPS_DIR.glob(DROPS_PATTERN))
    summary = []

    # gravity-settling parameters for yoganandan-style inputs
    settle_ms = float(config.get("yoganandan_settle_ms", 150.0))

    for fpath in drop_files:
        print(f"Processing {fpath.name}...")

        t, a_base_g, info = process_drop_csv_fixed_duration(
            fpath,
            cfc=CFC,
            peak_g=PEAK_THRESHOLD_G,
            freefall_g=FREE_FALL_THRESHOLD_G,
            duration_ms=DROP_SIM_DURATION_MS,
        )

        # Initial conditions:
        # - drop: start free (y=v=0) because base has freefall history (-1 g).
        # - yoganandan: run a gravity-settle stage (base accel = 0) and use its final state.
        y0 = np.zeros(model.size(), dtype=float)
        v0 = np.zeros(model.size(), dtype=float)
        s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

        run_dir = out_dir / fpath.stem
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        if info.style == "yoganandan":
            dt = info.dt_s
            n_settle = int(round((settle_ms / 1000.0) / dt)) + 1
            t_settle = dt * np.arange(n_settle)
            a_settle = np.zeros_like(t_settle)

            sim_settle = newmark_nonlinear(model, t_settle, a_settle, y0, v0, s0)

            plot_gravity_tensioning(
                sim_settle.time_s,
                sim_settle.y,
                model.node_names,
                run_dir / "gravity_tensioning.png",
            )

            y0 = sim_settle.y[-1].copy()
            v0 = sim_settle.v[-1].copy()
            s0 = sim_settle.maxwell_state_n[-1].copy()

        sim = newmark_nonlinear(model, t, a_base_g, y0, v0, s0)
        forces = sim.element_forces_n

        f_t12 = forces[:, t12_elem_idx]
        f_butt = forces[:, butt_elem_idx]

        head_idx = model.node_names.index("HEAD")
        pelvis_idx = model.node_names.index("pelvis")

        max_head_compression_mm = -float(np.min(sim.y[:, head_idx]) * 1000.0)
        max_pelvis_compression_mm = -float(np.min(sim.y[:, pelvis_idx]) * 1000.0)
        max_spine_shortening_mm = max_head_compression_mm - max_pelvis_compression_mm

        peak_base_g = float(np.max(a_base_g))
        min_base_g = float(np.min(a_base_g))
        peak_t12_kN = float(np.max(f_t12) / 1000.0)
        peak_butt_kN = float(np.max(f_butt) / 1000.0)

        write_timeseries_csv(
            run_dir / "timeseries.csv",
            sim.time_s,
            sim.base_accel_g,
            model.node_names,
            model.element_names,
            sim.y,
            sim.v,
            sim.a,
            forces,
        )

        plot_displacements(
            sim.time_s,
            sim.y,
            model.node_names,
            run_dir / "displacements.png",
            heights_from_model=heights_from_model,
            reference_frame="pelvis",
        )
        plot_forces(
            sim.time_s,
            forces,
            model.element_names,
            run_dir / "forces.png",
            highlight="T12-L1",
        )
        plot_displacement_colored_by_force(
            sim.time_s,
            sim.y,
            forces,
            model.node_names,
            model.element_names,
            run_dir / "mixed.png",
            heights_from_model=heights_from_model,
            reference_frame="pelvis",
        )

        # Element compression metrics (max compression in mm)
        elem_max_comp_mm: dict[str, float] = {}
        for e_idx, ename in enumerate(model.element_names):
            if e_idx == 0:
                comp = np.maximum(-(sim.y[:, 0] + model.gap_m[0]), 0.0)
            else:
                lower = e_idx - 1
                upper = e_idx
                comp = np.maximum(-(sim.y[:, upper] - sim.y[:, lower] + model.gap_m[e_idx]), 0.0)
            elem_max_comp_mm[ename] = float(np.max(comp) * 1000.0)

        print("  Input/base:")
        print(f"    style: {info.style}")
        print(f"    sample_rate: {info.sample_rate_hz:.1f} Hz, dt: {info.dt_s*1000.0:.3f} ms")
        print(f"    sim duration: {DROP_SIM_DURATION_MS:.0f} ms (forced)")
        print(f"    base accel peak: {peak_base_g:.2f} g, min: {min_base_g:.2f} g")

        print("  Forces:")
        print(f"    Peak buttocks: {peak_butt_kN:.2f} kN")
        print(f"    Peak T12-L1:   {peak_t12_kN:.2f} kN @ {float(sim.time_s[np.argmax(f_t12)]*1000.0):.1f} ms")

        # If filename matches a Yoganandan case, print reference peak
        case_name = get_case_name_from_filename(fpath.stem)
        if case_name and case_name in YOG2021_T12L1_PEAKS_KN:
            ref = YOG2021_T12L1_PEAKS_KN[case_name]
            print(f"    Yoganandan 2021 ref (T12-L1 peak): {ref:.2f} kN (Table 2)")

        print("  Displacements:")
        print(f"    Head compression:  {max_head_compression_mm:.1f} mm")
        print(f"    Pelvis compression:{max_pelvis_compression_mm:.1f} mm")
        print(f"    Spine shortening:  {max_spine_shortening_mm:.1f} mm")

        top_comp = sorted(elem_max_comp_mm.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("  Max element compressions (top 5):")
        for name, mm in top_comp:
            print(f"    {name:>8s}: {mm:7.2f} mm")

        summary.append(
            {
                "file": fpath.name,
                "processing": {
                    "style": info.style,
                    "sample_rate_hz": info.sample_rate_hz,
                    "dt_s": info.dt_s,
                    "start_idx": info.start_idx,
                    "end_idx": info.end_idx,
                    "freefall_median_g": info.freefall_median_g,
                    "bias_correction_g": info.bias_correction_g,
                    "sim_duration_ms": DROP_SIM_DURATION_MS,
                },
                "base_accel": {
                    "peak_g": peak_base_g,
                    "min_g": min_base_g,
                },
                "peak_forces_kN": {
                    "buttocks": peak_butt_kN,
                    "T12-L1": peak_t12_kN,
                },
                "time_to_peak_ms": float(sim.time_s[np.argmax(f_t12)] * 1000.0),
                "max_head_compression_mm": max_head_compression_mm,
                "max_pelvis_compression_mm": max_pelvis_compression_mm,
                "max_spine_shortening_mm": max_spine_shortening_mm,
                "max_element_compression_mm": elem_max_comp_mm,
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
