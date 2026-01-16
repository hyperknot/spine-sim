from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from spine_sim.calibration import YogCase, apply_calibration, calibrate_to_yoganandan
from spine_sim.filters import cfc_filter
from spine_sim.io import TimeSeries, parse_csv_series, resample_to_uniform
from spine_sim.model import SpineModel, initial_state_static, newmark_linear
from spine_sim.plotting import plot_displacement_colored_by_force, plot_displacements, plot_forces
from spine_sim.range import find_first_hit_range


def normalize_name(name: str) -> str:
    return ''.join(c for c in name.lower() if c.isalnum())


def load_masses_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def sum_masses_by_patterns(masses: dict, patterns: list[str]) -> tuple[float, list[str]]:
    matched = []
    total = 0.0
    for name, m in masses.items():
        n = normalize_name(name)
        if any(p in n for p in patterns):
            matched.append(name)
            total += m
    return total, matched


def build_mass_map(masses: dict, arm_recruitment: float, helmet_mass: float) -> dict:
    body_masses = masses['bodies']

    # Pelvis
    pelvis_mass, pelvis_names = sum_masses_by_patterns(body_masses, ['pelvis'])

    # Lumbar and thoracic
    def find_level_mass(level: str) -> float:
        # level like "l1", "t12"
        patterns = [
            f'lumbar{level[1:]}' if level.startswith('l') else f'thoracic{level[1:]}',
            f'{level}',
        ]
        total, names = sum_masses_by_patterns(body_masses, patterns)
        return total

    lumbar_levels = ['l5', 'l4', 'l3', 'l2', 'l1']
    thoracic_levels = [f't{n}' for n in range(12, 0, -1)]

    lumbar_masses = {lvl: find_level_mass(lvl) for lvl in lumbar_levels}
    thoracic_masses = {lvl: find_level_mass(lvl) for lvl in thoracic_levels}

    # Head + cervical
    head_mass, head_names = sum_masses_by_patterns(body_masses, ['head', 'skull'])
    cerv_mass, cerv_names = sum_masses_by_patterns(body_masses, ['cerv'])

    # Arms
    arm_mass, arm_names = sum_masses_by_patterns(
        body_masses,
        ['humerus', 'ulna', 'radius', 'hand', 'clavicle', 'scapula'],
    )
    arm_mass *= arm_recruitment

    head_total = head_mass + cerv_mass + helmet_mass + arm_mass

    return {
        'pelvis': pelvis_mass,
        **lumbar_masses,
        **thoracic_masses,
        'head': head_total,
    }


def build_spine_model(mass_map: dict) -> SpineModel:
    node_names = [
        'pelvis',
        'L5',
        'L4',
        'L3',
        'L2',
        'L1',
        'T12',
        'T11',
        'T10',
        'T9',
        'T8',
        'T7',
        'T6',
        'T5',
        'T4',
        'T3',
        'T2',
        'T1',
        'HEAD',
    ]

    masses = np.array(
        [
            mass_map['pelvis'],
            mass_map['l5'],
            mass_map['l4'],
            mass_map['l3'],
            mass_map['l2'],
            mass_map['l1'],
            mass_map['t12'],
            mass_map['t11'],
            mass_map['t10'],
            mass_map['t9'],
            mass_map['t8'],
            mass_map['t7'],
            mass_map['t6'],
            mass_map['t5'],
            mass_map['t4'],
            mass_map['t3'],
            mass_map['t2'],
            mass_map['t1'],
            mass_map['head'],
        ],
        dtype=float,
    )

    # Raj 2019 axial stiffnesses (N/m)
    k = {
        'head-c1': 0.55e6,
        'c1-c2': 0.3e6,
        'c2-c3': 0.7e6,
        'c3-c4': 0.76e6,
        'c4-c5': 0.794e6,
        'c5-c6': 0.967e6,
        'c6-c7': 1.014e6,
        'c7-t1': 1.334e6,
        't1-t2': 0.7e6,
        't2-t3': 1.2e6,
        't3-t4': 1.5e6,
        't4-t5': 2.1e6,
        't5-t6': 1.9e6,
        't6-t7': 1.8e6,
        't7-t8': 1.5e6,
        't8-t9': 1.5e6,
        't9-t10': 1.5e6,
        't10-t11': 1.5e6,
        't11-t12': 1.5e6,
        't12-l1': 1.8e6,
        'l1-l2': 2.13e6,
        'l2-l3': 2.0e6,
        'l3-l4': 2.0e6,
        'l4-l5': 1.87e6,
        'l5-s1': 1.47e6,
    }

    # Cervical equivalent stiffness (series)
    cerv_keys = ['head-c1', 'c1-c2', 'c2-c3', 'c3-c4', 'c4-c5', 'c5-c6', 'c6-c7', 'c7-t1']
    k_cerv_eq = 1.0 / sum(1.0 / k[key] for key in cerv_keys)

    # Damping baseline
    c_base = 1200.0

    def c_disc(name: str) -> float:
        # Thoracolumbar boost (T10-L5)
        if name in ['t10-t11', 't11-t12', 't12-l1', 'l1-l2', 'l2-l3', 'l3-l4', 'l4-l5', 'l5-s1']:
            return 5.0 * c_base
        return c_base

    # Elements (buttocks + discs)
    element_names = [
        'buttocks',
        'L5-S1',
        'L4-L5',
        'L3-L4',
        'L2-L3',
        'L1-L2',
        'T12-L1',
        'T11-T12',
        'T10-T11',
        'T9-T10',
        'T8-T9',
        'T7-T8',
        'T6-T7',
        'T5-T6',
        'T4-T5',
        'T3-T4',
        'T2-T3',
        'T1-T2',
        'T1-HEAD',
    ]

    k_elem = [
        8.8425e4,  # buttocks
        k['l5-s1'],
        k['l4-l5'],
        k['l3-l4'],
        k['l2-l3'],
        k['l1-l2'],
        k['t12-l1'],
        k['t11-t12'],
        k['t10-t11'],
        k['t9-t10'],
        k['t8-t9'],
        k['t7-t8'],
        k['t6-t7'],
        k['t5-t6'],
        k['t4-t5'],
        k['t3-t4'],
        k['t2-t3'],
        k['t1-t2'],
        k_cerv_eq,
    ]

    c_elem = [
        1700.0,  # buttocks
        c_disc('l5-s1'),
        c_disc('l4-l5'),
        c_disc('l3-l4'),
        c_disc('l2-l3'),
        c_disc('l1-l2'),
        c_disc('t12-l1'),
        c_disc('t11-t12'),
        c_disc('t10-t11'),
        c_disc('t9-t10'),
        c_disc('t8-t9'),
        c_disc('t7-t8'),
        c_disc('t6-t7'),
        c_disc('t5-t6'),
        c_disc('t4-t5'),
        c_disc('t3-t4'),
        c_disc('t2-t3'),
        c_disc('t1-t2'),
        c_base / len(cerv_keys),
    ]

    return SpineModel(
        node_names=node_names,
        masses_kg=masses,
        element_names=element_names,
        k_elem=np.array(k_elem, dtype=float),
        c_elem=np.array(c_elem, dtype=float),
    )


def load_yog_cases(config: dict) -> list[YogCase]:
    cases = []
    force_sign = config.get('force_sign', -1.0)
    for c in config['cases']:
        accel_series = parse_csv_series(
            Path(c['accel_csv']),
            time_candidates=['time', 'time0', 't'],
            value_candidates=['accel', 'acceleration'],
        )
        force_series = parse_csv_series(
            Path(c['force_csv']),
            time_candidates=['time', 'time0', 't'],
            value_candidates=['force', 'spinal', 'load'],
        )

        accel_series, _ = resample_to_uniform(accel_series)
        force_series, _ = resample_to_uniform(force_series)

        accel_g = np.asarray(accel_series.values, dtype=float)
        force_n = np.asarray(force_series.values, dtype=float) * 1000.0 * force_sign  # kN -> N

        cases.append(
            YogCase(
                name=c['name'],
                time_s=np.asarray(accel_series.time_s, dtype=float),
                accel_g=accel_g,
                force_time_s=np.asarray(force_series.time_s, dtype=float),
                force_n=force_n,
            )
        )
    return cases


def process_drop_csv(
    path: Path, cfc: float, peak_g: float, freefall_g: float
) -> tuple[np.ndarray, np.ndarray]:
    series = parse_csv_series(
        path,
        time_candidates=['time', 'time0', 't'],
        value_candidates=['accel', 'acceleration'],
    )
    series, sample_rate = resample_to_uniform(series)

    accel_raw = series.values
    accel_filtered = cfc_filter(accel_raw, sample_rate, cfc)
    accel = np.asarray(accel_filtered, dtype=float)

    hit = find_first_hit_range(
        accel.tolist(), peak_threshold_g=peak_g, free_fall_threshold_g=freefall_g
    )
    if not hit:
        return np.asarray(series.time_s), accel

    t = np.asarray(series.time_s)
    start = hit.start_idx
    end = hit.end_idx
    return t[start : end + 1] - t[start], accel[start : end + 1]


def write_timeseries_csv(
    out_path: Path,
    time_s: np.ndarray,
    base_accel_g: np.ndarray,
    node_names: list[str],
    elem_names: list[str],
    y: np.ndarray,
    forces_n: np.ndarray,
) -> None:
    import csv

    headers = ['time_s', 'base_accel_g']
    headers += [f'y_{n}_mm' for n in node_names]
    headers += [f'F_{e}_kN' for e in elem_names]

    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(time_s.size):
            row = [
                f'{time_s[i]:.6f}',
                f'{base_accel_g[i]:.6f}',
            ]
            row += [f'{(y[i, j] * 1000.0):.6f}' for j in range(y.shape[1])]
            row += [f'{(forces_n[i, j] / 1000.0):.6f}' for j in range(forces_n.shape[1])]
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description='Spine chain simulation and calibration')
    parser.add_argument('--config', required=True, type=Path, help='Config JSON')
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding='utf-8'))

    # Load masses
    masses = load_masses_json(Path(config['masses_json']))
    arm_recruitment = float(config['model'].get('arm_recruitment', 0.5))
    helmet_mass = float(config['model'].get('helmet_mass_kg', 0.7))
    mass_map = build_mass_map(masses, arm_recruitment=arm_recruitment, helmet_mass=helmet_mass)

    model = build_spine_model(mass_map)

    # T12-L1 element index
    t12_elem_idx = model.element_names.index('T12-L1')

    out_dir = Path(config['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calibration
    calib_cfg = config.get('yoganandan')
    if calib_cfg:
        cases = load_yog_cases(calib_cfg)
        calib = calibrate_to_yoganandan(model, cases, t12_elem_idx)

        calib_out = out_dir / 'calibration_result.json'
        calib_out.write_text(
            json.dumps(
                {
                    'scales': calib.scales,
                    'success': calib.success,
                    'cost': calib.cost,
                    'residual_norm': calib.residual_norm,
                },
                indent=2,
            )
        )

        model = apply_calibration(model, calib.scales)

    # Drop simulations
    drops_cfg = config['drops']
    input_dir = Path(drops_cfg['input_dir'])
    pattern = drops_cfg.get('pattern', '*.csv')

    cfc = float(config['processing'].get('cfc', 75))
    peak_g = float(config['processing'].get('peak_threshold_g', 5.0))
    freefall_g = float(config['processing'].get('free_fall_threshold_g', -0.85))

    drop_files = sorted(input_dir.glob(pattern))
    summary = []

    for fpath in drop_files:
        t, a_base_g = process_drop_csv(fpath, cfc=cfc, peak_g=peak_g, freefall_g=freefall_g)

        # Initial state in freefall baseline (a_base near -g)
        y0 = np.zeros(model.size(), dtype=float)
        v0 = np.zeros(model.size(), dtype=float)

        sim = newmark_linear(model, t, a_base_g, y0, v0)
        forces = sim.element_forces_n
        f_t12 = forces[:, t12_elem_idx]

        run_dir = out_dir / fpath.stem
        run_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        write_timeseries_csv(
            run_dir / 'timeseries.csv',
            sim.time_s,
            sim.base_accel_g,
            model.node_names,
            model.element_names,
            sim.y,
            forces,
        )

        # Charts
        plot_displacements(sim.time_s, sim.y, model.node_names, run_dir / 'displacements.png')
        plot_forces(
            sim.time_s, forces, model.element_names, run_dir / 'forces.png', highlight='T12-L1'
        )
        plot_displacement_colored_by_force(
            sim.time_s, sim.y, forces, model.node_names, model.element_names, run_dir / 'mixed.png'
        )

        summary.append(
            {
                'file': fpath.name,
                'peak_T12_L1_kN': float(np.max(f_t12) / 1000.0),
                'time_to_peak_ms': float(sim.time_s[np.argmax(f_t12)] * 1000.0),
            }
        )

    summary_path = out_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
