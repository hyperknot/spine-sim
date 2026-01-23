"""Drop simulation command (simplified: simulate-drop only)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from spine_sim.input_processing import process_input
from spine_sim.mass import build_mass_map
from spine_sim.model import initial_state_static, newmark_nonlinear
from spine_sim.model_components import build_spine_model
from spine_sim.output import write_timeseries_csv
from spine_sim.plotting import (
    plot_displacement_colored_by_force,
    plot_displacements,
    plot_forces,
    plot_gravity_settling,
)
from spine_sim.range import find_hit_range
from spine_sim.settings import (
    REPO_ROOT,
    load_json,
    read_config,
    req_float,
    req_int,
    req_str,
    resolve_path,
)


# Default IO
INPUT_BASE = REPO_ROOT / 'input'
INPUT_PATTERN = '*.csv'
OUTPUT_BASE = REPO_ROOT / 'output'


def _get_plotting_config(config: dict) -> tuple[float, dict | None]:
    buttocks_height_mm = float(req_float(config, ['plotting', 'buttocks_height_mm']))

    masses_path = resolve_path(req_str(config, ['model', 'masses_json']))
    masses = load_json(masses_path)
    heights_from_model = masses.get('heights_relative_to_pelvis_mm', None)

    return buttocks_height_mm, heights_from_model


def _interpolate_to_internal_dt(
    t_in: np.ndarray,
    a_in: np.ndarray,
    *,
    dt_internal_s: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if dt_internal_s <= 0.0:
        raise ValueError('solver.dt_internal_s must be > 0.')

    n = int(round(duration_s / dt_internal_s)) + 1
    t = dt_internal_s * np.arange(n, dtype=float)

    # Linear interpolation; outside range use endpoints (input is already padded in process_input)
    a = np.interp(t, t_in, a_in)
    return t, a


def run_simulate_drop(
    echo=print,
    buttock_override: dict | None = None,
    output_filename: str | None = None,
    subfolder: str | None = None,
    output_subfolder: str | None = None,
) -> list[dict]:
    config = read_config()

    # Apply buttock overrides if provided
    if buttock_override:
        if 'buttock' not in config:
            config['buttock'] = {}
        config['buttock'].update(buttock_override)

    # Internal solver settings
    dt_internal_s = float(req_float(config, ['solver', 'dt_internal_s']))
    max_newton_iter = int(req_int(config, ['solver', 'max_newton_iter']))
    newton_tol = float(req_float(config, ['solver', 'newton_tol']))

    masses_path = resolve_path(req_str(config, ['model', 'masses_json']))
    masses = load_json(masses_path)

    mass_map = build_mass_map(
        masses,
        arm_recruitment=float(req_float(config, ['model', 'arm_recruitment'])),
        helmet_mass=float(req_float(config, ['model', 'helmet_mass_kg'])),
        echo=echo,
    )

    model = build_spine_model(mass_map, config)

    # Debug buttocks bottom-out point (computed from force threshold)
    x0_m = model.buttocks_bottom_out_compression_m()
    x0_mm = x0_m * 1000.0
    echo('Buttocks bilinear model:')
    echo(f'  k1 = {model.buttocks_k1_n_per_m:.3g} N/m')
    echo(f'  k2 = {model.buttocks_k2_n_per_m:.3g} N/m')
    echo(
        f'  c  = {model.buttocks_c_ns_per_m:.3g} Ns/m (contact-only, both signs, clamped nonnegative)'
    )
    echo(f'  bottom_out_force = {model.buttocks_bottom_out_force_n / 1000.0:.3f} kN')
    echo(f'  implied bottom_out_compression = {x0_mm:.3f} mm')

    # Spine config debug
    echo('Spine model:')
    echo(f'  disc_height = {model.disc_height_m * 1000.0:.3f} mm (uniform)')
    echo(
        f'  damping = {float(req_float(config, ["spine", "damping_ns_per_m"])):.1f} Ns/m (all IVDs)'
    )
    echo(
        f'  tension_k_mult = {model.tension_k_mult:.3f} (tension stiffness is constant, not Kemper-scaled)'
    )
    echo('Kemper rate model:')
    echo(f'  normalize_to_eps = {model.kemper_normalize_to_eps_per_s:.3f} 1/s')
    echo(f'  smoothing_tau = {model.strain_rate_smoothing_tau_s * 1000.0:.3f} ms')
    echo(f'  warn_over_eps = {model.warn_over_eps_per_s:.3f} 1/s')
    echo('Solver:')
    echo(f'  dt_internal = {dt_internal_s * 1000.0:.3f} ms ({1.0 / dt_internal_s:.1f} Hz)')
    echo(f'  Newmark/Newton: max_iter={max_newton_iter}, tol={newton_tol:g}')

    buttocks_height_mm, heights_from_model = _get_plotting_config(config)

    if subfolder:
        inputs_dir = INPUT_BASE / subfolder
        out_dir = OUTPUT_BASE / subfolder
    else:
        inputs_dir = INPUT_BASE
        out_dir = OUTPUT_BASE

    # In batch mode, put outputs in a subfolder named after the output file
    if output_subfolder:
        out_dir = out_dir / output_subfolder

    pattern = INPUT_PATTERN

    # Determine config filename (matches output CSV name)
    config_basename = (output_filename.replace('.csv', '.json') if output_filename else 'config.json')
    out_config_path = out_dir / config_basename

    # Check if output already exists with same config
    if out_config_path.exists():
        with open(out_config_path, encoding='utf-8') as f:
            existing_config = json.load(f)
        if existing_config == config:
            echo(f'Skipping {subfolder or "output"}/{config_basename}: config unchanged')
            return []
        # Config changed - remove old output files for this specific config
        echo(f'Config changed for {config_basename}, removing old outputs')
        csv_path = out_dir / (output_filename or 'summary.csv')
        if csv_path.exists():
            csv_path.unlink()
        out_config_path.unlink()

    files = sorted(inputs_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No drop inputs found: {inputs_dir}/{pattern}')

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output folder
    with open(out_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    summary: list[dict] = []

    settle_ms = float(req_float(config, ['drop', 'gravity_settle_ms']))
    sim_duration_ms = float(req_float(config, ['drop', 'sim_duration_ms']))
    duration_s = sim_duration_ms / 1000.0

    t12_elem_idx = model.element_names.index('T12-L1')
    butt_elem_idx = model.element_names.index('buttocks')
    head_idx = model.node_names.index('HEAD')
    pelvis_idx = model.node_names.index('pelvis')

    warn_threshold = float(model.warn_over_eps_per_s)

    for fpath in files:
        echo(f'\nProcessing {fpath.name}...')

        t_in, a_in_g, info = process_input(
            fpath,
            cfc=float(req_float(config, ['drop', 'cfc'])),
            sim_duration_ms=sim_duration_ms,
            style_threshold_ms=float(req_float(config, ['drop', 'style_duration_threshold_ms'])),
            peak_threshold_g=float(req_float(config, ['drop', 'peak_threshold_g'])),
            freefall_threshold_g=float(req_float(config, ['drop', 'freefall_threshold_g'])),
        )

        t, a_g = _interpolate_to_internal_dt(
            np.asarray(t_in, dtype=float),
            np.asarray(a_in_g, dtype=float),
            dt_internal_s=dt_internal_s,
            duration_s=duration_s,
        )

        y0 = np.zeros(model.size(), dtype=float)
        v0 = np.zeros(model.size(), dtype=float)

        run_dir = out_dir / fpath.stem
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        # Flat-style: gravity settle at internal dt
        if info['style'] == 'flat' and settle_ms > 0.0:
            t_settle = np.arange(
                0.0, settle_ms / 1000.0 + dt_internal_s, dt_internal_s, dtype=float
            )
            a_settle = np.zeros_like(t_settle)

            # Start from static equilibrium guess (optional but helps)
            y_stat, v_stat = initial_state_static(model, base_accel_g0=0.0)
            y0 = y_stat
            v0 = v_stat

            sim_settle = newmark_nonlinear(
                model,
                t_settle,
                a_settle,
                y0,
                v0,
                max_newton_iter=max_newton_iter,
                newton_tol=newton_tol,
            )

            plot_gravity_settling(
                sim_settle.time_s,
                sim_settle.y,
                model.node_names,
                model.element_names,
                run_dir / 'gravity_settling.png',
                heights_from_model=heights_from_model,
                buttocks_height_mm=buttocks_height_mm,
            )

            y0, v0 = sim_settle.y[-1].copy(), sim_settle.v[-1].copy()

        sim = newmark_nonlinear(
            model,
            t,
            a_g,
            y0,
            v0,
            max_newton_iter=max_newton_iter,
            newton_tol=newton_tol,
        )

        forces = sim.element_forces_n
        f_t12 = forces[:, t12_elem_idx]
        f_butt = forces[:, butt_elem_idx]

        peak_base_g = float(np.max(a_g))
        min_base_g = float(np.min(a_g))
        peak_t12_kN = float(np.max(f_t12) / 1000.0)
        peak_butt_kN = float(np.max(f_butt) / 1000.0)
        t_peak_ms = float(sim.time_s[np.argmax(f_t12)] * 1000.0)

        max_head_compression_mm = -float(np.min(sim.y[:, head_idx]) * 1000.0)
        max_pelvis_compression_mm = -float(np.min(sim.y[:, pelvis_idx]) * 1000.0)
        max_spine_shortening_mm = max_head_compression_mm - max_pelvis_compression_mm

        # Buttocks bottom-out metrics (computed from force threshold -> implied displacement)
        x0_mm = model.buttocks_bottom_out_compression_m() * 1000.0
        y_pelvis_m = sim.y[:, pelvis_idx]
        butt_comp_m = np.maximum(-y_pelvis_m, 0.0)
        butt_comp_max_mm = float(np.max(butt_comp_m) * 1000.0)
        butt_bottomed_out = bool(
            (model.buttocks_bottom_out_force_n > 0.0) and (butt_comp_max_mm > x0_mm)
        )

        # Strain-rate warnings summary
        eps_smooth_max = float(np.max(sim.strain_rate_per_s))
        eps_over = sim.strain_rate_per_s > warn_threshold
        frac_over = float(np.mean(eps_over)) if eps_over.size else 0.0

        if eps_smooth_max > warn_threshold:
            per_elem_max = np.max(sim.strain_rate_per_s, axis=0)
            offenders = [
                (model.element_names[i], float(per_elem_max[i]))
                for i in range(len(model.element_names))
                if per_elem_max[i] > warn_threshold
            ]
            offenders.sort(key=lambda x: x[1], reverse=True)
            echo(f'WARNING: strain rate exceeded {warn_threshold:.1f} 1/s.')
            echo(f'  max_eps_any = {eps_smooth_max:.2f} 1/s')
            echo(f'  fraction_of_samples_over = {frac_over * 100.0:.3f}%')
            echo('  offenders (top):')
            for name, mx in offenders[:10]:
                echo(f'    {name}: {mx:.2f} 1/s')

        write_timeseries_csv(
            run_dir / 'timeseries.csv',
            sim.time_s,
            sim.base_accel_g,
            model.node_names,
            model.element_names,
            sim.y,
            sim.v,
            sim.a,
            forces,
            strain_rate_per_s=sim.strain_rate_per_s,
            k_dynamic_n_per_m=sim.k_dynamic_n_per_m,
        )

        plot_displacements(
            sim.time_s,
            sim.y,
            a_g,
            model.node_names,
            model.element_names,
            run_dir / 'displacements.png',
            heights_from_model=heights_from_model,
            buttocks_height_mm=buttocks_height_mm,
            plot_duration_ms=sim_duration_ms,
            reference_frame='base',
        )
        plot_forces(
            sim.time_s,
            forces,
            a_g,
            model.element_names,
            run_dir / 'forces.png',
            highlight='T12-L1',
            plot_duration_ms=sim_duration_ms,
        )
        plot_displacement_colored_by_force(
            sim.time_s,
            sim.y,
            forces,
            a_g,
            model.node_names,
            model.element_names,
            run_dir / 'mixed.png',
            heights_from_model=heights_from_model,
            buttocks_height_mm=buttocks_height_mm,
            plot_duration_ms=sim_duration_ms,
            reference_frame='base',
        )

        echo(f'  Style: {info["style"]}')
        echo(f'  Input sample rate (post-resample): {info["sample_rate_hz"]:.1f} Hz')
        echo(f'  Internal solver rate: {1.0 / dt_internal_s:.1f} Hz')
        echo(f'  Base accel: peak={peak_base_g:.2f} g, min={min_base_g:.2f} g')
        echo(f'  Peak buttocks: {peak_butt_kN:.2f} kN')
        echo(f'  Peak T12-L1: {peak_t12_kN:.2f} kN @ {t_peak_ms:.1f} ms')
        echo(f'  Spine shortening: {max_spine_shortening_mm:.1f} mm')
        echo(f'  Buttocks implied bottom-out compression: {x0_mm:.2f} mm')
        echo(
            f'  Buttocks max compression: {butt_comp_max_mm:.2f} mm (bottomed_out={butt_bottomed_out})'
        )
        echo(f'  max_eps_any: {eps_smooth_max:.2f} 1/s')

        summary.append(
            {
                'filename': fpath.name,
                'base_accel_peak_g': peak_base_g,
                'peak_T12L1_kN': peak_t12_kN,
                'time_to_peak_T12L1_ms': t_peak_ms,
                'peak_buttocks_kN': peak_butt_kN,
                'buttocks_k1_n_per_m': model.buttocks_k1_n_per_m,
                'buttocks_bottom_out_force_kN': model.buttocks_bottom_out_force_n / 1000.0,
                'buttocks_bottom_out_limit_mm': round(x0_mm, 3),
                'buttocks_max_compression_reached_mm': butt_comp_max_mm,
                'buttocks_bottomed_out': butt_bottomed_out,
            }
        )

    summary_filename = output_filename or 'summary.csv'
    summary_path = out_dir / summary_filename
    fieldnames = [
        'filename',
        'base_accel_peak_g',
        'peak_T12L1_kN',
        'time_to_peak_T12L1_ms',
        'peak_buttocks_kN',
        'buttocks_k1_n_per_m',
        'buttocks_bottom_out_force_kN',
        'buttocks_bottom_out_limit_mm',
        'buttocks_max_compression_reached_mm',
        'buttocks_bottomed_out',
    ]
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    echo(f'\nResults written to {summary_path}')
    return summary
