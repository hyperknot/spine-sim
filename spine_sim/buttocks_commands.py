"""Buttocks calibration and simulation commands."""

from __future__ import annotations

import json

from spine_sim.buttocks import generate_buttocks_plot
from spine_sim.config import read_config, resolve_path
from spine_sim.toen_drop import (
    TOEN_IMPACT_V_MPS,
    calibrate_toen_buttocks_model,
    run_toen_suite,
)
from spine_sim.toen_store import load_toen_drop_calibration, write_toen_drop_calibration


def run_calibrate_buttocks(echo=print) -> dict:
    """Calibrate buttocks model from Toen 2012 paper data."""
    config = read_config()
    bcfg = config.get("buttock", {})

    subject = str(bcfg.get("subject", "avg")).lower()
    if subject == "both":
        subject = "avg"

    target_set = str(bcfg.get("target_set", "avg")).lower()
    if target_set == "fig3":
        target_set = "subj3"

    dens = bcfg.get("densification", {}).get("buttocks", {})

    echo(f"Calibrating buttocks model: subject={subject}, target_set={target_set}")
    result = calibrate_toen_buttocks_model(
        subject_id=subject,
        target_set=target_set,
        male50_mass_kg=float(bcfg.get("male50_mass_kg", 75.4)),
        buttocks_stop_k_n_per_m=float(dens.get("stop_k_n_per_m", 5.0e6)),
        buttocks_stop_smoothing_mm=float(dens.get("smoothing_mm", 1.0)),
    )

    out_path = write_toen_drop_calibration(result, active=True)
    echo(f"Calibration saved: {out_path}")
    echo(json.dumps(result, indent=2))
    return result


def run_simulate_buttocks(echo=print) -> list[dict]:
    """Simulate Toen drop suite and generate plots."""
    config = read_config()
    bcfg = config.get("buttock", {})

    subject = str(bcfg.get("subject", "avg")).lower()
    target_set = str(bcfg.get("target_set", "avg")).lower()
    if target_set == "fig3":
        target_set = "subj3"

    velocities = [float(v) for v in bcfg.get("velocities_mps", [TOEN_IMPACT_V_MPS])]
    male50 = float(bcfg.get("male50_mass_kg", 75.4))

    solver = bcfg.get("solver", {})
    dt_s = float(solver.get("dt_s", 0.0005))
    duration_s = float(solver.get("duration_s", 0.15))
    max_newton_iter = int(solver.get("max_newton_iter", 10))

    dens = bcfg.get("densification", {}).get("buttocks", {})
    butt_limit_mm = float(dens.get("limit_mm", 39.0))
    butt_stop_k = float(dens.get("stop_k_n_per_m", 5.0e6))
    butt_smooth_mm = float(dens.get("smoothing_mm", 1.0))

    # Load saved calibration if available
    butt_k, butt_c = None, None
    if bcfg.get("use_saved_calibration", True):
        doc, path = load_toen_drop_calibration()
        if doc is not None:
            r = doc.get("result", {})
            saved_target = str(r.get("target_set", "")).lower()
            if saved_target == target_set:
                butt_k = r.get("buttocks_k_n_per_m")
                butt_c = r.get("buttocks_c_ns_per_m")
                butt_limit_mm = float(r.get("buttocks_limit_mm", butt_limit_mm))
                butt_stop_k = float(r.get("buttocks_stop_k_n_per_m", butt_stop_k))
                butt_smooth_mm = float(r.get("buttocks_stop_smoothing_mm", butt_smooth_mm))
                echo(f"Loaded calibration from {path}")

    subjects = ["avg", "3"] if subject == "both" else [subject]
    out_dir = resolve_path(str(bcfg.get("output_dir", "output/toen_drop")))
    out_dir.mkdir(parents=True, exist_ok=True)

    echo(f"Simulating: target_set={target_set}, subjects={subjects}, velocities={velocities}")

    all_results = []
    for sid in subjects:
        results = run_toen_suite(
            subject_id=sid, target_set=target_set, male50_mass_kg=male50,
            impact_velocities_mps=velocities, dt_s=dt_s, duration_s=duration_s,
            max_newton_iter=max_newton_iter, buttocks_k_n_per_m=butt_k,
            buttocks_c_ns_per_m=butt_c, buttocks_limit_mm=butt_limit_mm,
            buttocks_stop_k_n_per_m=butt_stop_k, buttocks_stop_smoothing_mm=butt_smooth_mm,
        )
        all_results.extend([r.__dict__ for r in results])

    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")

    # Generate plots for each velocity
    for v_plot in velocities:
        plot_path = generate_buttocks_plot(
            config, out_dir, v_plot, subject if subject != "both" else "avg",
            target_set, male50, butt_k, butt_c, butt_limit_mm, butt_stop_k, butt_smooth_mm,
            dt_s, duration_s, max_newton_iter,
        )
        echo(f"  Plot: {plot_path}")

    echo(f"Simulation and plots complete. Output: {out_dir}")
    return all_results
