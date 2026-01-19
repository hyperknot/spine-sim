#!/usr/bin/env -S uv run

"""
Spine simulation CLI.

Pipeline:
1. calibrate-buttocks: Calibrate buttocks model from Toen 2012 paper data
2. simulate-buttocks: Simulate Toen drop suite and plot results
3. calibrate-drop: Calibrate spine model against Yoganandan data (uses Toen buttocks)
4. simulate-drop: Run drop simulations on acceleration data
"""

import click
from spine_sim.simulation import (
    run_calibrate_buttocks,
    run_calibrate_drop,
    run_simulate_buttocks,
    run_simulate_drop,
)


@click.group()
def cli():
    """Spine-sim: 1D axial spine impact simulation."""
    pass


@cli.command('calibrate-buttocks')
def calibrate_buttocks():
    """Calibrate buttocks model from Toen 2012 paper data."""
    run_calibrate_buttocks(echo=click.echo)


@cli.command('simulate-buttocks')
def simulate_buttocks():
    """Simulate Toen drop suite and generate plots."""
    run_simulate_buttocks(echo=click.echo)


@cli.command('calibrate-drop-peaks')
def calibrate_drop_peaks():
    """Calibrate spine stiffness scales to Yoganandan peak forces."""
    run_calibrate_drop(echo=click.echo, mode='peaks')


@cli.command('calibrate-drop-curves')
def calibrate_drop_curves():
    """Calibrate spine model to Yoganandan force-time curves."""
    run_calibrate_drop(echo=click.echo, mode='curves')


@cli.command('simulate-drop')
def simulate_drop():
    """Run drop simulations on acceleration data."""
    run_simulate_drop(echo=click.echo)


if __name__ == '__main__':
    cli()
