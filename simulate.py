#!/usr/bin/env -S uv run

"""
Spine simulation CLI (simplified).

Pipeline:
- simulate-drop: Run drop simulations on acceleration data.
"""

import click
from spine_sim.drop_commands import run_simulate_drop


@click.group()
def cli():
    """Spine-sim: 1D axial spine impact simulation."""
    pass


@cli.command('simulate-drop')
def simulate_drop():
    """Run drop simulations on acceleration data."""
    run_simulate_drop(echo=click.echo)


if __name__ == '__main__':
    cli()
