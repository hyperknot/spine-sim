#!/usr/bin/env -S uv run

from spine_sim.drop_commands import run_simulate_drop


def main() -> None:
    run_simulate_drop(echo=print)


if __name__ == '__main__':
    main()
