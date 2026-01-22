#!/usr/bin/env -S uv run

import argparse

from spine_sim.drop_commands import run_simulate_drop


# Batch mode configurations
BATCH_STIFFNESS_DAMPING = [
    (305000, 5250, '305'),
    (85200, 1750, '85'),
    (180500, 3130, '180'),
]

BATCH_BOTTOM_OUT = [
    (7.0, '7'),
    (9999.0, 'unlimited'),
]


def main() -> None:
    parser = argparse.ArgumentParser(description='Run spine drop simulation')
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run batch mode with all buttock parameter combinations',
    )
    args = parser.parse_args()

    if args.batch:
        for k1, c, stiff_label in BATCH_STIFFNESS_DAMPING:
            for force, force_label in BATCH_BOTTOM_OUT:
                output_filename = f'{stiff_label}-{force_label}.json'
                print(f'\n{"=" * 60}')
                print(f'BATCH: k1={k1}, c={c}, bottom_out={force} -> {output_filename}')
                print(f'{"=" * 60}')
                run_simulate_drop(
                    echo=print,
                    buttock_override={
                        'k1_n_per_m': k1,
                        'c_ns_per_m': c,
                        'bottom_out_force_kN': force,
                    },
                    output_filename=output_filename,
                )
    else:
        run_simulate_drop(echo=print)


if __name__ == '__main__':
    main()
