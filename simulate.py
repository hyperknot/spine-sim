#!/usr/bin/env -S uv run

import argparse
import subprocess

from spine_sim.drop_commands import run_simulate_drop
from spine_sim.settings import read_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Run spine drop simulation')
    parser.add_argument(
        'subfolder',
        help='Subfolder in input/ to process (outputs to output/<subfolder>)',
    )
    parser.add_argument(
        '--mode',
        choices=['localized', 'uniform'],
        help='Buttocks mode for a single run (required unless --batch).',
    )
    parser.add_argument(
        '--profile',
        choices=['sporty', 'avg', 'soft'],
        help='Buttocks profile for a single run (required unless --batch).',
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run batch mode (always runs BOTH modes and ALL profiles).',
    )
    args = parser.parse_args()

    config = read_config()

    if args.batch:
        profiles = list(config['buttock']['profiles'].keys())

        preferred = ['sporty', 'avg', 'soft']
        if all(p in profiles for p in preferred):
            profiles = preferred
        else:
            profiles = sorted(profiles)

        modes = ['localized', 'uniform']

        for profile in profiles:
            for mode in modes:
                mode_label = 'loc' if mode == 'localized' else 'uni'
                batch_name = f'{profile}-{mode_label}'
                output_filename = f'{batch_name}.csv'
                print(f'\n{"=" * 60}')
                print(f'BATCH: profile={profile}, mode={mode} -> {batch_name}/')
                print(f'{"=" * 60}')
                run_simulate_drop(
                    echo=print,
                    subfolder=args.subfolder,
                    output_subfolder=batch_name,
                    output_filename=output_filename,
                    buttocks_mode=mode,
                    buttocks_profile=profile,
                )
    else:
        if not args.mode:
            raise SystemExit('--mode is required unless --batch.')
        if not args.profile:
            raise SystemExit('--profile is required unless --batch.')

        run_simulate_drop(
            echo=print,
            subfolder=args.subfolder,
            buttocks_mode=args.mode,
            buttocks_profile=args.profile,
        )

    subprocess.run(['./visualize_tables.py', args.subfolder])


if __name__ == '__main__':
    main()
