import argparse
import json
from pathlib import Path

import opensim as osim


def extract_masses(model_path: Path, geom_dir: Path | None) -> dict:
    if geom_dir:
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geom_dir))

    model = osim.Model(str(model_path))

    bodies = model.getBodySet()
    masses = {}
    total = 0.0

    for i in range(bodies.getSize()):
        body = bodies.get(i)
        m = float(body.getMass())
        name = body.getName()
        masses[name] = m
        total += m

    return {
        'model_name': model.getName(),
        'model_path': str(model_path),
        'total_mass_kg': total,
        'bodies': masses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract body masses from OpenSim model to JSON.')
    parser.add_argument('--model', required=True, type=Path, help='Path to .osim model')
    parser.add_argument('--geom', type=Path, default=None, help='Geometry directory (optional)')
    parser.add_argument('--out', required=True, type=Path, help='Output JSON path')
    args = parser.parse_args()

    data = extract_masses(args.model, args.geom)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f'Wrote: {args.out}')
    print(f'Total mass: {data["total_mass_kg"]:.3f} kg')
    print(f'Body count: {len(data["bodies"])}')


if __name__ == '__main__':
    main()
