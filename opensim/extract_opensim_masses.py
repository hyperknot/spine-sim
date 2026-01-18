import argparse
import json
from pathlib import Path

import opensim as osim


def extract_model_data(model_path: Path, geom_dir: Path | None) -> dict:
    """Extract body masses and vertical positions from OpenSim model.

    Heights are extracted as Y-coordinates (vertical in OpenSim convention)
    relative to the pelvis body frame origin, converted to mm.
    """
    if geom_dir:
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geom_dir))

    model = osim.Model(str(model_path))
    state = model.initSystem()

    # Realize to position stage to get body positions
    model.realizePosition(state)

    bodies = model.getBodySet()
    masses = {}
    positions_y_m = {}  # Y position in ground frame (meters)
    total = 0.0

    # First pass: collect all positions
    for i in range(bodies.getSize()):
        body = bodies.get(i)
        name = body.getName()
        m = float(body.getMass())
        masses[name] = m
        total += m

        # Get position of body frame origin in ground frame
        pos = body.findStationLocationInGround(state, osim.Vec3(0, 0, 0))
        # OpenSim convention: Y is vertical (up positive)
        positions_y_m[name] = float(pos[1])

    # Find pelvis position as reference
    pelvis_y = positions_y_m.get('pelvis', 0.0)

    # Compute heights relative to pelvis, in mm
    heights_mm = {}
    for name, y in positions_y_m.items():
        heights_mm[name] = (y - pelvis_y) * 1000.0

    return {
        'model_name': model.getName(),
        'model_path': str(model_path),
        'total_mass_kg': total,
        'bodies': masses,
        'heights_relative_to_pelvis_mm': heights_mm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract body masses and heights from OpenSim model to JSON.'
    )
    parser.add_argument('--model', required=True, type=Path, help='Path to .osim model')
    parser.add_argument('--geom', type=Path, default=None, help='Geometry directory (optional)')
    parser.add_argument('--out', required=True, type=Path, help='Output JSON path')
    args = parser.parse_args()

    data = extract_model_data(args.model, args.geom)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f'Wrote: {args.out}')
    print(f'Total mass: {data["total_mass_kg"]:.3f} kg')
    print(f'Body count: {len(data["bodies"])}')

    # Show spine heights for verification
    h = data['heights_relative_to_pelvis_mm']
    spine_bodies = [
        'pelvis',
        'lumbar5',
        'lumbar4',
        'lumbar3',
        'lumbar2',
        'lumbar1',
        'thoracic12',
        'thoracic11',
        'thoracic10',
        'thoracic9',
        'thoracic8',
        'thoracic7',
        'thoracic6',
        'thoracic5',
        'thoracic4',
        'thoracic3',
        'thoracic2',
        'thoracic1',
        'head_neck',
    ]
    print('\nSpine heights relative to pelvis:')
    for name in spine_bodies:
        if name in h:
            print(f'  {name}: {h[name]:.1f} mm')


if __name__ == '__main__':
    main()
