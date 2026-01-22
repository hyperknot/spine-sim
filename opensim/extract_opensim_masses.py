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


def main(model: Path, geom: Path | None, out: Path) -> None:
    data = extract_model_data(model, geom)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f'Wrote: {out}')
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


def debug_find_disks(model_path: Path, geom_dir: Path | None) -> None:
    """Debug: Search for intervertebral disks in the model."""
    if geom_dir:
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(geom_dir))

    model = osim.Model(str(model_path))
    state = model.initSystem()

    print('\n' + '=' * 60)
    print('DEBUG: Searching for intervertebral disks...')
    print('=' * 60)

    # Search in bodies
    bodies = model.getBodySet()
    print(f'\n--- Bodies ({bodies.getSize()} total) ---')
    disk_bodies = []
    for i in range(bodies.getSize()):
        name = bodies.get(i).getName()
        name_lower = name.lower()
        if 'disk' in name_lower or 'disc' in name_lower or 'ivd' in name_lower:
            disk_bodies.append(name)
        # Also print all body names to see what's available
    print('All body names:')
    for i in range(bodies.getSize()):
        print(f'  {bodies.get(i).getName()}')
    if disk_bodies:
        print(f'\nFound disk-related bodies: {disk_bodies}')
    else:
        print('\nNo disk-related bodies found.')

    # Search in joints
    joints = model.getJointSet()
    print(f'\n--- Joints ({joints.getSize()} total) ---')
    disk_joints = []
    for i in range(joints.getSize()):
        name = joints.get(i).getName()
        joint_type = type(joints.get(i)).__name__
        name_lower = name.lower()
        if 'disk' in name_lower or 'disc' in name_lower or 'ivd' in name_lower:
            disk_joints.append((name, joint_type))
    print('All joint names:')
    for i in range(joints.getSize()):
        j = joints.get(i)
        print(f'  {j.getName()} ({type(j).__name__})')
    if disk_joints:
        print(f'\nFound disk-related joints: {disk_joints}')
    else:
        print('\nNo disk-related joints found.')

    # Search in forces (disks might be modeled as BushingForce or similar)
    forces = model.getForceSet()
    print(f'\n--- Forces ({forces.getSize()} total) ---')
    disk_forces = []
    for i in range(forces.getSize()):
        name = forces.get(i).getName()
        force_type = type(forces.get(i)).__name__
        name_lower = name.lower()
        if (
            'disk' in name_lower
            or 'disc' in name_lower
            or 'ivd' in name_lower
            or 'bushing' in name_lower
        ):
            disk_forces.append((name, force_type))
    print('All force names:')
    for i in range(forces.getSize()):
        f = forces.get(i)
        print(f'  {f.getName()} ({type(f).__name__})')
    if disk_forces:
        print(f'\nFound disk-related forces: {disk_forces}')
    else:
        print('\nNo disk-related forces found.')

    # Search in constraints
    constraints = model.getConstraintSet()
    print(f'\n--- Constraints ({constraints.getSize()} total) ---')
    if constraints.getSize() > 0:
        print('All constraint names:')
        for i in range(constraints.getSize()):
            c = constraints.get(i)
            print(f'  {c.getName()} ({type(c).__name__})')
    else:
        print('No constraints in model.')

    # Detailed inspection of IVD joints
    print('\n--- IVD Joint Details ---')
    for i in range(joints.getSize()):
        j = joints.get(i)
        name = j.getName()
        if 'IVD' in name:
            print(f'\nJoint: {name}')
            print(f'  Type: {type(j).__name__}')
            print(f'  Concrete type: {j.getConcreteClassName()}')

            # Get parent and child frames
            try:
                parent = j.getParentFrame().getName()
                child = j.getChildFrame().getName()
                print(f'  Parent frame: {parent}')
                print(f'  Child frame: {child}')
            except:
                pass

            # Cast to CustomJoint and get coordinates
            try:
                cj = osim.CustomJoint.safeDownCast(j)
                if cj:
                    # Get SpatialTransform
                    st = cj.getSpatialTransform()
                    print('  SpatialTransform axes:')
                    for axis_idx in range(6):
                        axis = st.getTransformAxis(axis_idx)
                        coord_names = []
                        for ci in range(axis.getCoordinateNamesInArray().getSize()):
                            coord_names.append(axis.getCoordinateNamesInArray().get(ci))
                        func = axis.getFunction()
                        print(
                            f'    Axis {axis_idx}: coords={coord_names}, func={func.getConcreteClassName()}'
                        )

                    # Get coordinates from the joint
                    num_coords = cj.numCoordinates()
                    print(f'  Coordinates ({num_coords}):')
                    for ci in range(num_coords):
                        coord = cj.get_coordinates(ci)
                        print(f'    - {coord.getName()}:')
                        print(
                            f'        default_value: {coord.getDefaultValue():.6f} rad ({coord.getDefaultValue() * 180 / 3.14159:.2f} deg)'
                        )
                        print(
                            f'        range: [{coord.getRangeMin():.4f}, {coord.getRangeMax():.4f}] rad'
                        )
                        print(
                            f'        clamped: {coord.get_clamped()}, locked: {coord.get_locked()}'
                        )
            except Exception as e:
                print(f'  Error accessing CustomJoint: {e}')

    # Check if there are any BushingForce elements (would represent disk stiffness)
    print('\n--- Searching for BushingForce (disk stiffness) ---')
    bushing_count = 0
    for i in range(forces.getSize()):
        f = forces.get(i)
        fname = f.getName()
        ftype = f.getConcreteClassName()
        if 'Bushing' in ftype or 'bushing' in fname.lower():
            bushing_count += 1
            print(f'\nBushingForce: {fname}')
            print(f'  Type: {ftype}')
            try:
                # Try to get stiffness properties
                props = f.getPropertyNames()
                for prop in props:
                    if 'stiffness' in prop.lower() or 'damping' in prop.lower():
                        print(f'  {prop}: {f.getPropertyByName(prop).toString()}')
            except Exception as e:
                print(f'  Error: {e}')

    if bushing_count == 0:
        print(
            'No BushingForce elements found - IVD joints likely have NO passive stiffness modeled.'
        )
        print('(Per paper: joints are pure ball joints, stiffness comes from muscles only)')

    print('\n' + '=' * 60)
    print('DEBUG: End of disk search')
    print('=' * 60 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract body masses and heights from OpenSim model to JSON.'
    )
    parser.add_argument('--model', required=True, type=Path, help='Path to .osim model')
    parser.add_argument('--geom', type=Path, default=None, help='Geometry directory (optional)')
    parser.add_argument('--out', required=True, type=Path, help='Output JSON path')
    parser.add_argument(
        '--debug-disks', action='store_true', help='Debug: search for intervertebral disks'
    )
    args = parser.parse_args()

    if args.debug_disks:
        debug_find_disks(args.model, args.geom)

    main(args.model, args.geom, args.out)
