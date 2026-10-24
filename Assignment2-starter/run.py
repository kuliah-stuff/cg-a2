"""Command-line runner for Assignment 2.

Examples (run from the starter directory):
    python run.py meshes/cube.obj --check-input
    python run.py meshes/cube.obj --summary
    python run.py meshes/cube.obj --subdivide catmull-clark --levels 2 --png cube.png
    python run.py meshes/fan.obj --flip 3 --png fan.png
"""

import argparse
import json
import time
from pathlib import Path

import student
from halfedge import MeshError
from support import load_obj, render_png, save_obj

OPERATIONS = {
    'linear': student.subdivide_linear,
    'catmull-clark': student.subdivide_catmull_clark,
}


def main():
    parser = argparse.ArgumentParser(description='Assignment 2 mesh runner')
    parser.add_argument('mesh', type=Path, help='input OBJ file')
    parser.add_argument('--check-input', action='store_true',
                        help='load the mesh, check its connectivity, and stop')
    parser.add_argument('--flip', type=int, nargs='+', default=[], metavar='EDGE',
                        help='flip these edge indices, in the given order, before subdividing')
    parser.add_argument('--subdivide', choices=sorted(OPERATIONS))
    parser.add_argument('--levels', type=int, default=1, help='number of subdivision steps')
    parser.add_argument('--summary', action='store_true', help='print mesh_summary as JSON')
    parser.add_argument('--obj', type=Path, help='write the resulting mesh to this OBJ file')
    parser.add_argument('--png', type=Path, help='render the resulting mesh to this PNG file')
    parser.add_argument('--azimuth', type=float, default=35.0)
    parser.add_argument('--elevation', type=float, default=25.0)
    args = parser.parse_args()

    try:
        mesh = load_obj(args.mesh)
        mesh.validate()
    except MeshError as error:
        print(f'Invalid input mesh: {error}')
        raise SystemExit(1)
    if args.check_input:
        print(f'Loaded {len(mesh.vertices)} vertices, {len(mesh.edges)} edges, '
              f'{len(mesh.faces)} faces, {len(mesh.boundary_faces)} boundary loops')
        return

    try:
        for index in args.flip:
            student.flip_edge(mesh, mesh.edges[index])
            mesh.validate()
        for level in range(args.levels if args.subdivide else 0):
            start = time.perf_counter()
            mesh = OPERATIONS[args.subdivide](mesh)
            mesh.validate()
            print(f'{args.subdivide} level {level + 1}: {len(mesh.faces)} faces '
                  f'in {time.perf_counter() - start:.3f} s')
        if args.summary:
            print(json.dumps(student.mesh_summary(mesh), indent=2))
        if args.obj:
            save_obj(args.obj, mesh, student.face_vertices)
            print(args.obj)
        if args.png:
            render_png(args.png, mesh, student.face_vertices, _optional(student.face_normal, mesh),
                       azimuth=args.azimuth, elevation=args.elevation,
                       highlight=() if args.subdivide else args.flip)
            print(args.png)
    except NotImplementedError as error:
        print(f'Incomplete student implementation: {error}. Nothing further was written.')
    except MeshError as error:
        print(f'The resulting mesh is invalid: {error}')
        raise SystemExit(1)
    except ValueError as error:
        print(f'Operation rejected: {error}')
        raise SystemExit(2)


def _optional(function, mesh):
    """Use the student's function for shading only after it is implemented."""
    try:
        function(mesh.faces[0])
    except NotImplementedError:
        return None
    except (IndexError, AttributeError):
        pass
    return function


if __name__ == '__main__':
    main()
