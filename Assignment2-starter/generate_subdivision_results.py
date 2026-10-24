"""Generate OBJ meshes and summaries for bunny and torus, levels 1-4.

Run from Assignment2-starter: python generate_subdivision_results.py
The bunny has boundary loops, so this assignment's Catmull-Clark rule rejects it.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import student
from support import _camera, load_obj, render_png, save_obj


ROOT = Path(__file__).parent
OUT = ROOT / 'subdivision_results'


def read_obj_arrays(path):
    positions, polygons = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split('#', 1)[0].split()
        if not parts:
            continue
        if parts[0] == 'v':
            positions.append(tuple(map(float, parts[1:4])))
        elif parts[0] == 'f':
            polygons.append([int(token.split('/')[0]) - 1 for token in parts[1:]])
    sides = {len(p) for p in polygons}
    if len(sides) != 1:
        raise ValueError(f'Expected uniform polygon size in {path}')
    return np.asarray(positions, dtype=np.float64), np.asarray(polygons, dtype=np.int32)


def linear_arrays(positions, polygons):
    """Exactly the Task 5 vertex and face ordering, using compact arrays."""
    V, F = len(positions), len(polygons)
    edge_ids = {}
    for polygon in polygons:
        for previous, current in zip(polygon[-1:].tolist() + polygon[:-1].tolist(),
                                     polygon.tolist()):
            key = (min(previous, current), max(previous, current))
            if key not in edge_ids:
                edge_ids[key] = len(edge_ids)
    E = len(edge_ids)
    edge_points = np.empty((E, 3), dtype=np.float64)
    for (a, b), index in edge_ids.items():
        edge_points[index] = (positions[a] + positions[b]) / 2
    face_points = positions[polygons].mean(axis=1)
    refined = np.concatenate((positions, edge_points, face_points))

    quads = np.empty((polygons.size, 4), dtype=np.int32)
    row = 0
    for face_index, polygon in enumerate(polygons):
        n = len(polygon)
        for i, current in enumerate(polygon):
            previous, following = polygon[i - 1], polygon[(i + 1) % n]
            ahead = edge_ids[(min(current, following), max(current, following))]
            behind = edge_ids[(min(previous, current), max(previous, current))]
            quads[row] = (current, V + ahead, V + E + face_index, V + behind)
            row += 1
    return refined, quads, E


def write_obj_arrays(path, positions, polygons):
    with path.open('w', encoding='utf-8', buffering=1024 * 1024) as stream:
        for x, y, z in positions:
            stream.write(f'v {x:.9g} {y:.9g} {z:.9g}\n')
        for polygon in polygons:
            stream.write('f ' + ' '.join(str(int(i) + 1) for i in polygon) + '\n')


def surface_area(positions, polygons):
    """Fan-triangulation area in bounded chunks for uniform convex faces."""
    total = 0.0
    for offset in range(0, len(polygons), 10000):
        corners = positions[polygons[offset:offset + 10000]]
        origin = corners[:, 0]
        for i in range(1, corners.shape[1] - 1):
            cross = np.cross(corners[:, i] - origin, corners[:, i + 1] - origin)
            total += np.linalg.norm(cross, axis=1).sum() / 2
    return float(total)


def render_compact_png(path, positions, polygons, level, size=768):
    """Render large meshes directly from arrays, without halfedge objects."""
    eye, right, up, forward, _ = _camera(positions, 35.0, 25.0)
    relative = positions - eye
    depth = relative @ forward
    focal = size / (2 * np.tan(np.deg2rad(20)))
    screen = np.column_stack((size / 2 + focal * (relative @ right) / depth,
                              size / 2 - focal * (relative @ up) / depth))
    screen = np.rint(screen).astype(np.int32)

    face_depth = np.empty(len(polygons), dtype=np.float32)
    tones = np.empty(len(polygons), dtype=np.uint8)
    for offset in range(0, len(polygons), 50000):
        stop = min(offset + 50000, len(polygons))
        batch = polygons[offset:stop]
        face_depth[offset:stop] = depth[batch].mean(axis=1)
        corners = positions[batch[:, :3]]
        normals = np.cross(corners[:, 1] - corners[:, 0],
                           corners[:, 2] - corners[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0] = 1
        shade = 0.35 + 0.6 * np.abs((normals @ forward) / lengths)
        tones[offset:stop] = np.rint(shade * 255).astype(np.uint8)

    base = np.array([110, 150, 210])
    palette = [tuple(np.clip(base * (tone / 255) + 40, 0, 255).astype(np.uint8))
               for tone in range(256)]
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    outline = (35, 45, 60) if level <= 2 else None
    for index in np.argsort(-face_depth):
        corners = polygons[index]
        draw.polygon([tuple(screen[vertex]) for vertex in corners],
                     fill=palette[tones[index]], outline=outline)
    image.save(path)


def bunny_linear():
    original_mesh = load_obj(ROOT / 'meshes' / 'bunny.obj')
    original_mesh.validate()
    if not original_mesh.boundary_faces:
        raise AssertionError('Expected the supplied bunny to have a boundary')
    positions, polygons = read_obj_arrays(ROOT / 'meshes' / 'bunny.obj')
    results = []
    for level in range(1, 5):
        positions, polygons, old_edges = linear_arrays(positions, polygons)
        if level == 1:
            reference = student.subdivide_linear(original_mesh)
            reference.validate()
            np.testing.assert_allclose(positions,
                                       [v.position for v in reference.vertices], rtol=0, atol=1e-12)
            expected_faces = [[v.index for v in student.face_vertices(f)]
                              for f in reference.faces]
            if polygons.tolist() != expected_faces:
                raise AssertionError('Compact linear scheme differs from Task 5')
        path = OUT / f'bunny-linear-{level}.obj'
        preview = OUT / f'bunny-linear-{level}.png'
        write_obj_arrays(path, positions, polygons)
        render_compact_png(preview, positions, polygons, level)
        V, F = len(positions), len(polygons)
        # Every original edge splits in two; each new quad adds a spoke.
        E = 2 * old_edges + F
        row = {'level': level, 'vertices': V, 'edges': E, 'faces': F,
               'boundary_loops': 4, 'euler_characteristic': V - E + F,
               'surface_area': surface_area(positions, polygons),
               'obj': str(path.relative_to(ROOT)),
               'png': str(preview.relative_to(ROOT))}
        results.append(row)
        print('bunny linear', level, V, E, F, f"area={row['surface_area']:.8f}")
    return results


def torus_both():
    results = {}
    for scheme, operation in (('linear', student.subdivide_linear),
                              ('catmull-clark', student.subdivide_catmull_clark)):
        mesh = load_obj(ROOT / 'meshes' / 'torus.obj')
        rows = []
        for level in range(1, 5):
            mesh = operation(mesh)
            mesh.validate()
            obj_path = OUT / f'torus-{scheme}-{level}.obj'
            png_path = OUT / f'torus-{scheme}-{level}.png'
            save_obj(obj_path, mesh, student.face_vertices)
            render_png(png_path, mesh, student.face_vertices, student.face_normal)
            summary = student.mesh_summary(mesh)
            row = {'level': level, 'vertices': summary['vertices'],
                   'edges': summary['edges'], 'faces': summary['faces'],
                   'boundary_loops': summary['boundary loops'],
                   'euler_characteristic': summary['euler characteristic'],
                   'surface_area': summary['surface area'],
                   'obj': str(obj_path.relative_to(ROOT)),
                   'png': str(png_path.relative_to(ROOT))}
            rows.append(row)
            print('torus', scheme, level, row['vertices'], row['edges'],
                  row['faces'], f"area={row['surface_area']:.8f}")
        results[scheme] = rows
    return results


def main():
    OUT.mkdir(exist_ok=True)
    bunny = load_obj(ROOT / 'meshes' / 'bunny.obj')
    try:
        student.subdivide_catmull_clark(bunny)
    except ValueError as error:
        bunny_cc = f'rejected: {error}; {len(bunny.boundary_faces)} boundary loops'
    else:
        raise AssertionError('Catmull-Clark unexpectedly accepted the open bunny')
    result = {'bunny': {'linear': bunny_linear(),
                        'catmull-clark': bunny_cc},
              'torus': torus_both()}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
