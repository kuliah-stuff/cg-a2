"""Boundary-aware Catmull-Clark previews for the open Stanford bunny.

This is a separate visualization extension. Assignment Task 6 rejects open
meshes; it does not use this function. Run from Assignment2-starter with:
    python -u generate_bunny_cc_boundary.py
"""

import json
from pathlib import Path

import numpy as np

import student
from generate_subdivision_results import (OUT, ROOT, read_obj_arrays,
                                          render_compact_png, surface_area,
                                          write_obj_arrays)
from halfedge import HalfedgeMesh
from support import load_obj


def catmull_clark_boundary_arrays(positions, polygons):
    """Use smooth interior stencils and crease stencils along open boundaries."""
    V, F = len(positions), len(polygons)
    face_points = positions[polygons].mean(axis=1)

    edge_ids, pairs, adjacent_faces = {}, [], []
    for face_index, polygon in enumerate(polygons):
        for previous, current in zip(polygon[-1:].tolist() + polygon[:-1].tolist(),
                                     polygon.tolist()):
            key = (min(previous, current), max(previous, current))
            if key not in edge_ids:
                edge_ids[key] = len(pairs)
                pairs.append(key)
                adjacent_faces.append([face_index, -1])
            else:
                index = edge_ids[key]
                if adjacent_faces[index][1] != -1:
                    raise ValueError(f'Nonmanifold edge {key}')
                adjacent_faces[index][1] = face_index

    endpoints = np.asarray(pairs, dtype=np.int32)
    adjacent_faces = np.asarray(adjacent_faces, dtype=np.int32)
    E = len(endpoints)
    first, second = endpoints[:, 0], endpoints[:, 1]
    midpoints = (positions[first] + positions[second]) / 2
    edge_points = midpoints.copy()
    interior_edges = adjacent_faces[:, 1] >= 0
    edge_points[interior_edges] = (
        positions[first[interior_edges]] + positions[second[interior_edges]]
        + face_points[adjacent_faces[interior_edges, 0]]
        + face_points[adjacent_faces[interior_edges, 1]]
    ) / 4

    face_sum = np.zeros((V, 3))
    face_count = np.zeros(V, dtype=np.int32)
    for corner in range(polygons.shape[1]):
        np.add.at(face_sum, polygons[:, corner], face_points)
        np.add.at(face_count, polygons[:, corner], 1)

    midpoint_sum = np.zeros((V, 3))
    edge_count = np.zeros(V, dtype=np.int32)
    np.add.at(midpoint_sum, first, midpoints)
    np.add.at(midpoint_sum, second, midpoints)
    np.add.at(edge_count, first, 1)
    np.add.at(edge_count, second, 1)

    vertex_points = positions.copy()
    active = edge_count > 0
    smooth = active & (face_count == edge_count)
    n = edge_count[smooth, None]
    Q = face_sum[smooth] / n
    R = midpoint_sum[smooth] / n
    vertex_points[smooth] = (Q + 2 * R + (n - 3) * positions[smooth]) / n

    boundary = adjacent_faces[:, 1] < 0
    boundary_first, boundary_second = first[boundary], second[boundary]
    boundary_sum = np.zeros((V, 3))
    boundary_count = np.zeros(V, dtype=np.int32)
    np.add.at(boundary_sum, boundary_first, positions[boundary_second])
    np.add.at(boundary_sum, boundary_second, positions[boundary_first])
    np.add.at(boundary_count, boundary_first, 1)
    np.add.at(boundary_count, boundary_second, 1)
    if np.any((boundary_count != 0) & (boundary_count != 2)):
        raise ValueError('Expected each boundary vertex to have two boundary neighbors')
    on_boundary = boundary_count == 2
    vertex_points[on_boundary] = (
        6 * positions[on_boundary] + boundary_sum[on_boundary]
    ) / 8

    refined = np.concatenate((vertex_points, edge_points, face_points))
    quads = np.empty((polygons.size, 4), dtype=np.int32)
    row = 0
    for face_index, polygon in enumerate(polygons):
        size = len(polygon)
        for i, current in enumerate(polygon):
            previous, following = polygon[i - 1], polygon[(i + 1) % size]
            ahead = edge_ids[(min(current, following), max(current, following))]
            behind = edge_ids[(min(previous, current), max(previous, current))]
            quads[row] = (current, V + ahead, V + E + face_index, V + behind)
            row += 1
    return refined, quads, E


def verify_closed_case():
    positions, polygons = read_obj_arrays(ROOT / 'meshes' / 'torus.obj')
    refined, quads, _ = catmull_clark_boundary_arrays(positions, polygons)
    reference = student.subdivide_catmull_clark(load_obj(ROOT / 'meshes' / 'torus.obj'))
    np.testing.assert_allclose(refined, [v.position for v in reference.vertices],
                               rtol=0, atol=1e-12)
    expected_faces = [[v.index for v in student.face_vertices(f)]
                      for f in reference.faces]
    if quads.tolist() != expected_faces:
        raise AssertionError('Closed-case connectivity differs from Task 6')


def main():
    verify_closed_case()
    OUT.mkdir(exist_ok=True)
    positions, polygons = read_obj_arrays(ROOT / 'meshes' / 'bunny.obj')
    rows = []
    for level in range(1, 5):
        positions, polygons, old_edges = catmull_clark_boundary_arrays(
            positions, polygons)
        if level == 1:
            mesh = HalfedgeMesh.from_polygons(positions, polygons.tolist())
            mesh.validate()
            if len(mesh.boundary_faces) != 4:
                raise AssertionError('Boundary loop count changed')
        obj = OUT / f'bunny-cc-boundary-{level}.obj'
        png = OUT / f'bunny-cc-boundary-{level}.png'
        write_obj_arrays(obj, positions, polygons)
        render_compact_png(png, positions, polygons, level)
        V, F = len(positions), len(polygons)
        E = 2 * old_edges + F
        row = {'level': level, 'vertices': V, 'edges': E, 'faces': F,
               'euler_characteristic': V - E + F,
               'boundary_loops': 4, 'surface_area': surface_area(positions, polygons),
               'obj': str(obj.relative_to(ROOT)), 'png': str(png.relative_to(ROOT))}
        rows.append(row)
        print(level, V, E, F, f"area={row['surface_area']:.8f}", flush=True)
    (OUT / 'bunny_cc_boundary_summary.json').write_text(
        json.dumps({'method': 'Boundary-aware Catmull-Clark (outside Task 6)',
                    'boundary_rule': 'boundary edge midpoint; boundary vertex (6S + N1 + N2)/8',
                    'levels': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
