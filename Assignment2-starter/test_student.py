"""Reproducible numerical checks for the Assignment 2 report."""

import unittest
from pathlib import Path

import numpy as np

import student
from halfedge import HalfedgeMesh
from support import load_obj


MESHES = Path(__file__).parent / 'meshes'
NAMES = ('cube', 'fan', 'grid', 'lshape', 'octahedron', 'pyramid',
         'tetrahedron', 'torus')


def connectivity(mesh):
    return [(h.twin.index, h.next.index, h.vertex.index,
             h.edge.index, h.face.index, h.face.is_boundary)
            for h in mesh.halfedges]


class AssignmentChecks(unittest.TestCase):
    def test_all_inputs_traversal_geometry_and_summary(self):
        required = {'vertices', 'edges', 'faces', 'halfedges', 'triangles',
                    'quads', 'polygons', 'boundary loops', 'boundary vertices',
                    'euler characteristic', 'surface area'}
        for name in NAMES:
            with self.subTest(mesh=name):
                mesh = load_obj(MESHES / f'{name}.obj')
                self.assertTrue(mesh.validate())
                for face in mesh.faces:
                    hs = student.face_halfedges(face)
                    self.assertEqual(len(hs), len(set(hs)))
                    self.assertEqual([h.vertex for h in hs], student.face_vertices(face))
                    self.assertEqual(student.face_centroid(face).shape, (3,))
                    self.assertAlmostEqual(np.linalg.norm(student.face_normal(face)), 1)
                    self.assertGreater(student.face_area(face), 0)
                for vertex in mesh.vertices:
                    hs = student.vertex_halfedges(vertex)
                    self.assertEqual(len(hs), len(set(hs)))
                    self.assertTrue(all(h.twin.vertex is vertex for h in hs))
                    self.assertEqual([h.vertex for h in hs], student.vertex_neighbors(vertex))
                    self.assertTrue(all(not f.is_boundary for f in student.vertex_faces(vertex)))
                summary = student.mesh_summary(mesh)
                self.assertEqual(set(summary), required)
                self.assertEqual(summary['euler characteristic'],
                                 len(mesh.vertices) - len(mesh.edges) + len(mesh.faces))
                self.assertEqual(summary['boundary vertices'],
                                 sum(any(h.face.is_boundary or h.twin.face.is_boundary
                                         for h in student.vertex_halfedges(v))
                                     for v in mesh.vertices))

    def test_rotated_face_start_and_concave_area(self):
        mesh = load_obj(MESHES / 'lshape.obj')
        face = next(f for f in mesh.faces if len(student.face_halfedges(f)) > 4)
        centroid = student.face_centroid(face)
        normal = student.face_normal(face)
        area = student.face_area(face)
        original = face.halfedge
        face.halfedge = original.next
        self.assertTrue(np.allclose(student.face_centroid(face), centroid))
        self.assertTrue(np.allclose(student.face_normal(face), normal))
        self.assertAlmostEqual(student.face_area(face), area)
        face.halfedge = original
        self.assertAlmostEqual(area, 8.0)

    def test_flip_each_edge_and_rejections(self):
        for name in ('octahedron', 'fan', 'pyramid', 'tetrahedron', 'cube'):
            initial = load_obj(MESHES / f'{name}.obj')
            for index in range(len(initial.edges)):
                with self.subTest(mesh=name, edge=index):
                    mesh = load_obj(MESHES / f'{name}.obj')
                    before = connectivity(mesh)
                    positions = [v.position.copy() for v in mesh.vertices]
                    edge = mesh.edges[index]
                    try:
                        self.assertIs(student.flip_edge(mesh, edge), edge)
                    except ValueError:
                        self.assertEqual(connectivity(mesh), before)
                    else:
                        self.assertTrue(mesh.validate())
                        self.assertTrue(all(not student.is_boundary_vertex(v) or
                                            v.halfedge.twin.face.is_boundary
                                            for v in mesh.vertices))
                        self.assertIs(student.flip_edge(mesh, edge), edge)
                        self.assertTrue(mesh.validate())
                        self.assertEqual(connectivity(mesh), before)
                    self.assertTrue(all(np.array_equal(v.position, p)
                                        for v, p in zip(mesh.vertices, positions)))

    def test_linear_subdivision_all_meshes_and_index_order(self):
        for name in NAMES:
            with self.subTest(mesh=name):
                mesh = load_obj(MESHES / f'{name}.obj')
                before = connectivity(mesh)
                V, E, F = len(mesh.vertices), len(mesh.edges), len(mesh.faces)
                child = student.subdivide_linear(mesh)
                self.assertTrue(child.validate())
                self.assertEqual(connectivity(mesh), before)
                self.assertEqual(len(child.vertices), V + E + F)
                self.assertEqual(len(child.faces),
                                 sum(len(student.face_halfedges(f)) for f in mesh.faces))
                self.assertEqual(len(child.boundary_faces), len(mesh.boundary_faces))
                for v in mesh.vertices:
                    np.testing.assert_allclose(child.vertices[v.index].position, v.position)
                for e in mesh.edges:
                    h = e.halfedge
                    np.testing.assert_allclose(child.vertices[V + e.index].position,
                                               (h.vertex.position + h.twin.vertex.position) / 2)
                for f in mesh.faces:
                    np.testing.assert_allclose(child.vertices[V + E + f.index].position,
                                               student.face_centroid(f))
                expected = student._subdivision_polygons(mesh)
                actual = [[v.index for v in student.face_vertices(f)] for f in child.faces]
                self.assertEqual(actual, expected)

    def test_catmull_clark_closed_open_and_isolated(self):
        for name in NAMES:
            mesh = load_obj(MESHES / f'{name}.obj')
            before = connectivity(mesh)
            if mesh.boundary_faces:
                with self.subTest(mesh=name):
                    with self.assertRaises(ValueError):
                        student.subdivide_catmull_clark(mesh)
                    self.assertEqual(connectivity(mesh), before)
                continue
            with self.subTest(mesh=name):
                child = student.subdivide_catmull_clark(mesh)
                self.assertTrue(child.validate())
                self.assertEqual(connectivity(mesh), before)
                self.assertEqual(len(child.vertices),
                                 len(mesh.vertices) + len(mesh.edges) + len(mesh.faces))
                self.assertEqual([[v.index for v in student.face_vertices(f)]
                                  for f in child.faces], student._subdivision_polygons(mesh))
        cube = load_obj(MESHES / 'cube.obj')
        child = student.subdivide_catmull_clark(cube)
        fpoints = [student.face_centroid(f) for f in cube.faces]
        for edge in cube.edges:
            h = edge.halfedge
            expected = (h.vertex.position + h.twin.vertex.position +
                        fpoints[h.face.index] + fpoints[h.twin.face.index]) / 4
            np.testing.assert_allclose(child.vertices[len(cube.vertices) + edge.index].position,
                                       expected)
        extra = np.vstack(([v.position for v in cube.vertices], [10, 10, 10]))
        polygons = [[v.index for v in student.face_vertices(f)] for f in cube.faces]
        isolated = HalfedgeMesh.from_polygons(extra, polygons)
        result = student.subdivide_catmull_clark(isolated)
        np.testing.assert_array_equal(result.vertices[len(cube.vertices)].position,
                                      [10, 10, 10])

    def test_known_counts_and_four_cube_levels(self):
        expected = {
            'cube': (8, 12, 6, 2, 24.0),
            'fan': (7, 12, 6, 1, None),
            'lshape': (12, 18, 8, 2, 23.2),
            'torus': (72, 144, 72, 0, None),
        }
        for name, (V, E, F, chi, area) in expected.items():
            with self.subTest(mesh=name):
                summary = student.mesh_summary(load_obj(MESHES / f'{name}.obj'))
                self.assertEqual((summary['vertices'], summary['edges'],
                                  summary['faces'], summary['euler characteristic']),
                                 (V, E, F, chi))
                if area is not None:
                    self.assertAlmostEqual(summary['surface area'], area)
        for operation in (student.subdivide_linear, student.subdivide_catmull_clark):
            mesh = load_obj(MESHES / 'cube.obj')
            for level in range(1, 5):
                mesh = operation(mesh)
                self.assertTrue(mesh.validate())
                self.assertEqual(len(mesh.faces), 6 * 4 ** level)
                self.assertEqual(len(mesh.vertices), len(mesh.faces) + 2)
                self.assertEqual(sum(len(student.vertex_neighbors(v)) != 4
                                     for v in mesh.vertices), 8)


if __name__ == '__main__':
    unittest.main()
