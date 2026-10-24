"""Halfedge mesh queries and operations for Assignment 2."""

import numpy as np

from halfedge import HalfedgeMesh


# Task 1: traversal

def face_halfedges(face):
    """Return a face loop, beginning at its representative halfedge."""
    if face is None or face.halfedge is None:
        return []
    result = []
    h = face.halfedge
    while True:
        result.append(h)
        h = h.next
        if h is face.halfedge:
            return result


def face_vertices(face):
    return [h.vertex for h in face_halfedges(face)]


def vertex_halfedges(vertex):
    """Return outgoing halfedges in circulation order."""
    if vertex is None or vertex.halfedge is None:
        return []
    result = []
    h = vertex.halfedge
    while True:
        result.append(h)
        h = h.twin.next
        if h is vertex.halfedge:
            return result


def vertex_neighbors(vertex):
    return [h.vertex for h in vertex_halfedges(vertex)]


def vertex_faces(vertex):
    return [h.face for h in vertex_halfedges(vertex) if not h.face.is_boundary]


def is_boundary_vertex(vertex):
    return vertex is not None and vertex.halfedge is not None and vertex.halfedge.twin.face.is_boundary


# Task 2: face geometry

def face_centroid(face):
    return np.mean([v.position for v in face_vertices(face)], axis=0)


def face_normal(face):
    points = [v.position for v in face_vertices(face)]
    normal = np.zeros(3)
    for p, q in zip(points, points[1:] + points[:1]):
        normal += np.cross(p - q, p + q)
    length = np.linalg.norm(normal)
    return normal / length if length >= 1e-12 else np.zeros(3)


def face_area(face):
    points = [v.position for v in face_vertices(face)]
    origin = points[0]
    direction = face_normal(face)
    area = 0.0
    for i in range(1, len(points) - 1):
        cross = np.cross(points[i] - origin, points[i + 1] - origin)
        area += np.linalg.norm(cross) * (1 if cross @ direction >= 0 else -1)
    return float(area / 2)


# Task 3: mesh summary

def mesh_summary(mesh):
    sides = [len(face_halfedges(f)) for f in mesh.faces]
    return {
        'vertices': len(mesh.vertices),
        'edges': len(mesh.edges),
        'faces': len(mesh.faces),
        'halfedges': len(mesh.halfedges),
        'triangles': sides.count(3),
        'quads': sides.count(4),
        'polygons': sum(n > 4 for n in sides),
        'boundary loops': len(mesh.boundary_faces),
        'boundary vertices': sum(is_boundary_vertex(v) for v in mesh.vertices),
        'euler characteristic': len(mesh.vertices) - len(mesh.edges) + len(mesh.faces),
        'surface area': float(sum(face_area(f) for f in mesh.faces)),
    }


# Task 4: edge flip

def flip_edge(mesh, edge):
    """Flip an interior edge shared by triangles, reusing every element."""
    ab = edge.halfedge
    ba = ab.twin
    if ab.face.is_boundary or ba.face.is_boundary:
        raise ValueError('cannot flip a boundary edge')
    if len(face_halfedges(ab.face)) != 3 or len(face_halfedges(ba.face)) != 3:
        raise ValueError('both incident faces must be triangles')

    bc, ca = ab.next, ab.next.next
    ad, db = ba.next, ba.next.next
    c, d = bc.vertex, ad.vertex
    if any(h.vertex is d for h in vertex_halfedges(c)):
        raise ValueError('opposite vertices already share an edge')

    f0, f1 = ab.face, ba.face
    a, b = ca.vertex, db.vertex
    # The diagonal key reverses its ordering relative to the opposite
    # pair on the next flip. Alternating the two valid orientations makes
    # flipping the same labeled edge twice an exact inverse.
    diagonal_key = tuple(sorted((a.index, b.index)))
    opposite_key = tuple(sorted((c.index, d.index)))
    if diagonal_key < opposite_key:
        # f0: c->d->b->c; f1: d->c->a->d.
        ab.vertex, ba.vertex = d, c
        ab.next, db.next, bc.next = db, bc, ab
        ba.next, ca.next, ad.next = ca, ad, ba
        db.face = bc.face = f0
        ca.face = ad.face = f1
    else:
        # f0: d->c->a->d; f1: c->d->b->c.
        ab.vertex, ba.vertex = c, d
        ab.next, ca.next, ad.next = ca, ad, ab
        ba.next, db.next, bc.next = db, bc, ba
        ca.face = ad.face = f0
        db.face = bc.face = f1

    if f0.halfedge.face is not f0:
        f0.halfedge = ab
    if f1.halfedge.face is not f1:
        f1.halfedge = ba

    # Only the old diagonal sources can lose their representative halfedge.
    if a.halfedge is not None and a.halfedge.twin.vertex is not a:
        a.halfedge = ad
    if b.halfedge is not None and b.halfedge.twin.vertex is not b:
        b.halfedge = bc
    return edge


def _subdivision_polygons(mesh):
    """Shared quad connectivity, in the prescribed face and loop order."""
    V, E = len(mesh.vertices), len(mesh.edges)
    return [[h.vertex.index, V + h.next.edge.index,
             V + E + f.index, V + h.edge.index]
            for f in mesh.faces for h in face_halfedges(f)]


# Task 5: linear subdivision

def subdivide_linear(mesh):
    positions = [v.position.copy() for v in mesh.vertices]
    positions += [(e.halfedge.vertex.position + e.halfedge.twin.vertex.position) / 2
                  for e in mesh.edges]
    positions += [face_centroid(f) for f in mesh.faces]
    return HalfedgeMesh.from_polygons(positions, _subdivision_polygons(mesh))


# Task 6: Catmull-Clark subdivision

def subdivide_catmull_clark(mesh):
    if mesh.boundary_faces:
        raise ValueError('Catmull-Clark requires a closed mesh')

    face_points = [face_centroid(f) for f in mesh.faces]
    edge_midpoints = [(e.halfedge.vertex.position + e.halfedge.twin.vertex.position) / 2
                      for e in mesh.edges]
    edge_points = []
    for e in mesh.edges:
        h = e.halfedge
        edge_points.append((h.vertex.position + h.twin.vertex.position
                            + face_points[h.face.index] + face_points[h.twin.face.index]) / 4)

    vertex_points = []
    for v in mesh.vertices:
        outgoing = vertex_halfedges(v)
        if not outgoing:
            vertex_points.append(v.position.copy())
            continue
        n = len(outgoing)
        Q = np.mean([face_points[h.face.index] for h in outgoing], axis=0)
        R = np.mean([edge_midpoints[h.edge.index] for h in outgoing], axis=0)
        vertex_points.append((Q + 2 * R + (n - 3) * v.position) / n)

    return HalfedgeMesh.from_polygons(vertex_points + edge_points + face_points,
                                      _subdivision_polygons(mesh))
