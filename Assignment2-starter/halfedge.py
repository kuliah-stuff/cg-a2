"""Halfedge mesh data structure used by Assignment 2.

Do not modify this file. The grader uses its own copy.

Conventions
-----------
* ``Halfedge.vertex`` is the TARGET vertex of the halfedge (the vertex the
  halfedge points to), as in the lecture slides.
* ``Vertex.halfedge`` is an OUTGOING halfedge of the vertex (its target is a
  neighbor). For a vertex on the boundary, it is the outgoing interior
  halfedge that lies on a boundary edge.
* ``Face.halfedge`` is one halfedge of the face loop.
* Interior faces are listed in ``mesh.faces``. Every boundary loop is
  represented by one extra face with ``is_boundary = True``, listed in
  ``mesh.boundary_faces``. Boundary halfedges are linked with ``next``
  pointers into closed loops, so ``h = h.next`` and ``h = h.twin.next``
  always terminate.
* Every element stores ``index``, its position in the corresponding list of
  the mesh. Indices are assigned when the mesh is built and never change.
"""

import numpy as np


class Vertex:
    __slots__ = ('index', 'position', 'halfedge')

    def __init__(self, index, position):
        self.index = index
        self.position = np.asarray(position, dtype=float).copy()
        self.halfedge = None

    def __repr__(self):
        return f'Vertex({self.index})'


class Edge:
    __slots__ = ('index', 'halfedge')

    def __init__(self, index):
        self.index = index
        self.halfedge = None

    def __repr__(self):
        return f'Edge({self.index})'


class Face:
    __slots__ = ('index', 'halfedge', 'is_boundary')

    def __init__(self, index, is_boundary=False):
        self.index = index
        self.halfedge = None
        self.is_boundary = is_boundary

    def __repr__(self):
        kind = 'BoundaryFace' if self.is_boundary else 'Face'
        return f'{kind}({self.index})'


class Halfedge:
    __slots__ = ('index', 'twin', 'next', 'vertex', 'edge', 'face')

    def __init__(self, index):
        self.index = index
        self.twin = None
        self.next = None
        self.vertex = None
        self.edge = None
        self.face = None

    def __repr__(self):
        return f'Halfedge({self.index})'


class MeshError(ValueError):
    """Raised when a polygon list cannot be stored as a manifold halfedge mesh."""


class HalfedgeMesh:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.faces = []
        self.boundary_faces = []
        self.halfedges = []

    @classmethod
    def from_polygons(cls, positions, polygons):
        """Build a mesh from vertex positions (n, 3) and a list of polygons.

        Each polygon is a list of at least three vertex indices in
        counter-clockwise order seen from outside. Interior faces, vertices,
        and edges receive indices in the order they first appear:
        ``mesh.faces[i]`` is ``polygons[i]``, ``mesh.vertices[i]`` is
        ``positions[i]``, and edges are numbered in the order their first
        halfedge is created while walking the polygons. ``face.halfedge`` is
        the halfedge from ``polygon[-1]`` to ``polygon[0]``, so walking the
        face loop and collecting targets yields ``polygon`` in its given order.
        """
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise MeshError('positions must have shape (n, 3)')
        if not np.all(np.isfinite(positions)):
            raise MeshError('positions contain infinity or NaN')
        mesh = cls()
        for i, p in enumerate(positions):
            mesh.vertices.append(Vertex(i, p))

        directed = {}
        for polygon in polygons:
            polygon = [int(i) for i in polygon]
            if len(polygon) < 3:
                raise MeshError(f'polygon {polygon} has fewer than three vertices')
            if len(set(polygon)) != len(polygon):
                raise MeshError(f'polygon {polygon} repeats a vertex')
            for i in polygon:
                if not 0 <= i < len(mesh.vertices):
                    raise MeshError(f'polygon {polygon} uses missing vertex {i}')
            face = Face(len(mesh.faces))
            mesh.faces.append(face)
            loop = []
            n = len(polygon)
            for k in range(n):
                source, target = polygon[k - 1], polygon[k]
                if (source, target) in directed:
                    raise MeshError(
                        f'directed edge {source}->{target} is used twice; '
                        'the mesh is non-manifold or inconsistently oriented')
                h = mesh._new_halfedge()
                h.vertex = mesh.vertices[target]
                h.face = face
                directed[(source, target)] = h
                loop.append(h)
            for k in range(n):
                loop[k].next = loop[(k + 1) % n]
            face.halfedge = loop[0]

        for (source, target), h in directed.items():
            if h.edge is not None:
                continue
            edge = Edge(len(mesh.edges))
            mesh.edges.append(edge)
            edge.halfedge = h
            h.edge = edge
            other = directed.get((target, source))
            if other is not None:
                other.edge = edge
                h.twin = other
                other.twin = h

        mesh._close_boundary(directed)
        mesh._assign_vertex_halfedges(directed)
        return mesh

    def _new_halfedge(self):
        h = Halfedge(len(self.halfedges))
        self.halfedges.append(h)
        return h

    def _close_boundary(self, directed):
        # A boundary halfedge b runs opposite to an interior halfedge without a twin.
        outgoing = {}
        for (source, target), h in directed.items():
            if h.twin is not None:
                continue
            b = self._new_halfedge()
            b.vertex = self.vertices[source]
            b.edge = h.edge
            b.twin = h
            h.twin = b
            if target in outgoing:
                raise MeshError(
                    f'vertex {target} touches the boundary more than once; '
                    'the mesh is not manifold at this vertex')
            outgoing[target] = b
        unvisited = set(outgoing)
        while unvisited:
            start = outgoing[min(unvisited)]
            face = Face(len(self.boundary_faces), is_boundary=True)
            self.boundary_faces.append(face)
            face.halfedge = start
            b = start
            while True:
                unvisited.discard(b.twin.vertex.index)
                b.face = face
                b.next = outgoing[b.vertex.index]
                b = b.next
                if b is start:
                    break

    def _assign_vertex_halfedges(self, directed):
        for (source, target), h in directed.items():
            v = self.vertices[source]
            if v.halfedge is None:
                v.halfedge = h
            elif h.twin.face.is_boundary and not v.halfedge.twin.face.is_boundary:
                v.halfedge = h

    def is_closed(self):
        """Return True when the mesh has no boundary loop."""
        return not self.boundary_faces

    def validate(self, max_steps=100000):
        """Check pointer invariants and raise MeshError on the first violation.

        This checks connectivity only. It does not check geometry.
        """
        halfedges = set(map(id, self.halfedges))
        for h in self.halfedges:
            if h.twin is None or h.next is None or h.vertex is None or h.edge is None or h.face is None:
                raise MeshError(f'{h} has a missing pointer')
            if h.twin is h or h.twin.twin is not h:
                raise MeshError(f'{h}.twin.twin is not {h}')
            if h.twin.edge is not h.edge:
                raise MeshError(f'{h} and its twin store different edges')
            if h.vertex is h.twin.vertex:
                raise MeshError(f'{h} starts and ends at {h.vertex}')
            if h.next.face is not h.face:
                raise MeshError(f'{h} and {h.next} belong to different faces')
            if h.next.twin.vertex is not h.vertex:
                raise MeshError(f'{h}.next does not start where {h} ends')
            if id(h.next) not in halfedges or id(h.twin) not in halfedges:
                raise MeshError(f'{h} points to a halfedge outside the mesh')
        for e in self.edges:
            if e.halfedge is None or e.halfedge.edge is not e:
                raise MeshError(f'{e}.halfedge does not belong to {e}')
        for f in self.faces + self.boundary_faces:
            h = f.halfedge
            if h is None or h.face is not f:
                raise MeshError(f'{f}.halfedge does not belong to {f}')
            steps = 0
            while True:
                if h.face is not f:
                    raise MeshError(f'the loop of {f} leaves the face')
                h = h.next
                steps += 1
                if h is f.halfedge:
                    break
                if steps > max_steps:
                    raise MeshError(f'the loop of {f} does not close')
            if not f.is_boundary and steps < 3:
                raise MeshError(f'{f} has fewer than three sides')
        for v in self.vertices:
            if v.halfedge is not None and v.halfedge.twin.vertex is not v:
                raise MeshError(f'{v}.halfedge does not start at {v}')
        seen = {}
        for h in self.halfedges:
            key = (h.twin.vertex.index, h.vertex.index)
            if key in seen:
                raise MeshError(f'two halfedges run from vertex {key[0]} to vertex {key[1]}')
            seen[key] = h
        return True
