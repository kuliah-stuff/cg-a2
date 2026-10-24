"""Implement the tasks described in the Assignment 2 PDF.

Only this file is submitted. Do not change the function signatures.
Read the conventions at the top of halfedge.py before starting.
"""

import numpy as np

from halfedge import HalfedgeMesh


# Task 1: traversal

def face_halfedges(face):
    """Task 1. Return the halfedges of the face loop, starting at face.halfedge."""
    if(face == None): return []
    hlist = []
    start = face.halfedge
    hlist.append(start)
    h = start.next

    while(h != start):
        hlist.append(h)
        h = h.next
    return hlist

def face_vertices(face):
    """Task 1. Return the target vertices of face_halfedges(face), in the same order."""
    return [h.vertex for h in face_halfedges(face)] if face != None else []

def vertex_halfedges(vertex):
    """Task 1. Return the outgoing halfedges of the vertex, starting at vertex.halfedge."""
    if(vertex == None): return []
    hlist = []
    start = vertex.halfedge
    h = start.twin.next

    hlist.append(start)

    while(h != start):
        hlist.append(h)
        start.twin.next

    return hlist


def vertex_neighbors(vertex):
    """Task 1. Return the neighboring vertices, in the order of vertex_halfedges(vertex)."""
    return [h.vertex for h in vertex_halfedges(vertex)] if vertex != None else []


def vertex_faces(vertex):
    """Task 1. Return the adjacent interior faces, in the order of vertex_halfedges(vertex)."""
    return [h.face for h in vertex_halfedges(vertex)] if vertex != None else []


def is_boundary_vertex(vertex):
    """Task 1. Return True if the vertex lies on a boundary loop."""
    return vertex.halfedge.face.is_boundary if vertex != None else False


# Task 2: face geometry

def face_centroid(face):
    """Task 2. Return the average of the face vertex positions, shape (3,)."""
    np.array([v.position for v in vertex_neighbors(face.halfedge.vertex)]).mean(axis=1)


def face_normal(face):
    """Task 2. Return the unit normal from Newell's method, shape (3,)."""
    vertexes = vertex_neighbors(face.halfedge.vertex)
    n = (
        np.cross((vertexes[0] - vertexes[1]), (vertexes[0] + vertexes[1])) +
        np.cross((vertexes[1] - vertexes[2]), (vertexes[1] + vertexes[2]))
    )
    return n if np.norm(n) >= 1e-12 else np.zeros(3) 

def face_area(face):
    """Task 2. Return the area of the planar polygon as a float."""
    vertexes = vertex_neighbors(face.halfedge.vertex)
    return float(np.norm(np.cross((vertexes[1] - vertexes[0]), (vertexes[2] + vertexes[0]))).value/2)


# Task 3: mesh summary

def mesh_summary(mesh):
    """Task 3. Return a dict with the keys listed in the PDF."""
    traingles = 0
    quads = 0
    polygons = 0

    for f in mesh.face:
        if f.is_boundary: 
            continue
        else: 
            sides = 2
            start = f.halfedge
            h = start.next

            while(h != start):
                sides += 1
                h = h.next

            if sides == 3: traingles += 1
            elif sides == 4: quads += 1
            else: polygons += 1         

    return {
        "vertices" : len(mesh.vertices),
        "edges" : len(mesh.edges), 
        "faces" : len(mesh.faces), 
        "halfedges" : len(mesh.halfedges), 
        "traingles" : traingles,
        "quads" : quads,
        "polygons" : polygons,        
        "boundary_loops" : len(mesh.boundary_faces),
        "boundary vertices" : sum([is_boundary_vertex(v) for v in mesh.vertex]),
        "euler_characteristics" : len(mesh.vertices) - len(mesh.edges) + len(mesh.faces) - sum([is_boundary_vertex(f) for f in mesh.faces]),
        "surface area" : sum([face_area(f) for f in mesh.faces])
    }


# Task 4: edge flip

def flip_edge(mesh, edge):
    """Task 4. Flip an interior edge between two triangles in place and return it."""
    # there are two cases, the other face having `ba` as its halfedge or not. if not, say the other 
    # face have `db` as its half edge, then its he will remain unchanged after flipping. If `ba` 
    # were the he, then it'd also automatically be chaned and handled as we modify directly that 
    # he's attribute and the face object doesn't have any he or vertex attributes other than its
    # starting he

    h0 = edge.halfedge # ab
    h1 = h0.twin #ba

    for h in [h0, h1]:
        temp_h = h.next
        sides = 2
        while temp_h != h:
            if sides > 3: raise ValueError("N eof the faces are not a trianngle")
            sides += 1
            temp_h = temp_h.next

    c = h0.next.vertex
    d = h1.next.vertex

    if d in vertex_neighbors(c): raise ValueError("c and d have been connected by an edge")

    ca = h0.next.next
    ad = h1.next
    db = h1.next.next
    bc = h0.next

    h0.next = ca
    ad.next = h0
    h1.next = db
    bc.next = h1

    h0.vertex = c
    h1.vertex = d
    

# Task 5: linear subdivision

def subdivide_linear(mesh):
    """Task 5. Return a new HalfedgeMesh after one step of linear subdivision."""
    V = len(mesh.vertices)
    E = len(mesh.edges)
    F = len(mesh.faces)
    vertices = mesh.vertices + [0]*(E + F)
    polygons = []

    for f in mesh.faces:
        start = f.halfedge
        h = start
        vek = V + E + f.index

        f_center = start.twin.vertex.position
        count =  1
        while h.vertex.index != start.twin.vertex.index:
            f_center += h.vertex.position

            j = h.edge.index # index of current edge
            vj = V + j # index of new vertex midpoint of current edge
            vertices[vj] = (h.vertex.position + h.twin.vertex.position)/2

            polygons.append([h.vertex.index, vj, vek, vj-1])
            
            h = h.next
            count += 1
        f_center /= count
        vertices[vek] = f_center

    return HalfedgeMesh.from_polygons(vertices, polygons)


# Task 6: Catmull-Clark subdivision

def subdivide_catmull_clark(mesh):
    """Task 6. Return a new HalfedgeMesh after one Catmull-Clark step on a closed mesh."""

    V = len(mesh.vertices)
    E = len(mesh.edges)
    F = len(mesh.faces)
    vertices = mesh.vertices + [0]*(E + F)
    polygons = []

    for f in mesh.faces:
        start = f.halfedge
        h = start
        vek = V + E + f.index

        f_center = start.twin.vertex.position
        count =  1
        while h.vertex.index != start.twin.vertex.index:
            f_center += h.vertex.position

            j = h.edge.index # index of current edge
            vj = V + j # index of new vertex midpoint of current edge
            vertices[vj] = (h.vertex.position + h.twin.vertex.position)/2

            polygons.append([h.vertex.index, vj, vek, vj-1])
            
            h = h.next
            count += 1
        f_center /= count
        vertices[vek] = f_center

    sub = HalfedgeMesh.from_polygons(vertices, polygons)

    old_vertices = sub.vertices
    new_vertices = sub.vertices

    for i in range(V, V+E):
        new_vertices[i].position = [v.position for v in vertex_neighbors(old_vertices[i])].mean(axis=1)

    for i in range(V+E, V+E+F):
        sum = [0, 0, 0]
        faces = vertex_faces(old_vertices[i])
        for f in faces:
            sum += [v.postion for v in face_vertices(f)].sum(axis=1)

        sum -= len(faces)*old_vertices[i]
        sum /= len(faces)


