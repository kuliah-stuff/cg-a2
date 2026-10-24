"""File input/output, rendering, and command helpers for Assignment 2.

Do not modify this file. The grader uses its own copy.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from halfedge import HalfedgeMesh, MeshError


def load_obj(path):
    """Read vertex positions and polygons from an OBJ file (v and f lines only)."""
    positions, polygons = [], []
    with open(path, encoding='utf-8') as file:
        for number, line in enumerate(file, 1):
            parts = line.split('#', 1)[0].split()
            if not parts:
                continue
            if parts[0] == 'v':
                if len(parts) < 4:
                    raise MeshError(f'{path}:{number}: a vertex needs three coordinates')
                positions.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                polygon = []
                for token in parts[1:]:
                    index = int(token.split('/')[0])
                    polygon.append(index - 1 if index > 0 else len(positions) + index)
                polygons.append(polygon)
    return HalfedgeMesh.from_polygons(np.array(positions, dtype=float).reshape(-1, 3), polygons)


def save_obj(path, mesh, face_vertices):
    """Write the interior faces of the mesh, using the student's face_vertices."""
    with open(path, 'w', encoding='utf-8') as file:
        for v in mesh.vertices:
            x, y, z = v.position
            file.write(f'v {x:.9g} {y:.9g} {z:.9g}\n')
        for face in mesh.faces:
            indices = ' '.join(str(v.index + 1) for v in face_vertices(face))
            file.write(f'f {indices}\n')


def _camera(points, azimuth, elevation):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.linalg.norm(points - center, axis=1).max()), 1e-9)
    az, el = math.radians(azimuth), math.radians(elevation)
    forward = -np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    eye = center - forward * radius * 3.0
    return eye, right, up, forward, radius


def render_png(path, mesh, face_vertices, face_normal=None, size=512,
               azimuth=35.0, elevation=25.0, highlight=()):
    """Render interior faces with a pinhole camera and the painter's algorithm.

    face_vertices is required. If face_normal is given, faces are shaded by the
    angle between their normal and the view direction; otherwise all faces use
    one flat color. Edges whose index is in highlight are drawn in red.
    """
    points = np.array([v.position for v in mesh.vertices]).reshape(-1, 3)
    image = Image.new('RGB', (size, size), (255, 255, 255))
    if len(points) == 0 or not mesh.faces:
        image.save(path)
        return
    eye, right, up, forward, radius = _camera(points, azimuth, elevation)
    rel = points - eye
    depth = rel @ forward
    focal = size / (2 * math.tan(math.radians(20)))
    xs = size / 2 + focal * (rel @ right) / depth
    ys = size / 2 - focal * (rel @ up) / depth
    screen = np.stack([xs, ys], axis=1)

    polygons = []
    for face in mesh.faces:
        ids = [v.index for v in face_vertices(face)]
        shade = 0.85
        if face_normal is not None:
            n = np.asarray(face_normal(face), dtype=float)
            shade = 0.35 + 0.6 * abs(float(n @ forward))
        polygons.append((depth[ids].mean(), ids, shade))
    polygons.sort(key=lambda item: -item[0])

    draw = ImageDraw.Draw(image)
    base = np.array([110, 150, 210])
    for _, ids, shade in polygons:
        fill = tuple(int(c) for c in np.clip(base * shade + 40, 0, 255))
        draw.polygon([tuple(screen[i]) for i in ids], fill=fill, outline=(30, 30, 30))
    for index in highlight:
        h = mesh.edges[index].halfedge
        a, b = screen[h.twin.vertex.index], screen[h.vertex.index]
        draw.line([tuple(a), tuple(b)], fill=(220, 30, 30), width=3)
    image.save(path)
