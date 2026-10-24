"""Generate the numerical results and sample renders cited in report.pdf.

Run from Assignment2-starter with: python report_data.py
"""

import json
from collections import Counter
from pathlib import Path

import student
from support import load_obj, render_png


ROOT = Path(__file__).parent
MESHES = ROOT / 'meshes'
ASSETS = ROOT / 'report_assets'
NAMES = ('cube', 'fan', 'grid', 'lshape', 'octahedron', 'pyramid',
         'tetrahedron', 'torus')


def metrics(mesh):
    result = student.mesh_summary(mesh)
    result['non4'] = sum(len(student.vertex_neighbors(v)) != 4
                         for v in mesh.vertices)
    return result


def rejection_reason(message):
    if 'boundary' in message:
        return 'boundary'
    if 'triangles' in message:
        return 'nontriangle'
    if 'share an edge' in message:
        return 'existing diagonal'
    raise ValueError(f'Unexpected flip rejection: {message}')


def main():
    ASSETS.mkdir(exist_ok=True)
    data = {'meshes': {}, 'flips': {}, 'cube_levels': {}}
    for name in NAMES:
        source = load_obj(MESHES / f'{name}.obj')
        variants = {'original': metrics(source)}
        linear = student.subdivide_linear(source)
        linear.validate()
        variants['linear'] = metrics(linear)
        if source.is_closed():
            cc = student.subdivide_catmull_clark(source)
            cc.validate()
            variants['catmull-clark'] = metrics(cc)
        data['meshes'][name] = variants

    for name in ('octahedron', 'fan', 'pyramid', 'tetrahedron'):
        edge_count = len(load_obj(MESHES / f'{name}.obj').edges)
        counts = Counter()
        examples = {}
        for index in range(edge_count):
            mesh = load_obj(MESHES / f'{name}.obj')
            try:
                student.flip_edge(mesh, mesh.edges[index])
                mesh.validate()
            except ValueError as error:
                reason = rejection_reason(str(error))
                counts[reason] += 1
                examples.setdefault(reason, index)
            else:
                counts['accepted'] += 1
                examples.setdefault('accepted', index)
        data['flips'][name] = {'counts': {key: counts[key] for key in
                                        ('accepted', 'boundary', 'nontriangle',
                                         'existing diagonal')},
                               'example_edges': examples}

    for scheme, operation in (('linear', student.subdivide_linear),
                              ('catmull-clark', student.subdivide_catmull_clark)):
        mesh = load_obj(MESHES / 'cube.obj')
        levels = [metrics(mesh)]
        for level in range(1, 5):
            mesh = operation(mesh)
            mesh.validate()
            levels.append(metrics(mesh))
            if level in (1, 4):
                render_png(ASSETS / f'cube-{scheme}-{level}.png', mesh,
                           student.face_vertices, student.face_normal)
        data['cube_levels'][scheme] = levels

    fan = load_obj(MESHES / 'fan.obj')
    render_png(ASSETS / 'fan-before.png', fan, student.face_vertices,
               student.face_normal, elevation=75, highlight=(0,))
    student.flip_edge(fan, fan.edges[0])
    fan.validate()
    render_png(ASSETS / 'fan-after.png', fan, student.face_vertices,
               student.face_normal, elevation=75, highlight=(0,))

    path = ASSETS / 'report_data.json'
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(path)
    for name, results in data['flips'].items():
        print(name, results)
    for scheme, rows in data['cube_levels'].items():
        print(scheme, [(r['vertices'], r['faces'], round(r['surface area'], 6),
                        r['non4']) for r in rows])


if __name__ == '__main__':
    main()
