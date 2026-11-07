import time

import numpy as np
import pytest

import sdfnumba as sn

try:
    import sdf as fg   # fogleman's sdf, for reference values
except ImportError:
    fg = None

needs_fogleman = pytest.mark.skipif(fg is None, reason="fogleman's sdf not installed")

RNG = np.random.default_rng(1)
POINTS = RNG.uniform(-2, 2, (4000, 3))


def same_shape(build):
    """Build the same shape with both libraries (they share an API)."""
    return build(sn), build(fg)


CASES = {
    'sphere': lambda m: m.sphere(1.2, center=(0.1, 0.2, -0.3)),
    'plane': lambda m: m.plane((1, 2, 3), (0.1, 0, 0)),
    'slab': lambda m: m.slab(x0=-1, x1=1, z0=-0.5, k=0.1),
    'box': lambda m: m.box((1, 1.5, 2), center=(0.2, 0, 0)),
    'box_ab': lambda m: m.box(a=(-1, -0.5, 0), b=(1, 0.5, 1)),
    'rounded_box': lambda m: m.rounded_box((1, 1.5, 2), 0.2),
    'torus': lambda m: m.torus(1, 0.25),
    'capsule': lambda m: m.capsule((-1, 0, 0), (1, 0.5, 0.2), 0.3),
    'cylinder': lambda m: m.cylinder(0.5),
    'capped_cylinder': lambda m: m.capped_cylinder((0, 0, -1), (0.2, 0.3, 1), 0.5),
    'rounded_cylinder': lambda m: m.rounded_cylinder(0.6, 0.1, 1.5),
    'capped_cone': lambda m: m.capped_cone((0, 0, -1), (0, 0.2, 1), 0.8, 0.3),
    'rounded_cone': lambda m: m.rounded_cone(0.6, 0.2, 1.2),
    'ellipsoid': lambda m: m.ellipsoid((1, 0.6, 0.4)),
    'tetrahedron': lambda m: m.tetrahedron(1),
    'octahedron': lambda m: m.octahedron(1),
    'union': lambda m: m.sphere(1) | m.box(1.5).translate((0.5, 0, 0)),
    'smooth_union': lambda m: m.sphere(1) | m.box(1.5).translate((0.5, 0, 0)).k(0.25),
    'difference': lambda m: m.box(1.5) - m.sphere(0.9),
    'smooth_difference': lambda m: m.difference(m.box(1.5), m.sphere(0.9), k=0.2),
    'intersection': lambda m: m.sphere(1) & m.box(1.5),
    'smooth_intersection': lambda m: m.intersection(m.sphere(1), m.box(1.5), k=0.3),
    'blend': lambda m: m.blend(m.sphere(1), m.box(1.5), k=0.3),
    'negate': lambda m: m.sphere(1).negate(),
    'dilate': lambda m: m.box(1).dilate(0.2),
    'erode': lambda m: m.box(1).erode(0.2),
    'shell': lambda m: m.sphere(1).shell(0.1),
    'scale': lambda m: m.box(1).scale((1, 2, 0.5)),
    'rotate': lambda m: m.box((1, 0.5, 0.2)).rotate(0.7, (1, 1, 0)),
    'orient': lambda m: m.capped_cylinder((0, 0, 0), (0, 0, 1), 0.3).orient((1, 0.5, 0)),
    'twist': lambda m: m.box((1, 0.5, 2)).twist(0.8),
    'bend': lambda m: m.box((2, 0.3, 0.5)).bend(0.5),
    'elongate': lambda m: m.sphere(0.5).elongate((0.5, 0.2, 0)),
    'circular_array': lambda m: m.sphere(0.2).circular_array(6, 1.0),
    'repeat': lambda m: m.sphere(0.2).repeat(0.6, count=2),
    'repeat_padded': lambda m: m.sphere(0.35).translate((0.1, 0, 0)).repeat((0.6, 0.6, 0), padding=1),
    'nested': lambda m: (m.sphere(1) & m.box(1.5)) - (m.cylinder(0.5) | m.cylinder(0.5).orient(m.X)),
}


@needs_fogleman
@pytest.mark.parametrize('name', CASES)
def test_matches_fogleman(name):
    ours, theirs = same_shape(CASES[name])
    np.testing.assert_allclose(ours(POINTS), theirs(POINTS), rtol=1e-6, atol=1e-6)


def edges_of(faces):
    return np.stack([faces, np.roll(faces, -1, axis=1)], -1).reshape(-1, 2)


def assert_closed_manifold(faces):
    """Every directed edge appears exactly once, and so does its reverse."""
    e = edges_of(faces)
    directed = {tuple(x) for x in e}
    assert len(directed) == len(e), 'an edge is used twice in the same direction'
    assert all((b, a) in directed for a, b in directed), 'mesh has a boundary'


def signed_volume(verts, faces):
    tris = sn.triangulate(faces, verts) if faces.shape[1] == 4 else faces
    a, b, c = (verts[tris[:, i]] for i in range(3))
    return np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6


def test_sphere_mesh():
    verts, faces = sn.sphere(1).generate(samples=2 ** 18)
    assert_closed_manifold(faces)
    assert abs(signed_volume(verts, faces) / (4 / 3 * np.pi) - 1) < 0.01
    r = np.linalg.norm(verts, axis=1)
    assert np.abs(r - 1).max() < 0.02


def test_csg_genus():
    f = sn.sphere(1) & sn.box(1.5)
    c = sn.cylinder(0.5)
    f -= c.orient(sn.X) | c.orient(sn.Y) | c
    verts, faces = f.generate(samples=2 ** 19)
    assert_closed_manifold(faces)
    euler = len(verts) - len(faces) * 2 + len(faces)   # quads: E = 2F
    assert euler == 2 - 2 * 5                           # three crossing holes
    assert signed_volume(verts, faces) > 0


def test_cut_by_bounds_is_capped():
    # bounds cut the sphere in half; the result must still be closed
    verts, faces = sn.sphere(1).generate(bounds=((-1.1, -1.1, 0), (1.1, 1.1, 1.1)),
                                         samples=2 ** 17)
    assert_closed_manifold(faces)
    assert signed_volume(verts, faces) > 0


def test_triangles_option():
    verts, faces = sn.torus(1, 0.3).generate(samples=2 ** 17, triangles=True)
    assert faces.shape[1] == 3
    assert_closed_manifold(faces)


def test_skipping_does_not_change_the_mesh():
    f = sn.sphere(1) - sn.box(0.6).repeat(0.9, count=1)
    prog = f.program()
    lo, step, shape = np.full(3, -1.2), np.full(3, 0.02), (121, 121, 121)
    skipped_vals, skipped = sn.sample_grid(prog, lo, step, shape, block=8)
    assert skipped > 0
    axes = [lo[i] + np.arange(shape[i]) * step[i] for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    full_vals = sn.evaluate(prog, pts).reshape(shape).astype(np.float32)
    v1, q1 = sn.surface_nets(skipped_vals, lo, step)
    v2, q2 = sn.surface_nets(full_vals, lo, step)
    np.testing.assert_allclose(v1, v2, atol=1e-6)
    np.testing.assert_array_equal(q1, q2)


def test_param_change_keeps_code():
    a = (sn.sphere(1) | sn.box(1.5).k(0.2)).program()
    b = (sn.sphere(0.7) | sn.box(1.1).k(0.4)).program()
    c = (sn.sphere(1) & sn.box(1.5)).program()
    assert a.key == b.key
    assert a.key != c.key


# --- dual contouring ------------------------------------------------------

@pytest.mark.parametrize('sharp', [False, True])
def test_csg_closed_both_modes(sharp):
    f = sn.box(1.5) - sn.sphere(0.4).translate((0.75, 0.75, 0.75))
    verts, faces = f.generate(samples=2 ** 18, sharp=sharp)
    assert_closed_manifold(faces)
    assert signed_volume(verts, faces) > 0


def test_dc_hits_box_corners_and_edges():
    f = sn.box((1.5, 1.2, 1.0))
    verts, _ = f.generate(samples=2 ** 17, sharp=True)
    half = np.array([0.75, 0.6, 0.5])
    for corner in np.array(np.meshgrid(*[(-1, 1)] * 3)).T.reshape(-1, 3) * half:
        assert np.linalg.norm(verts - corner, axis=1).min() < 1e-6
    # every vertex lies on the box surface
    assert np.abs(f(verts)).max() < 1e-6


def test_dc_sphere_still_round():
    verts, faces = sn.sphere(1).generate(samples=2 ** 18, sharp=True)
    assert_closed_manifold(faces)
    assert np.abs(np.linalg.norm(verts, axis=1) - 1).max() < 0.005


def test_dc_cut_by_bounds_is_capped():
    verts, faces = sn.box(1.5).generate(bounds=((-1, -1, 0), (1, 1, 1)),
                                        samples=2 ** 17, sharp=True)
    assert_closed_manifold(faces)
    assert signed_volume(verts, faces) > 0


# --- remeshing --------------------------------------------------------------

def edge_lengths(verts, tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)


def test_remesh_sphere_matches_sizing():
    f = sn.sphere(1)
    verts, faces = f.generate(samples=2 ** 15)
    V, T = sn.remesh(f, verts, faces, max_len=0.1)
    assert_closed_manifold(T)
    assert np.abs(f(V)).max() < 1e-9
    # curvature 1 never asks for less than max_len here, so edges ~ max_len
    assert 0.07 < np.median(edge_lengths(V, T)) < 0.11


@pytest.mark.parametrize('creases', [True, False])
def test_remesh_box(creases):
    f = sn.box(1.5)
    verts, faces = f.generate(samples=2 ** 15, sharp=True)
    V, T = sn.remesh(f, verts, faces, creases=creases)
    assert_closed_manifold(T)
    assert np.abs(f(V)).max() < 1e-9
    vol = signed_volume(V, T)
    if creases:
        assert abs(vol - 1.5 ** 3) < 1e-6          # edges and corners kept exactly
    else:
        assert 0.98 * 1.5 ** 3 < vol < 1.5 ** 3   # slightly rounded


def test_remesh_finer_along_creases():
    f = sn.box(1.5)
    verts, faces = f.generate(samples=2 ** 15, sharp=True)
    V, T = sn.remesh(f, verts, faces, max_len=0.12, crease_len=0.03)
    # edges on the box's 12 edges vs edges in the middle of the faces
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    mid = 0.5 * (V[e[:, 0]] + V[e[:, 1]])
    near_edge = (np.abs(np.abs(mid) - 0.75) < 1e-6).sum(1) >= 2
    centre = (np.abs(mid) < 0.4).sum(1) >= 2
    L = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    assert np.median(L[near_edge]) < 0.5 * np.median(L[centre])


def test_remesh_csg():
    f = sn.sphere(1) & sn.box(1.5)
    c = sn.cylinder(0.5)
    f -= c.orient(sn.X) | c.orient(sn.Y) | c
    verts, faces = f.generate(samples=2 ** 16, sharp=True)
    V, T = sn.remesh(f, verts, faces)
    assert_closed_manifold(T)
    euler = len(V) - len(T) * 3 // 2 + len(T)
    assert euler == 2 - 2 * 5
    assert np.abs(f(V)).max() < 1e-3


# --- generated code and primitive ids ---------------------------------------

from sdfnumba import codegen, interp
from sdfnumba.grid import _eval_points, _eval_ids


@pytest.mark.parametrize('name', CASES)
def test_generated_matches_interpreter(name):
    prog = CASES[name](sn).program()
    run, run_id = codegen.compile(prog)
    a, b = np.empty(len(POINTS)), np.empty(len(POINTS))
    _eval_points(interp.run, prog.code, prog.params, POINTS, a)
    _eval_points(run, prog.code, prog.params, POINTS, b)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    ia, ib = np.empty(len(POINTS), np.int64), np.empty(len(POINTS), np.int64)
    _eval_ids(interp.run_id, prog.code, prog.params, POINTS, ia)
    _eval_ids(run_id, prog.code, prog.params, POINTS, ib)
    np.testing.assert_array_equal(ia, ib)


def test_param_change_reuses_generated_code():
    a = (sn.sphere(1) | sn.box(1.5).k(0.2)).program()
    b = (sn.sphere(0.5) | sn.box(0.7).k(0.1)).program()
    assert codegen.compile(a) == codegen.compile(b)


def test_background_compile_takes_over():
    # a structure no other test builds, so it starts uncompiled
    f = sn.torus(1, 0.2).twist(0.3).translate((0.1, 0.2, 0.3)).shell(0.05)
    prog = f.program()
    first = prog.run
    deadline = time.time() + 120
    while not codegen.ready(prog) and time.time() < deadline:
        time.sleep(0.2)
    assert codegen.ready(prog)
    assert prog.run is not interp.run
    if first is interp.run:       # it really started on the interpreter
        np.testing.assert_allclose(f(POINTS[:100]).ravel(),
                                   sn.evaluate(prog, POINTS[:100]).ravel())


def test_primitive_ids():
    f = sn.sphere(1) | sn.box(1).translate((2, 0, 0)) | sn.torus(1, 0.2).translate((0, 3, 0))
    ids = f.ids([(0, 0, 0.9), (2, 0, 0.4), (1, 3, 0), (-0.9, 0, 0)])
    assert list(ids) == [0, 1, 2, 0]
    g = sn.box(1.5) - sn.sphere(1)
    assert list(g.ids([(0.75, 0.1, 0.1), (0.75, 0.74, 0.74)])) == [1, 0]


# --- 2D profiles and revolve -----------------------------------------------

def lamp_shade():
    """spomso's lamp shade example: an arc revolved into a thin shell."""
    profile = sn.arc(1, np.pi, np.pi + 0.7 * np.pi / 2).translate((1.2, 0, 0))
    return profile.rotate(np.pi / 10, sn.Z).revolve().dilate(0.02).translate((0, 0, 0.3))


def test_arc_distance():
    a = sn.arc(1, 0, np.pi / 2)
    d = a(np.array([[1, 0, 0], [0, 2, 0], [0.5, 0.5, 5], [-1, 0, 0], [0, -1, 0]])).ravel()
    np.testing.assert_allclose(d, [0, 1, 1 - np.sqrt(0.5), np.sqrt(2), np.sqrt(2)], atol=1e-12)


def test_revolve_of_circle_is_torus():
    pts = RNG.uniform(-2, 2, (2000, 3))
    ring = sn.arc(0.3, -np.pi, np.pi).translate((1, 0, 0)).revolve()
    torus = sn.torus(1, 0.3)
    np.testing.assert_allclose(np.abs(ring(pts)), np.abs(torus(pts)), atol=1e-12)


def test_lamp_shade_matches_spomso():
    spomso_geom = pytest.importorskip('spomso.cores.geom_2d')
    shade = spomso_geom.Arc(1, np.pi, np.pi + 0.7 * np.pi / 2)
    shade.axis_revolution(1 + 0.2, -np.pi / 10)
    shade.rounding(0.02)
    shade.rotate(np.pi / 10, (0, 0, 1))
    shade.rotate(np.pi / 2, (1, 0, 0))
    shade.move((0, 0, 0.3))
    pts = RNG.uniform([-1.1, -1.1, -0.6], [1.1, 1.1, 0.6], (3000, 3))
    np.testing.assert_allclose(lamp_shade()(pts).ravel(), shade.create(pts.T.copy()),
                               atol=1e-12)


def test_lamp_shade_mesh_is_closed():
    verts, faces = lamp_shade().generate(samples=2 ** 18)
    assert_closed_manifold(faces)
    assert signed_volume(verts, faces) > 0


def test_remesh_is_deterministic():
    f = sn.sphere(1) & sn.box(1.5)
    f -= sn.cylinder(0.5)
    f.compile()                     # same code path for every run
    verts, faces = f.generate(samples=2 ** 15, sharp=True)
    runs = [sn.remesh(f, verts, faces) for _ in range(3)]
    for V, T in runs[1:]:
        np.testing.assert_array_equal(V, runs[0][0])
        np.testing.assert_array_equal(T, runs[0][1])
