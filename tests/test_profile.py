import numpy as np
import pytest

import sdfnumba as sn
from sdfnumba import api, profile

RNG = np.random.default_rng(3)
POINTS = RNG.uniform(-2.5, 2.5, (4000, 3))

# r in [0, 1], z in [-0.5, 0.5], drawn in the XZ plane
RECT = np.array([[0, 0, -.5], [1, 0, -.5], [1, 0, .5], [0, 0, .5]], float)
CYLINDER = api.rounded_cylinder(1, 0, 1)


def same(a, b, tol=1e-9):
    np.testing.assert_allclose(a(POINTS).ravel(), b(POINTS).ravel(), atol=tol)


def vase():
    """A profile with a hole and an island, in the YZ plane."""
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    outer = np.array([[0, 0, -1], [1.2, 0, -1], [0.8, 0, 0], [1.4, 0, 1], [0, 0, 1]])
    hole = np.stack([0.6 + 0.2 * np.cos(t), 0 * t, -0.5 + 0.2 * np.sin(t)], 1)
    island = np.stack([2.0 + 0.3 * np.cos(t), 0 * t, 0.3 * np.sin(t)], 1)
    return [l[:, [1, 0, 2]] for l in (outer, hole, island)]


def test_polygon_matches_box():
    square = [np.array([[-1, -.5], [1, -.5], [1, .5], [-1, .5]])]
    flat = api.Transform(sn.polygon(square), [('scale_point', (1, 1, 1))])
    P = POINTS.copy()
    P[:, 2] = 0
    box = api.box((2, 1, 10))
    np.testing.assert_allclose(flat(P).ravel(), box(P).ravel(), atol=1e-9)


def test_revolved_rectangle_is_a_cylinder():
    shape, info = sn.revolve_profile([RECT], 'Z')
    same(shape, CYLINDER)
    assert info.moved == 0 and not info.tilted and not info.mirrored


@pytest.mark.parametrize('flip', [False, True])
def test_either_side_of_the_axis(flip):
    shape, _ = sn.revolve_profile([RECT * (-1 if flip else 1, 1, 1)], 'Z')
    same(shape, CYLINDER)


def test_profile_across_the_axis():
    wide = RECT * (2, 1, 1) - (1, 0, 0)          # x from -1 to 1
    shape, info = sn.revolve_profile([wide], 'Z')
    assert info.mirrored
    same(shape, CYLINDER)


def test_tilted_plane_containing_the_axis():
    c, s = np.cos(0.7), np.sin(0.7)
    shape, info = sn.revolve_profile([RECT @ [[c, s, 0], [-s, c, 0], [0, 0, 1]]], 'Z')
    assert info.tilted
    same(shape, CYLINDER)


@pytest.mark.parametrize('axis', 'XY')
def test_other_axes(axis):
    k = 'XYZ'.index(axis)
    perm = {0: [2, 1, 0], 1: [0, 2, 1]}[k]       # swap z with the axis
    shape, _ = sn.revolve_profile([RECT[:, perm]], axis)
    same(shape, api.Transform(CYLINDER, [('matrix', np.eye(3)[perm].ravel())]))


def test_plane_perpendicular_to_axis_is_refused():
    with pytest.raises(ValueError, match='perpendicular'):
        sn.revolve_profile([RECT[:, [0, 2, 1]]], 'Z')


def test_stray_vertex_is_flattened_and_reported():
    noisy = RECT.copy()
    noisy[1, 1] = 0.01
    shape, info = sn.revolve_profile([noisy], 'Z')
    assert info.moved == 1 and info.max_offset == pytest.approx(0.01)
    assert not info.tilted
    same(shape, CYLINDER)


def test_float32_noise_is_not_reported():
    noisy = RECT + RNG.normal(0, 1e-7, RECT.shape)
    _, info = sn.revolve_profile([noisy.astype(np.float32)], 'Z')
    assert info.moved == 0


def test_loops_from_unordered_edges():
    loops = sn.profile_loops(RECT, edges=[[2, 3], [0, 1], [3, 0], [1, 2]])
    same(sn.revolve_profile(loops, 'Z')[0], CYLINDER)


def test_loops_from_points_drop_repeated_closing_point():
    assert len(sn.profile_loops(np.vstack([RECT, RECT[:1]]))[0]) == 4


def test_adjacent_polygons_merge():
    V = np.array([[0, 0, -.5], [.5, 0, -.5], [1, 0, -.5],
                  [1, 0, .5], [.5, 0, .5], [0, 0, .5]], float)
    loops = sn.profile_loops(V, polygons=[[0, 1, 4, 5], [1, 2, 3, 4]])
    assert len(loops) == 1 and len(loops[0]) == 6
    same(sn.revolve_profile(loops, 'Z')[0], CYLINDER)


def test_open_edges_are_refused():
    with pytest.raises(ValueError, match='exactly 2'):
        sn.profile_loops(RECT, edges=[[0, 1], [1, 2], [2, 3]])


def test_holes_and_islands():
    loops = vase()
    shape, info = sn.revolve_profile(loops, 'Z')
    assert info.loops == 3
    d = lambda *p: shape(np.array([p], float))[0]
    assert d(0, 0.2, -0.5) < 0          # in the body
    assert d(0, 0.6, -0.5) > 0          # in the hole's ring
    assert d(0, 2.0, 0) < 0             # in the island's torus
    assert d(0, 1.6, 0) > 0             # between body and island
    # distances are exact: compare with brute force over the 2D outline
    segs = profile._segments([l[:, 1:] for l in loops])
    segs = segs[(segs[:, 0] != 0) | (segs[:, 2] != 0)]      # the axis is no surface
    P = POINTS[:500]
    q = np.stack([np.hypot(P[:, 0], P[:, 1]), P[:, 2]], 1)
    a, b = segs[None, :, :2], segs[None, :, 2:]
    t = np.clip(((q[:, None] - a) * (b - a)).sum(2) / ((b - a) ** 2).sum(2), 0, 1)
    brute = np.linalg.norm(q[:, None] - a - (b - a) * t[..., None], axis=2).min(1)
    np.testing.assert_allclose(np.abs(shape(P).ravel()), brute, atol=1e-9)


def test_meshes_closed():
    shape, _ = sn.revolve_profile(vase(), 'Z')
    V, F = sn.generate(shape, 2 ** 15, triangles=True)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _, counts = np.unique(E, axis=0, return_counts=True)
    assert (counts == 2).all()
