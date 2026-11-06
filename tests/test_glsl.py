"""The viewer's GLSL against the numba code, on a (software) GPU."""

import numpy as np
import pytest

import sdfnumba as sn
from sdfnumba import glsl, ops
from test_sdfnumba import CASES, POINTS, lamp_shade

GLSL_CASES = dict(CASES, lamp_shade=lambda m: lamp_shade())

import gl_harness

CTX = gl_harness.context()
needs_gl = pytest.mark.skipif(CTX is None, reason='no OpenGL context available')


def test_every_op_has_glsl():
    assert set(ops.REGISTRY) <= set(glsl.OPS)
    assert {op.winner for op in ops.REGISTRY.values() if op.winner} <= set(glsl.WINNERS)


@needs_gl
@pytest.mark.parametrize('name', GLSL_CASES)
def test_glsl_matches_numba(name):
    f = GLSL_CASES[name](sn)
    prog = f.program()
    d, ids = gl_harness.evaluate(CTX, prog, POINTS)
    np.testing.assert_allclose(d, f(POINTS).ravel(), rtol=2e-4, atol=2e-4)
    # ids can legitimately differ where two primitives tie (to float32 precision)
    agree = (ids == sn.primitive_ids(prog, POINTS)) | (prog.prim_number[ids] ==
                                                       sn.primitive_ids(prog, POINTS))
    assert agree.mean() > 0.995


@needs_gl
def test_raymarched_surface_is_on_the_shape():
    f = sn.sphere(1) & sn.box(1.5)
    c = sn.cylinder(0.5)
    f -= c.orient(sn.X) | c.orient(sn.Y) | c
    prog = f.program()
    lo, hi = sn.estimate_bounds(prog)
    rgba, depth, vp = gl_harness.render(CTX, prog, (lo, hi), show_outline=0.0)
    hit = depth < 1.0
    assert 0.15 < hit.mean() < 0.8
    world = gl_harness.unproject(depth, vp)[hit]
    # the depth written for each pixel puts it on the surface
    assert np.percentile(np.abs(f(world)), 99) < 0.02
