"""Evaluating a program: at arbitrary points, and over a regular grid.

The grid is sampled in blocks.  A block is skipped (filled with its centre
distance) when the centre is further from the surface than the block's
half-diagonal plus one cell diagonal: no surface can then cross any grid edge
that touches the block, so the filled values only need the right sign.
"""

import math

import numpy as np
from numba import njit, prange

from .interp import STACK_DEPTH
from .sigs import FN, FN_ID, A1, A2, A2i, A3f, A1i, i64, f64
from numba import types

CHUNK = 4096
BLOCK = 8


@njit(types.void(FN, A2i, A1, A2, A1), parallel=True, cache=True)
def _eval_points(run, code, P, pts, out):
    n = pts.shape[0]
    nchunks = (n + CHUNK - 1) // CHUNK
    for c in prange(nchunks):
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        for i in range(c * CHUNK, min(n, (c + 1) * CHUNK)):
            out[i] = run(code, P, pts[i, 0], pts[i, 1], pts[i, 2], ps, ds, aux)


@njit(types.void(FN_ID, A2i, A1, A2, A1i), parallel=True, cache=True)
def _eval_ids(run_id, code, P, pts, out):
    n = pts.shape[0]
    nchunks = (n + CHUNK - 1) // CHUNK
    for c in prange(nchunks):
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        ids = np.empty(STACK_DEPTH, dtype=np.int64)
        for i in range(c * CHUNK, min(n, (c + 1) * CHUNK)):
            out[i] = run_id(code, P, pts[i, 0], pts[i, 1], pts[i, 2], ps, ds, aux, ids)


def _points(points):
    return np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)


def evaluate(program, points):
    """Distances at an (n, 3) array of points, as an (n, 1) array."""
    pts = _points(points)
    out = np.empty(len(pts))
    _eval_points(program.run, program.code, program.params, pts, out)
    return out.reshape(-1, 1)


def primitive_ids(program, points):
    """For each point, which primitive the surface there belongs to: the
    index of the primitive in the shape (0, 1, 2 ... in the order they were
    built).  Where it changes across a surface there is a boolean seam."""
    pts = _points(points)
    out = np.empty(len(pts), dtype=np.int64)
    _eval_ids(program.run_id, program.code, program.params, pts, out)
    return program.prim_number[out]


@njit(i64(FN, A2i, A1, A1, A1, i64, i64, i64, i64, A3f), parallel=True, cache=True)
def _sample_grid(run, code, P, origin, step, nx, ny, nz, block, out):
    bx = (nx + block - 1) // block
    by = (ny + block - 1) // block
    bz = (nz + block - 1) // block
    margin = math.sqrt(step[0] ** 2 + step[1] ** 2 + step[2] ** 2)
    skipped = np.zeros(bx * by * bz, dtype=np.uint8)
    for b in prange(bx * by * bz):
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        i0 = (b // (by * bz)) * block
        j0 = ((b // bz) % by) * block
        k0 = (b % bz) * block
        i1 = min(i0 + block, nx)
        j1 = min(j0 + block, ny)
        k1 = min(k0 + block, nz)
        # block centre and half-diagonal, in world space
        cx = origin[0] + 0.5 * (i0 + i1 - 1) * step[0]
        cy = origin[1] + 0.5 * (j0 + j1 - 1) * step[1]
        cz = origin[2] + 0.5 * (k0 + k1 - 1) * step[2]
        hx = 0.5 * (i1 - i0 - 1) * step[0]
        hy = 0.5 * (j1 - j0 - 1) * step[1]
        hz = 0.5 * (k1 - k0 - 1) * step[2]
        d = run(code, P, cx, cy, cz, ps, ds, aux)
        if abs(d) > math.sqrt(hx * hx + hy * hy + hz * hz) + margin:
            skipped[b] = 1
            for i in range(i0, i1):
                for j in range(j0, j1):
                    for k in range(k0, k1):
                        out[i, j, k] = d
            continue
        for i in range(i0, i1):
            x = origin[0] + i * step[0]
            for j in range(j0, j1):
                y = origin[1] + j * step[1]
                for k in range(k0, k1):
                    z = origin[2] + k * step[2]
                    out[i, j, k] = run(code, P, x, y, z, ps, ds, aux)
    return skipped.sum()


def sample_grid(program, origin, step, shape, block=BLOCK):
    """Distances on a regular grid of `shape` points starting at `origin`.
    Returns (values float32 (nx, ny, nz), number of skipped blocks)."""
    nx, ny, nz = (int(n) for n in shape)
    out = np.empty((nx, ny, nz), dtype=np.float32)
    skipped = _sample_grid(program.run, program.code, program.params,
                           np.asarray(origin, dtype=np.float64),
                           np.asarray(step, dtype=np.float64),
                           nx, ny, nz, block, out)
    return out, int(skipped)


def estimate_bounds(program):
    """Shrink a huge box onto the surface (same approach as fogleman's sdf)."""
    s = 16
    lo = np.full(3, -1e9)
    hi = np.full(3, 1e9)
    prev = None
    for _ in range(32):
        axes = [np.linspace(lo[i], hi[i], s) for i in range(3)]
        d = np.array([a[1] - a[0] for a in axes])
        threshold = np.linalg.norm(d) / 2
        if threshold == prev:
            break
        prev = threshold
        P = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        volume = evaluate(program, P).reshape(s, s, s)
        where = np.argwhere(np.abs(volume) <= threshold)
        if len(where) == 0:
            raise ValueError('could not find the surface; pass bounds=')
        hi = lo + where.max(axis=0) * d + d / 2
        lo = lo + where.min(axis=0) * d - d / 2
    return tuple(lo), tuple(hi)
