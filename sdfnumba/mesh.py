"""Grid to closed mesh, with Surface Nets.

One vertex per cell whose corners change sign, placed at the average of the
cell's edge crossings.  One quad per grid edge that crosses the surface,
joining the four cells around that edge.  Inside is negative.

The grid's outer layer is forced to "outside", so the mesh is always closed,
even when the bounds cut through the shape (the cut is capped).

With sharp=True the vertices are placed by dual contouring instead (dc.py);
the cells and quads are the same, so the mesh is just as closed.
"""

import time

import numpy as np
from numba import njit, prange

from .grid import estimate_bounds, sample_grid

SAMPLES = 2 ** 22

# cell corners as (dx, dy, dz) and the 12 cell edges as corner pairs
CORNERS = np.array([(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.int64)
EDGES = np.array([(a, b) for a in range(8) for b in range(8)
                  if a < b and bin(a ^ b).count('1') == 1], dtype=np.int64)


@njit(cache=True)
def _cell_mask(v, i, j, k):
    m = 0
    for c in range(8):
        if v[i + CORNERS[c, 0], j + CORNERS[c, 1], k + CORNERS[c, 2]] < 0.0:
            m |= 1 << c
    return m


@njit(parallel=True, cache=True)
def _place_vertices(v):
    nx, ny, nz = v.shape
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    index = np.full((cx, cy, cz), -1, dtype=np.int32)

    counts = np.zeros(cx, dtype=np.int64)
    for i in prange(cx):
        n = 0
        for j in range(cy):
            for k in range(cz):
                m = _cell_mask(v, i, j, k)
                if m != 0 and m != 255:
                    n += 1
        counts[i] = n
    starts = np.zeros(cx + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    verts = np.empty((starts[cx], 3))
    for i in prange(cx):
        n = starts[i]
        for j in range(cy):
            for k in range(cz):
                m = _cell_mask(v, i, j, k)
                if m == 0 or m == 255:
                    continue
                sx = sy = sz = 0.0
                hits = 0
                for e in range(12):
                    a, b = EDGES[e, 0], EDGES[e, 1]
                    ina = (m >> a) & 1
                    inb = (m >> b) & 1
                    if ina == inb:
                        continue
                    va = v[i + CORNERS[a, 0], j + CORNERS[a, 1], k + CORNERS[a, 2]]
                    vb = v[i + CORNERS[b, 0], j + CORNERS[b, 1], k + CORNERS[b, 2]]
                    t = va / (va - vb)
                    sx += CORNERS[a, 0] + t * (CORNERS[b, 0] - CORNERS[a, 0])
                    sy += CORNERS[a, 1] + t * (CORNERS[b, 1] - CORNERS[a, 1])
                    sz += CORNERS[a, 2] + t * (CORNERS[b, 2] - CORNERS[a, 2])
                    hits += 1
                verts[n, 0] = i + sx / hits
                verts[n, 1] = j + sy / hits
                verts[n, 2] = k + sz / hits
                index[i, j, k] = n
                n += 1
    return verts, index


@njit(cache=True)
def _crossing(v, i, j, k, axis):
    """0 = no crossing, 1 = inside->outside going +axis, -1 = the reverse."""
    a = v[i, j, k] < 0.0
    if axis == 0:
        b = v[i + 1, j, k] < 0.0
    elif axis == 1:
        b = v[i, j + 1, k] < 0.0
    else:
        b = v[i, j, k + 1] < 0.0
    if a == b:
        return 0
    return 1 if a else -1


@njit(cache=True)
def _edge_ok(nx, ny, nz, i, j, k, axis):
    # the edge needs all four surrounding cells to exist
    if axis == 0:
        return i < nx - 1 and 0 < j < ny - 1 and 0 < k < nz - 1
    if axis == 1:
        return j < ny - 1 and 0 < i < nx - 1 and 0 < k < nz - 1
    return k < nz - 1 and 0 < i < nx - 1 and 0 < j < ny - 1


@njit(parallel=True, cache=True)
def _make_quads(v, index):
    nx, ny, nz = v.shape
    counts = np.zeros(nx, dtype=np.int64)
    for i in prange(nx):
        n = 0
        for j in range(ny):
            for k in range(nz):
                for axis in range(3):
                    if _edge_ok(nx, ny, nz, i, j, k, axis) and _crossing(v, i, j, k, axis) != 0:
                        n += 1
        counts[i] = n
    starts = np.zeros(nx + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)

    quads = np.empty((starts[nx], 4), dtype=np.int32)
    for i in prange(nx):
        n = starts[i]
        for j in range(ny):
            for k in range(nz):
                for axis in range(3):
                    if not _edge_ok(nx, ny, nz, i, j, k, axis):
                        continue
                    s = _crossing(v, i, j, k, axis)
                    if s == 0:
                        continue
                    if axis == 0:
                        q0 = index[i, j - 1, k - 1]
                        q1 = index[i, j, k - 1]
                        q2 = index[i, j, k]
                        q3 = index[i, j - 1, k]
                    elif axis == 1:
                        q0 = index[i - 1, j, k - 1]
                        q1 = index[i - 1, j, k]
                        q2 = index[i, j, k]
                        q3 = index[i, j, k - 1]
                    else:
                        q0 = index[i - 1, j - 1, k]
                        q1 = index[i, j - 1, k]
                        q2 = index[i, j, k]
                        q3 = index[i - 1, j, k]
                    if s > 0:
                        quads[n, 0], quads[n, 1], quads[n, 2], quads[n, 3] = q0, q1, q2, q3
                    else:
                        quads[n, 0], quads[n, 1], quads[n, 2], quads[n, 3] = q0, q3, q2, q1
                    n += 1
    return quads


def _close_boundary(v):
    """Force the outer layer of grid points outside, so the mesh is closed."""
    eps = np.float32(1e-6)
    for sl in (np.s_[0], np.s_[-1]):
        v[sl] = np.maximum(v[sl], eps)
        v[:, sl] = np.maximum(v[:, sl], eps)
        v[:, :, sl] = np.maximum(v[:, :, sl], eps)


def surface_nets(values, origin, step, program=None):
    """Mesh a grid of distances. Returns (verts (n, 3), quads (m, 4)).
    With `program`, vertices are placed by dual contouring (sharp edges)."""
    if program is None:
        verts, index = _place_vertices(values)
    else:
        from .dc import place_vertices_dc
        verts, index = place_vertices_dc(values, program, origin, step)
    quads = _make_quads(values, index)
    return verts * np.asarray(step) + np.asarray(origin), quads


def triangulate(quads, verts, program=None):
    """Split each quad in two.  With `program`, pick the diagonal whose
    midpoint is closer to the true surface (follows sharp creases);
    otherwise the shorter diagonal."""
    a, b, c, d = quads.T
    p = verts
    if program is None:
        ac = np.einsum('ij,ij->i', p[a] - p[c], p[a] - p[c])
        bd = np.einsum('ij,ij->i', p[b] - p[d], p[b] - p[d])
    else:
        from .grid import evaluate
        ac = np.abs(evaluate(program, (p[a] + p[c]) / 2)).ravel()
        bd = np.abs(evaluate(program, (p[b] + p[d]) / 2)).ravel()
    use_ac = ac <= bd
    t1 = np.where(use_ac[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
    t2 = np.where(use_ac[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
    return np.concatenate([t1, t2]).astype(np.int32)


def generate(sdf, step=None, bounds=None, samples=SAMPLES, sharp=False,
             triangles=False, verbose=False):
    """Mesh a shape. Returns (verts float64 (n, 3), faces int32 (m, 4) or (m, 3)).

    step     grid spacing (one number or three); default from `samples`
    bounds   ((x0, y0, z0), (x1, y1, z1)); default is estimated
    samples  roughly how many grid points to use when step is not given
    sharp    dual contouring: keep sharp edges and corners (slower)
    """
    t0 = time.perf_counter()
    program = sdf.program() if hasattr(sdf, 'program') else sdf
    if bounds is None:
        bounds = estimate_bounds(program)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    if step is None:
        step = (np.prod(hi - lo) / samples) ** (1 / 3)
    step = np.broadcast_to(np.asarray(step, float), (3,)).copy()

    # pad by two cells so the surface sits inside the closed outer layer
    lo -= 2 * step
    hi += 2 * step
    shape = np.floor((hi - lo) / step).astype(int) + 1

    values, skipped = sample_grid(program, lo, step, shape)
    _close_boundary(values)
    verts, faces = surface_nets(values, lo, step, program if sharp else None)
    if triangles:
        faces = triangulate(faces, verts, program if sharp else None)

    if verbose:
        print(f'grid {shape[0]}x{shape[1]}x{shape[2]}, step {step[0]:g}, '
              f'{skipped} blocks skipped, {len(verts)} verts, {len(faces)} faces, '
              f'{time.perf_counter() - t0:.3f}s')
    return verts, faces
