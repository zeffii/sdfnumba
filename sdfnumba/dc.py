"""Dual contouring: vertex placement that keeps sharp edges and corners.

Same cells and quads as Surface Nets (mesh.py); only the vertex position in
each surface cell changes:

  1. for every cell edge that crosses the surface, find the crossing exactly
     (a few false-position steps on the real SDF, not the grid)
  2. take the SDF gradient there as the surface normal
  3. place the vertex at the point that best fits all those tangent planes
     (the QEF), solved relative to the crossings' average with small
     eigenvalues dropped, so flat and curved areas stay smooth
  4. clamp the result to the cell

Cells on the outer grid layer use the plain average, since their crossings
there come from the forced "outside" cap rather than from the shape.

The 3x3 solve uses a small Jacobi eigen-solver, so nothing here needs SciPy.
"""

import math

import numpy as np
from numba import njit, prange

from numba import types

from .interp import STACK_DEPTH
from .mesh import CORNERS, EDGES, _cell_mask
from .sigs import FN, A1, A2, A2i, A3f, A3i32

REFINE_STEPS = 4
EIGEN_CUTOFF = 0.1   # eigenvalues below this fraction of the largest are dropped


@njit(cache=True, fastmath=True)
def _jacobi_eigen(A, V, w):
    """Eigen-decompose the symmetric 3x3 A in place: A -> diag, V = vectors."""
    for r in range(3):
        for c in range(3):
            V[r, c] = 1.0 if r == c else 0.0
    for _ in range(12):
        off = A[0, 1] ** 2 + A[0, 2] ** 2 + A[1, 2] ** 2
        if off < 1e-20:
            break
        for p in range(2):
            for q in range(p + 1, 3):
                if abs(A[p, q]) < 1e-30:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                t = (1.0 if theta >= 0.0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(3):
                    akp, akq = A[k, p], A[k, q]
                    A[k, p], A[k, q] = c * akp - s * akq, s * akp + c * akq
                for k in range(3):
                    apk, aqk = A[p, k], A[q, k]
                    A[p, k], A[q, k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(3):
                    vkp, vkq = V[k, p], V[k, q]
                    V[k, p], V[k, q] = c * vkp - s * vkq, s * vkp + c * vkq
    for k in range(3):
        w[k] = A[k, k]


@njit(types.Tuple((A2, A3i32))(FN, A3f, A2i, A1, A1, A1),
      parallel=True, cache=True, fastmath=True)
def _place_vertices_dc(run, v, code, P, origin, step):
    nx, ny, nz = v.shape
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    index = np.full((cx, cy, cz), -1, dtype=np.int32)
    h = 1e-4 * min(step[0], min(step[1], step[2]))

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
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        ATA = np.empty((3, 3))
        V = np.empty((3, 3))
        w = np.empty(3)
        n = starts[i]
        for j in range(cy):
            for k in range(cz):
                m = _cell_mask(v, i, j, k)
                if m == 0 or m == 255:
                    continue
                # cell corner in world space; positions below are relative to it
                ox = origin[0] + i * step[0]
                oy = origin[1] + j * step[1]
                oz = origin[2] + k * step[2]
                edge_cell = (i == 0 or j == 0 or k == 0 or
                             i == cx - 1 or j == cy - 1 or k == cz - 1)
                for r in range(3):
                    for c in range(3):
                        ATA[r, c] = 0.0
                b0 = b1 = b2 = 0.0
                mx = my = mz = 0.0
                hits = 0
                for e in range(12):
                    a, b = EDGES[e, 0], EDGES[e, 1]
                    if ((m >> a) & 1) == ((m >> b) & 1):
                        continue
                    ax, ay, az = CORNERS[a, 0] * step[0], CORNERS[a, 1] * step[1], CORNERS[a, 2] * step[2]
                    bx, by, bz = CORNERS[b, 0] * step[0], CORNERS[b, 1] * step[1], CORNERS[b, 2] * step[2]
                    da = v[i + CORNERS[a, 0], j + CORNERS[a, 1], k + CORNERS[a, 2]]
                    db = v[i + CORNERS[b, 0], j + CORNERS[b, 1], k + CORNERS[b, 2]]
                    t = da / (da - db)
                    if not edge_cell:
                        # false position on the exact SDF
                        ta, tb, fa, fb = 0.0, 1.0, float(da), float(db)
                        for _ in range(REFINE_STEPS):
                            f = run(code, P, ox + ax + t * (bx - ax), oy + ay + t * (by - ay),
                                     oz + az + t * (bz - az), ps, ds, aux)
                            if (f < 0.0) == (fa < 0.0):
                                ta, fa = t, f
                            else:
                                tb, fb = t, f
                            if fa == fb:
                                break
                            t = ta + (tb - ta) * fa / (fa - fb)
                    px = ax + t * (bx - ax)
                    py = ay + t * (by - ay)
                    pz = az + t * (bz - az)
                    mx += px
                    my += py
                    mz += pz
                    hits += 1
                    if edge_cell:
                        continue
                    wx, wy, wz = ox + px, oy + py, oz + pz
                    gx = (run(code, P, wx + h, wy, wz, ps, ds, aux) -
                          run(code, P, wx - h, wy, wz, ps, ds, aux))
                    gy = (run(code, P, wx, wy + h, wz, ps, ds, aux) -
                          run(code, P, wx, wy - h, wz, ps, ds, aux))
                    gz = (run(code, P, wx, wy, wz + h, ps, ds, aux) -
                          run(code, P, wx, wy, wz - h, ps, ds, aux))
                    gl = math.sqrt(gx * gx + gy * gy + gz * gz)
                    if gl == 0.0:
                        continue
                    gx, gy, gz = gx / gl, gy / gl, gz / gl
                    # accumulate the plane  n . x = n . p
                    d = gx * px + gy * py + gz * pz
                    ATA[0, 0] += gx * gx
                    ATA[0, 1] += gx * gy
                    ATA[0, 2] += gx * gz
                    ATA[1, 1] += gy * gy
                    ATA[1, 2] += gy * gz
                    ATA[2, 2] += gz * gz
                    b0 += gx * d
                    b1 += gy * d
                    b2 += gz * d
                mx /= hits
                my /= hits
                mz /= hits
                qx, qy, qz = mx, my, mz
                if not edge_cell:
                    ATA[1, 0] = ATA[0, 1]
                    ATA[2, 0] = ATA[0, 2]
                    ATA[2, 1] = ATA[1, 2]
                    # residual at the mass point: r = ATb - ATA m
                    r0 = b0 - (ATA[0, 0] * mx + ATA[0, 1] * my + ATA[0, 2] * mz)
                    r1 = b1 - (ATA[1, 0] * mx + ATA[1, 1] * my + ATA[1, 2] * mz)
                    r2 = b2 - (ATA[2, 0] * mx + ATA[2, 1] * my + ATA[2, 2] * mz)
                    _jacobi_eigen(ATA, V, w)
                    wmax = max(abs(w[0]), max(abs(w[1]), abs(w[2])))
                    for c in range(3):
                        if wmax == 0.0 or abs(w[c]) < EIGEN_CUTOFF * wmax:
                            continue
                        # project the residual onto this eigenvector, scale by 1/w
                        s = (V[0, c] * r0 + V[1, c] * r1 + V[2, c] * r2) / w[c]
                        qx += s * V[0, c]
                        qy += s * V[1, c]
                        qz += s * V[2, c]
                    qx = min(max(qx, 0.0), step[0])
                    qy = min(max(qy, 0.0), step[1])
                    qz = min(max(qz, 0.0), step[2])
                # back to grid units, as Surface Nets returns them
                verts[n, 0] = i + qx / step[0]
                verts[n, 1] = j + qy / step[1]
                verts[n, 2] = k + qz / step[2]
                index[i, j, k] = n
                n += 1
    return verts, index


def place_vertices_dc(values, program, origin, step):
    return _place_vertices_dc(program.run, values, program.code, program.params,
                              np.asarray(origin, dtype=np.float64),
                              np.asarray(step, dtype=np.float64))
