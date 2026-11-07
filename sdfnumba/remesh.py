"""Adaptive isotropic remeshing against the exact SDF (Botsch & Kobbelt 2004,
with the curvature-adaptive sizing of Dunyach et al. 2013).

Each iteration:
  1. sizing   target edge length per vertex, from the SDF's curvature, made
              smaller along creases and graded so it grows smoothly away
  2. split    edges longer than 4/3 of their target
  3. collapse edges shorter than 4/5 of their target
  4. flip     edges where that brings valences closer to 6
  5. relax    move vertices to their neighbours' centroid, within the
              tangent plane (crease vertices only along their crease,
              corners stay put), then project onto the SDF surface

Topology work happens in batches: an operation locks the vertices it
touches, so every operation in a batch sees an up-to-date neighbourhood.
Adjacency is rebuilt between batches.  The mesh must be closed and
manifold, which is what generate() makes.
"""

import math

import numpy as np
from numba import njit, prange

from numba import types

from .interp import STACK_DEPTH
from .dc import _jacobi_eigen
from .sigs import FN, FN_ID, A1, A2, A2i, A1i, A1b, f64, i64, b1


@njit(cache=True)
def _normal(V, i, j, k):
    ux, uy, uz = V[j, 0] - V[i, 0], V[j, 1] - V[i, 1], V[j, 2] - V[i, 2]
    vx, vy, vz = V[k, 0] - V[i, 0], V[k, 1] - V[i, 1], V[k, 2] - V[i, 2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return nx, ny, nz


# --- per-edge tests, evaluated only for edges an operation is considering ----

SEAM_COS_NB = math.cos(math.radians(5.0))
EDGE_SIDE = 0.1   # how far from an edge (as a share of the way across) crease tests look

@njit(cache=True, fastmath=True)
def _side_grad(run, run_id, code, P, V, a, b, c, h, ps, ds, aux, idstack):
    """Unit SDF gradient just beside edge a-b, on the side of the face whose
    far corner is c, and the primitive there.  Close to the edge on purpose:
    across a real crease the normal jumps even over a short distance, while
    on a tight but smooth curve (a rounded rim) it barely turns."""
    mx = 0.5 * (V[a, 0] + V[b, 0])
    my = 0.5 * (V[a, 1] + V[b, 1])
    mz = 0.5 * (V[a, 2] + V[b, 2])
    x = mx + EDGE_SIDE * (V[c, 0] - mx)
    y = my + EDGE_SIDE * (V[c, 1] - my)
    z = mz + EDGE_SIDE * (V[c, 2] - mz)
    gx = run(code, P, x + h, y, z, ps, ds, aux) - run(code, P, x - h, y, z, ps, ds, aux)
    gy = run(code, P, x, y + h, z, ps, ds, aux) - run(code, P, x, y - h, z, ps, ds, aux)
    gz = run(code, P, x, y, z + h, ps, ds, aux) - run(code, P, x, y, z - h, ps, ds, aux)
    gl = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-300
    pid = run_id(code, P, x, y, z, ps, ds, aux, idstack)
    return gx / gl, gy / gl, gz / gl, pid


@njit(cache=True, fastmath=True)
def _edge_turn(run, run_id, code, P, V, a, b, c, d, h, ps, ds, aux, idstack):
    """Across edge a-b (far corners c and d): cos of the turn of the surface
    normal, and whether a different primitive is on each side."""
    n0 = _side_grad(run, run_id, code, P, V, a, b, c, h, ps, ds, aux, idstack)
    n1 = _side_grad(run, run_id, code, P, V, a, b, d, h, ps, ds, aux, idstack)
    return n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2], n0[3] != n1[3]


@njit(cache=True, fastmath=True)
def _is_crease_edge(run, run_id, code, P, V, F, a, b, f0, f1, vcrease, h, crease_cos,
                    ps, ds, aux, idstack):
    """Both ends on a crease, and across the edge the SDF normal turns sharply
    (or two primitives meet at a slight angle, see _crease_info)."""
    if crease_cos < -1.5 or vcrease[a] == 0 or vcrease[b] == 0:
        return False
    c = d = -1
    for s in range(3):
        u = F[f0, s]
        if u != a and u != b:
            c = u
        u = F[f1, s]
        if u != a and u != b:
            d = u
    cos, seam = _edge_turn(run, run_id, code, P, V, a, b, c, d, h, ps, ds, aux, idstack)
    return cos < crease_cos or (seam and cos < SEAM_COS_NB)


@njit(cache=True, fastmath=True)
def _unit_grad(run, code, P, x, y, z, h, ps, ds, aux):
    gx = run(code, P, x + h, y, z, ps, ds, aux) - run(code, P, x - h, y, z, ps, ds, aux)
    gy = run(code, P, x, y + h, z, ps, ds, aux) - run(code, P, x, y - h, z, ps, ds, aux)
    gz = run(code, P, x, y, z + h, ps, ds, aux) - run(code, P, x, y, z - h, ps, ds, aux)
    gl = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-300
    return gx / gl, gy / gl, gz / gl


@njit(cache=True, fastmath=True)
def _faces_out(run, code, P, V, i, j, k, h, ps, ds, aux):
    """Does triangle i, j, k face the same way as the surface at its centre?"""
    nx, ny, nz = _normal(V, i, j, k)
    cx = (V[i, 0] + V[j, 0] + V[k, 0]) / 3.0
    cy = (V[i, 1] + V[j, 1] + V[k, 1]) / 3.0
    cz = (V[i, 2] + V[j, 2] + V[k, 2]) / 3.0
    gx, gy, gz = _unit_grad(run, code, P, cx, cy, cz, h, ps, ds, aux)
    return nx * gx + ny * gy + nz * gz > 0.0


@njit(cache=True, fastmath=True)
def _cuts_crease(run, code, P, V, a, b, h, crease_cos, ps, ds, aux):
    """Does edge a-b run across a crease?  Compares the surface normal a
    little either side of its midpoint, along the edge: it jumps across a
    crease however short that step, while on a smooth curve, even a tight
    one, it barely turns.  (Testing how far the midpoint sags off the
    surface instead would flag every edge on a tightly curved rim.)"""
    if crease_cos < -1.5:
        return False
    mx = 0.5 * (V[a, 0] + V[b, 0])
    my = 0.5 * (V[a, 1] + V[b, 1])
    mz = 0.5 * (V[a, 2] + V[b, 2])
    tx = EDGE_SIDE * (V[b, 0] - V[a, 0])
    ty = EDGE_SIDE * (V[b, 1] - V[a, 1])
    tz = EDGE_SIDE * (V[b, 2] - V[a, 2])
    n0 = _unit_grad(run, code, P, mx - tx, my - ty, mz - tz, h, ps, ds, aux)
    n1 = _unit_grad(run, code, P, mx + tx, my + ty, mz + tz, h, ps, ds, aux)
    return n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2] < crease_cos


# --- SDF queries ---------------------------------------------------------

@njit(cache=True, fastmath=True)
def _grad(run, code, P, x, y, z, h, ps, ds, aux):
    gx = (run(code, P, x + h, y, z, ps, ds, aux) - run(code, P, x - h, y, z, ps, ds, aux)) / (2 * h)
    gy = (run(code, P, x, y + h, z, ps, ds, aux) - run(code, P, x, y - h, z, ps, ds, aux)) / (2 * h)
    gz = (run(code, P, x, y, z + h, ps, ds, aux) - run(code, P, x, y, z - h, ps, ds, aux)) / (2 * h)
    return gx, gy, gz


@njit(types.void(FN, A2i, A1, A2, A2, f64), parallel=True, cache=True, nogil=True,
      fastmath=True)
def _normals(run, code, P, V, out, h):
    """Unit SDF gradient at each point."""
    for i in prange(V.shape[0]):
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        gx, gy, gz = _grad(run, code, P, V[i, 0], V[i, 1], V[i, 2], h, ps, ds, aux)
        gl = math.sqrt(gx * gx + gy * gy + gz * gz)
        if gl == 0.0:
            out[i, 0] = out[i, 1] = out[i, 2] = 0.0
        else:
            out[i, 0], out[i, 1], out[i, 2] = gx / gl, gy / gl, gz / gl


@njit(types.void(FN, A2i, A1, A2, A1b, A2, f64, i64), parallel=True, cache=True,
      nogil=True, fastmath=True)
def _project(run, code, P, V, mask, ref, h, steps):
    """Newton steps onto the zero level: p -= d * g / |g|^2.

    Guarded: `ref` is the normal the vertex should end up facing (zero: no
    guard).  Where a part of the shape is thinner than the mesh is fine, the
    nearest surface can be the opposite side; if the vertex lands on a
    surface facing away from `ref`, it is projected along `ref` instead, and
    if that fails too it stays where it was."""
    for i in prange(V.shape[0]):
        if not mask[i]:
            continue
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        x0, y0, z0 = V[i, 0], V[i, 1], V[i, 2]
        x, y, z = x0, y0, z0
        for _ in range(steps):
            d = run(code, P, x, y, z, ps, ds, aux)
            gx, gy, gz = _grad(run, code, P, x, y, z, h, ps, ds, aux)
            g2 = gx * gx + gy * gy + gz * gz
            if g2 < 1e-12:
                break
            x -= d * gx / g2
            y -= d * gy / g2
            z -= d * gz / g2
        rx, ry, rz = ref[i, 0], ref[i, 1], ref[i, 2]
        if rx * rx + ry * ry + rz * rz > 0.0:
            gx, gy, gz = _grad(run, code, P, x, y, z, h, ps, ds, aux)
            if gx * rx + gy * ry + gz * rz <= 0.0:
                # wrong side: search along the reference normal instead
                x, y, z = x0, y0, z0
                for _ in range(2 * steps):
                    d = run(code, P, x, y, z, ps, ds, aux)
                    gr = (run(code, P, x + h * rx, y + h * ry, z + h * rz, ps, ds, aux) -
                          run(code, P, x - h * rx, y - h * ry, z - h * rz, ps, ds, aux)) / (2 * h)
                    if gr <= 1e-6:
                        break
                    x -= d / gr * rx
                    y -= d / gr * ry
                    z -= d / gr * rz
                gx, gy, gz = _grad(run, code, P, x, y, z, h, ps, ds, aux)
                if gx * rx + gy * ry + gz * rz <= 0.0:
                    x, y, z = x0, y0, z0
        V[i, 0], V[i, 1], V[i, 2] = x, y, z


@njit(A1(FN, A2i, A1, A2, f64, f64), parallel=True, cache=True, nogil=True, fastmath=True)
def _local_size(run, code, P, V, h, reach):
    """How thin the shape is at each vertex: the smaller of its wall
    thickness (probing inwards along the normal) and the gap to the next
    surface outside (probing outwards), both found as twice the largest
    distance value met along the probe.  Capped at 2 * reach.

    The normal is measured over a large step (h): on a convex crease a tight
    normal belongs to one face only, and probing along it would slide along
    the other face and read a wall of no thickness; the wide one is the
    bisector, which points properly into (or out of) the material."""
    n = V.shape[0]
    out = np.empty(n)
    for i in prange(n):
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        x, y, z = V[i, 0], V[i, 1], V[i, 2]
        gx, gy, gz = _grad(run, code, P, x, y, z, h, ps, ds, aux)
        gl = math.sqrt(gx * gx + gy * gy + gz * gz)
        if gl == 0.0:
            out[i] = 2 * reach
            continue
        nx, ny, nz = gx / gl, gy / gl, gz / gl
        inside = outside = 0.0
        for k in range(1, 17):
            t = reach * k / 16.0
            inside = max(inside, -run(code, P, x - t * nx, y - t * ny, z - t * nz, ps, ds, aux))
            outside = max(outside, run(code, P, x + t * nx, y + t * ny, z + t * nz, ps, ds, aux))
        out[i] = 2.0 * min(inside, outside)
    return out


@njit(types.void(FN, A2i, A1, A2, A2, A2i, A1i, A1i, A1b, A1i, f64),
      parallel=True, cache=True, nogil=True, fastmath=True)
def _snap_step(run, code, P, V, out, F, vf_start, vf, mask, rank, h):
    """One snapping step: reads V, writes the moved positions to out."""
    for i in prange(V.shape[0]):
        if not mask[i]:
            continue
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        A = np.empty((3, 3))
        Q = np.empty((3, 3))
        w = np.empty(3)
        x, y, z = V[i, 0], V[i, 1], V[i, 2]
        for r in range(3):
            for c in range(3):
                A[r, c] = 0.0
        b0 = b1 = b2 = 0.0
        reach = 1e300
        for k in range(vf_start[i], vf_start[i + 1]):
            f = vf[k]
            cx = (V[F[f, 0], 0] + V[F[f, 1], 0] + V[F[f, 2], 0]) / 3.0
            cy = (V[F[f, 0], 1] + V[F[f, 1], 1] + V[F[f, 2], 1]) / 3.0
            cz = (V[F[f, 0], 2] + V[F[f, 1], 2] + V[F[f, 2], 2]) / 3.0
            reach = min(reach, math.sqrt((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2))
            sx, sy, sz = x + 0.5 * (cx - x), y + 0.5 * (cy - y), z + 0.5 * (cz - z)
            d = run(code, P, sx, sy, sz, ps, ds, aux)
            gx = run(code, P, sx + h, sy, sz, ps, ds, aux) - run(code, P, sx - h, sy, sz, ps, ds, aux)
            gy = run(code, P, sx, sy + h, sz, ps, ds, aux) - run(code, P, sx, sy - h, sz, ps, ds, aux)
            gz = run(code, P, sx, sy, sz + h, ps, ds, aux) - run(code, P, sx, sy, sz - h, ps, ds, aux)
            gl = math.sqrt(gx * gx + gy * gy + gz * gz)
            if gl == 0.0:
                continue
            gx, gy, gz = gx / gl, gy / gl, gz / gl
            # plane: g . p = g . s - d   (the SDF's zero level, locally)
            # residual measured from the current vertex
            rhs = gx * (sx - x) + gy * (sy - y) + gz * (sz - z) - d
            A[0, 0] += gx * gx
            A[0, 1] += gx * gy
            A[0, 2] += gx * gz
            A[1, 1] += gy * gy
            A[1, 2] += gy * gz
            A[2, 2] += gz * gz
            b0 += gx * rhs
            b1 += gy * rhs
            b2 += gz * rhs
        A[1, 0], A[2, 0], A[2, 1] = A[0, 1], A[0, 2], A[1, 2]
        _jacobi_eigen(A, Q, w)
        wmax = max(abs(w[0]), max(abs(w[1]), abs(w[2])))
        # rank of each eigenvalue by size (0 = largest)
        mx = my = mz = 0.0
        for c in range(3):
            if wmax == 0.0 or abs(w[c]) < 0.05 * wmax:
                continue
            order = 0
            for c2 in range(3):
                if abs(w[c2]) > abs(w[c]) or (abs(w[c2]) == abs(w[c]) and c2 < c):
                    order += 1
            if order >= rank[i]:
                continue
            t = (Q[0, c] * b0 + Q[1, c] * b1 + Q[2, c] * b2) / w[c]
            mx += t * Q[0, c]
            my += t * Q[1, c]
            mz += t * Q[2, c]
        ml = math.sqrt(mx * mx + my * my + mz * mz)
        if ml > reach:          # never jump past the neighbourhood
            mx, my, mz = mx * reach / ml, my * reach / ml, mz * reach / ml
        out[i, 0], out[i, 1], out[i, 2] = x + mx, y + my, z + mz


@njit(types.void(FN, A2i, A1, A2, A2i, A1i, A1i, A1b, A1i, f64, i64),
      cache=True, nogil=True, fastmath=True)
def _project_features(run, code, P, V, F, vf_start, vf, mask, rank, h, steps):
    """Snap vertices onto creases and corners.

    Each face around the vertex gives a sample point a little way into that
    face; the SDF value and gradient there define that side's tangent plane.
    The vertex moves to the point nearest to it that lies on all those planes
    (a small QEF, as in dual contouring): two distinct planes put it on the
    crease line, three put it on the corner, one is a plain projection.

    Each step reads every position from `src` and writes to `dst`, so no
    vertex sees a neighbour half-way through its own move: the result
    doesn't depend on how the threads happen to be scheduled."""
    src = V.copy()
    dst = V
    for _ in range(steps):
        _snap_step(run, code, P, src, dst, F, vf_start, vf, mask, rank, h)
        src[:] = dst


@njit(A1(FN, A2i, A1, A2, f64), parallel=True, cache=True, nogil=True, fastmath=True)
def _curvature(run, code, P, V, h):
    """Largest absolute principal curvature at each vertex, from the SDF's
    Hessian restricted to the tangent plane.  `h` sets the scale it sees."""
    n = V.shape[0]
    out = np.empty(n)
    for i in prange(n):
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        H = np.empty((3, 3))
        T = np.empty((3, 3))
        Q = np.empty((3, 3))
        w = np.empty(3)
        x, y, z = V[i, 0], V[i, 1], V[i, 2]
        f0 = run(code, P, x, y, z, ps, ds, aux)
        g = np.empty(3)
        e = np.zeros(3)
        for a in range(3):
            e[:] = 0.0
            e[a] = h
            fp = run(code, P, x + e[0], y + e[1], z + e[2], ps, ds, aux)
            fm = run(code, P, x - e[0], y - e[1], z - e[2], ps, ds, aux)
            g[a] = (fp - fm) / (2 * h)
            H[a, a] = (fp - 2 * f0 + fm) / (h * h)
        for a in range(3):
            for b in range(a + 1, 3):
                ea = np.zeros(3)
                eb = np.zeros(3)
                ea[a] = h
                eb[b] = h
                fpp = run(code, P, x + ea[0] + eb[0], y + ea[1] + eb[1], z + ea[2] + eb[2], ps, ds, aux)
                fpm = run(code, P, x + ea[0] - eb[0], y + ea[1] - eb[1], z + ea[2] - eb[2], ps, ds, aux)
                fmp = run(code, P, x - ea[0] + eb[0], y - ea[1] + eb[1], z - ea[2] + eb[2], ps, ds, aux)
                fmm = run(code, P, x - ea[0] - eb[0], y - ea[1] - eb[1], z - ea[2] - eb[2], ps, ds, aux)
                H[a, b] = H[b, a] = (fpp - fpm - fmp + fmm) / (4 * h * h)
        gl = math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
        if gl < 1e-12:
            out[i] = 0.0
            continue
        nn = g / gl
        # T = (I - n n^T) H (I - n n^T) / |g|
        for r in range(3):
            for c in range(3):
                Q[r, c] = (1.0 if r == c else 0.0) - nn[r] * nn[c]
        for r in range(3):
            for c in range(3):
                s = 0.0
                for k in range(3):
                    s += H[r, k] * Q[k, c]
                T[r, c] = s
        for r in range(3):
            for c in range(3):
                s = 0.0
                for k in range(3):
                    s += Q[r, k] * T[k, c]
                H[r, c] = s / gl
        _jacobi_eigen(H, T, w)
        out[i] = max(abs(w[0]), max(abs(w[1]), abs(w[2])))
    return out


# --- adjacency -------------------------------------------------------------

# --- triangle shape -------------------------------------------------------------
#
# Two kinds of bad triangle:
#   needle  one edge much shorter than the other two (shortest / longest small)
#           -> collapse the short edge
#   cap     the vertex opposite the longest edge lies almost on it
#           (height over the longest edge / its length small) -> flip that
#           edge; where it is a crease edge, only if the cap is completely
#           flat (its apex on the same crease)

OPEN_COS = math.cos(math.radians(15.0))   # open-surface flips: triangles this flat
NEEDLE_RATIO = 0.3      # shortest / longest edge below this: a needle
CAP_RATIO = 0.25        # height / longest edge below this: a cap


@njit(cache=True)
def _edge_len(V, i, j):
    return math.sqrt((V[i, 0] - V[j, 0]) ** 2 + (V[i, 1] - V[j, 1]) ** 2 +
                     (V[i, 2] - V[j, 2]) ** 2)


@njit(cache=True)
def _min_angle(V, i, j, k):
    """Smallest angle of triangle i, j, k, in radians."""
    a, b, c = _edge_len(V, j, k), _edge_len(V, k, i), _edge_len(V, i, j)
    if a == 0.0 or b == 0.0 or c == 0.0:
        return 0.0
    # the smallest angle is opposite the shortest edge
    if a <= b and a <= c:
        cos = (b * b + c * c - a * a) / (2 * b * c)
    elif b <= c:
        cos = (a * a + c * c - b * b) / (2 * a * c)
    else:
        cos = (a * a + b * b - c * c) / (2 * a * b)
    return math.acos(min(max(cos, -1.0), 1.0))


@njit(cache=True)
def _is_needle_edge(V, a, b, c):
    """Is a-b the short edge of a needle a, b, c?"""
    ab, bc, ca = _edge_len(V, a, b), _edge_len(V, b, c), _edge_len(V, c, a)
    return ab <= bc and ab <= ca and ab < NEEDLE_RATIO * max(bc, ca)


@njit(cache=True)
def _cap_apex_t(V, a, b, c):
    """If a, b, c is a cap with a-b as its longest edge, where along a-b
    (0..1) its apex c sits; otherwise -1."""
    ab, bc, ca = _edge_len(V, a, b), _edge_len(V, b, c), _edge_len(V, c, a)
    if ab < bc or ab < ca or ab == 0.0:
        return -1.0
    nx, ny, nz = _normal(V, a, b, c)
    height = math.sqrt(nx * nx + ny * ny + nz * nz) / ab
    if height >= CAP_RATIO * ab:
        return -1.0
    t = ((V[c, 0] - V[a, 0]) * (V[b, 0] - V[a, 0]) + (V[c, 1] - V[a, 1]) * (V[b, 1] - V[a, 1]) +
         (V[c, 2] - V[a, 2]) * (V[b, 2] - V[a, 2])) / (ab * ab)
    return min(max(t, 0.0), 1.0)



@njit(cache=True, nogil=True)
def _build(V, F):
    """Edges, valence, vertex->face lists, edge lengths and dihedral cosines.
    Edges are matched by bucketing half-edges on their smaller vertex."""
    n, m = V.shape[0], F.shape[0]
    count = np.zeros(n + 1, dtype=np.int64)
    for f in range(m):
        for s in range(3):
            a, b = F[f, s], F[f, (s + 1) % 3]
            count[min(a, b) + 1] += 1
    start = np.cumsum(count)
    fill = start[:-1].copy()
    hb = np.empty(3 * m, dtype=np.int64)      # half-edge id, bucketed
    for f in range(m):
        for s in range(3):
            a, b = F[f, s], F[f, (s + 1) % 3]
            lo = min(a, b)
            hb[fill[lo]] = 3 * f + s
            fill[lo] += 1
    E = np.empty((3 * m // 2, 6), dtype=np.int64)
    ne = 0
    ok = True
    for lo in range(n):
        s0, s1 = start[lo], start[lo + 1]
        for x in range(s0, s1):
            h = hb[x]
            if h < 0:
                continue
            f, s = h // 3, h % 3
            a, b = F[f, s], F[f, (s + 1) % 3]
            found = -1
            for y in range(x + 1, s1):
                g = hb[y]
                if g < 0:
                    continue
                f2, s2 = g // 3, g % 3
                if F[f2, s2] == b and F[f2, (s2 + 1) % 3] == a:
                    found = y
                    break
            if found < 0 or ne >= E.shape[0]:
                ok = False
                continue
            g = hb[found]
            hb[found] = -1
            f2, s2 = g // 3, g % 3
            E[ne, 0], E[ne, 1], E[ne, 2], E[ne, 3] = a, b, f, f2
            E[ne, 4] = F[f, (s + 2) % 3]
            E[ne, 5] = F[f2, (s2 + 2) % 3]
            ne += 1
    valence = np.zeros(n, dtype=np.int64)
    for e in range(ne):
        valence[E[e, 0]] += 1
        valence[E[e, 1]] += 1
    vf_start = np.zeros(n + 1, dtype=np.int64)
    for f in range(m):
        for s in range(3):
            vf_start[F[f, s] + 1] += 1
    vf_start = np.cumsum(vf_start)
    fill2 = vf_start[:-1].copy()
    vf = np.empty(3 * m, dtype=np.int64)
    for f in range(m):
        for s in range(3):
            v = F[f, s]
            vf[fill2[v]] = f
            fill2[v] += 1
    nrm = np.empty((m, 3))
    for f in range(m):
        nx, ny, nz = _normal(V, F[f, 0], F[f, 1], F[f, 2])
        l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-300
        nrm[f, 0], nrm[f, 1], nrm[f, 2] = nx / l, ny / l, nz / l
    edge_len = np.empty(ne)
    dih = np.empty(ne)
    for e in range(ne):
        a, b, f0, f1 = E[e, 0], E[e, 1], E[e, 2], E[e, 3]
        edge_len[e] = math.sqrt((V[a, 0] - V[b, 0]) ** 2 + (V[a, 1] - V[b, 1]) ** 2 +
                                (V[a, 2] - V[b, 2]) ** 2)
        dih[e] = nrm[f0, 0] * nrm[f1, 0] + nrm[f0, 1] * nrm[f1, 1] + nrm[f0, 2] * nrm[f1, 2]
    return ok and ne * 2 == 3 * m, E[:ne], valence, vf_start, vf, edge_len, dih


class Mesh:
    """Faces (m, 3) plus the adjacency derived from them."""

    def __init__(self, V, F):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        ok, self.E, self.valence, self.vf_start, self.vf, self.edge_len, \
            self.dihedral_cos = _build(self.V, self.F)
        if not ok:
            raise ValueError('remesh needs a closed manifold mesh')


@njit(types.Tuple((A1, A1b))(FN, FN_ID, A2i, A1, A2, A2i, f64),
      parallel=True, cache=True, nogil=True, fastmath=True)
def _edge_turns(run, run_id, code, P, V, E, h):
    """_edge_turn for every edge."""
    m = E.shape[0]
    cos = np.empty(m)
    seam = np.empty(m, dtype=np.bool_)
    for e in prange(m):
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        idstack = np.empty(STACK_DEPTH, dtype=np.int64)
        cos[e], seam[e] = _edge_turn(run, run_id, code, P, V, E[e, 0], E[e, 1], E[e, 4],
                                     E[e, 5], h, ps, ds, aux, idstack)
    return cos, seam


SEAM_COS = math.cos(math.radians(5.0))


def _crease_info(mesh, crease_cos, program, h):
    """Crease edges, and per vertex the number of crease edges through it.
    An edge is a crease where the SDF's normal turns sharply across it, or
    where two different primitives meet at even a slight angle (a boolean
    seam).  crease_cos=None: no creases."""
    n_e = len(mesh.E)
    if crease_cos is None:
        return np.zeros(n_e, dtype=bool), np.zeros(len(mesh.V), dtype=np.int64)
    cos, seam = _edge_turns(program.run, program.run_id, program.code, program.params,
                            mesh.V, mesh.E, h)
    crease = (cos < crease_cos) | (seam & (cos < SEAM_COS))
    count = np.bincount(mesh.E[crease][:, :2].ravel(), minlength=len(mesh.V))
    return crease, count


# --- sizing ------------------------------------------------------------------

def _sizing(mesh, program, min_len, max_len, tolerance, crease_len, crease_count,
            gradation, shortest=0.0):
    V = mesh.V
    kappa = _curvature(program.run, program.code, program.params, V, 0.1 * min_len)
    eps = tolerance
    with np.errstate(divide='ignore', invalid='ignore'):
        L = np.sqrt(np.maximum(6 * eps / kappa - 3 * eps * eps, 0.0))
    L = np.where(kappa > 1e-12, L, max_len)
    L = np.clip(L, min_len, max_len)
    L[crease_count > 0] = crease_len
    # edges must not be longer than the shape is thin, or splits and
    # projections can jump to the opposite surface (and fold the mesh)
    thin = _local_size(program.run, program.code, program.params, V,
                       2.0 * max_len / 16, 2.0 * max_len)
    L = np.minimum(L, np.maximum(thin, 0.5 * min_len))
    # where the floor overrides the thickness limit (counted for the report)
    mesh.floor_conflict = int((np.maximum(thin, 0.5 * min_len) < shortest).sum()) \
        if shortest else 0
    # grade: a vertex's target can't exceed a neighbour's by more than
    # gradation * (distance between them)
    a, b = mesh.E[:, 0], mesh.E[:, 1]
    step = gradation * mesh.edge_len
    for _ in range(64):
        before = L.copy()
        np.minimum.at(L, a, L[b] + step)
        np.minimum.at(L, b, L[a] + step)
        if np.array_equal(before, L):
            break
    # the hard floor wins over everything.  Edges are only collapsed below
    # 0.8 of their target, so the target floor sits above it
    return np.maximum(L, shortest / 0.8)


# --- topology batches ----------------------------------------------------------

@njit(types.UniTuple(i64, 3)(FN, FN_ID, A2i, A1, A2, A2i, i64, i64, A2i, A1, A1b, A1i,
                              f64, f64, A2i), cache=True, nogil=True)
def _split_batch(run, run_id, code, P, V, F, nv, nf, E, L, lock, vcrease, h, crease_cos,
                 parents):
    """Split long edges.  An edge's two faces must be untouched this batch
    (`lock` is per face).  New vertices are appended at V[nv...], new
    faces at F[nf...], and each new vertex's two parents at parents[...]; a
    new vertex splitting a crease edge, or an edge that cuts across a crease,
    is marked as a crease vertex (vcrease = 2) so that it is snapped onto the
    crease.  Returns (nv, nf, splits)."""
    ps = np.empty((STACK_DEPTH, 3))
    ds = np.empty(STACK_DEPTH)
    aux = np.empty(STACK_DEPTH)
    idstack = np.empty(STACK_DEPTH, dtype=np.int64)
    splits = 0
    nv0 = nv
    for e in range(E.shape[0]):
        a, b, f0, f1, c, d = E[e, 0], E[e, 1], E[e, 2], E[e, 3], E[e, 4], E[e, 5]
        if lock[f0] or lock[f1]:
            continue
        dx = V[a, 0] - V[b, 0]
        dy = V[a, 1] - V[b, 1]
        dz = V[a, 2] - V[b, 2]
        target = min(L[a], L[b])
        len2 = dx * dx + dy * dy + dz * dz
        if len2 <= (4.0 / 3.0 * target) ** 2:
            continue
        t = 0.5
        on_crease = crease_cos > -1.5 and (
            _cuts_crease(run, code, P, V, a, b, h, crease_cos, ps, ds, aux) or
            _is_crease_edge(run, run_id, code, P, V, F, a, b, f0, f1, vcrease, h, crease_cos, ps, ds, aux,
                           idstack))
        m = nv
        V[m, 0] = V[a, 0] + t * (V[b, 0] - V[a, 0])
        V[m, 1] = V[a, 1] + t * (V[b, 1] - V[a, 1])
        V[m, 2] = V[a, 2] + t * (V[b, 2] - V[a, 2])
        L[m] = L[a] + t * (L[b] - L[a])
        vcrease[m] = 2 if on_crease else 0
        parents[m - nv0, 0], parents[m - nv0, 1] = a, b
        # f0 = (a, b, c) -> (a, m, c), (m, b, c)
        # f1 = (b, a, d) -> (b, m, d), (m, a, d)
        F[f0, 0], F[f0, 1], F[f0, 2] = a, m, c
        F[nf, 0], F[nf, 1], F[nf, 2] = m, b, c
        F[f1, 0], F[f1, 1], F[f1, 2] = b, m, d
        F[nf + 1, 0], F[nf + 1, 1], F[nf + 1, 2] = m, a, d
        lock[f0] = lock[f1] = True
        nv += 1
        nf += 2
        splits += 1
    return nv, nf, splits


@njit(i64(FN, FN_ID, A2i, A1, A2, A2i, A2i, A1, A1i, A1i, A1i, A1b, A1b, A1i, A1i,
          f64, f64), cache=True, nogil=True)
def _collapse_batch(run, run_id, code, P, V, F, E, L, valence, vf_start, vf, lock, alive,
                    stamp, vcrease, h, crease_cos):
    """Collapse short edges, and the short edge of any needle, b into a.
    Faces f0, f1 die; b is left unused."""
    ps = np.empty((STACK_DEPTH, 3))
    ds = np.empty(STACK_DEPTH)
    aux = np.empty(STACK_DEPTH)
    idstack = np.empty(STACK_DEPTH, dtype=np.int64)
    collapses = 0
    tick = 0
    for e in range(E.shape[0]):
        a, b, f0, f1, c, d = E[e, 0], E[e, 1], E[e, 2], E[e, 3], E[e, 4], E[e, 5]
        if lock[a] or lock[b] or lock[c] or lock[d]:
            continue
        target = min(L[a], L[b])
        dx = V[a, 0] - V[b, 0]
        dy = V[a, 1] - V[b, 1]
        dz = V[a, 2] - V[b, 2]
        needle = _is_needle_edge(V, a, b, c) or _is_needle_edge(V, a, b, d)
        if dx * dx + dy * dy + dz * dz >= (0.8 * target) ** 2 and not needle:
            continue
        if valence[c] <= 3 or valence[d] <= 3:
            continue
        # creases: corners never move; crease vertices only along a crease
        # (an edge of practically no length is merged whatever it is)
        ka, kb = vcrease[a], vcrease[b]
        tiny = dx * dx + dy * dy + dz * dz < (1e-6 * target) ** 2
        if not tiny and ka > 0 and kb > 0 and not _is_crease_edge(
                run, run_id, code, P, V, F, a, b, f0, f1, vcrease, h, crease_cos, ps, ds, aux,
                idstack):
            continue
        if not tiny and kb > 2 and ka > 2:
            continue
        if kb > ka:
            a, b = b, a
            f0, f1 = f1, f0
            c, d = d, c
            ka, kb = kb, ka
        if ka > 0:
            px, py, pz = V[a, 0], V[a, 1], V[a, 2]
        else:
            px, py, pz = 0.5 * (V[a, 0] + V[b, 0]), 0.5 * (V[a, 1] + V[b, 1]), 0.5 * (V[a, 2] + V[b, 2])
        # link condition: a and b share exactly the neighbours c and d
        tick += 1
        for r in range(vf_start[a], vf_start[a + 1]):
            f = vf[r]
            for s in range(3):
                stamp[F[f, s]] = tick
        shared = 0
        for r in range(vf_start[b], vf_start[b + 1]):
            f = vf[r]
            for s in range(3):
                u = F[f, s]
                if u != a and u != b and stamp[u] == tick:
                    stamp[u] = -tick   # count each vertex once
                    shared += 1
        if shared != 2:
            continue
        # no new edge too long, no face flipped or squashed
        ok = True
        for side in range(2):
            v0 = a if side == 0 else b
            for r in range(vf_start[v0], vf_start[v0 + 1]):
                f = vf[r]
                if f == f0 or f == f1:
                    continue
                i, j, k = F[f, 0], F[f, 1], F[f, 2]
                onx, ony, onz = _normal(V, i, j, k)
                ol = math.sqrt(onx * onx + ony * ony + onz * onz)
                ox, oy, oz = V[v0, 0], V[v0, 1], V[v0, 2]
                V[v0, 0], V[v0, 1], V[v0, 2] = px, py, pz
                nnx, nny, nnz = _normal(V, i, j, k)
                V[v0, 0], V[v0, 1], V[v0, 2] = ox, oy, oz
                nl = math.sqrt(nnx * nnx + nny * nny + nnz * nnz)
                if nl < 1e-3 * ol or (onx * nnx + ony * nny + onz * nnz) < 0.5 * ol * nl:
                    ok = False
                    break
                V[v0, 0], V[v0, 1], V[v0, 2] = px, py, pz
                out = _faces_out(run, code, P, V, i, j, k, h, ps, ds, aux)
                V[v0, 0], V[v0, 1], V[v0, 2] = ox, oy, oz
                if not out:
                    ok = False
                    break
                for s in range(3):
                    u = F[f, s]
                    if u == v0 or needle:     # a needle goes even if edges grow
                        continue
                    ex, ey, ez = V[u, 0] - px, V[u, 1] - py, V[u, 2] - pz
                    if ex * ex + ey * ey + ez * ez > (4.0 / 3.0 * min(L[u], target)) ** 2:
                        ok = False
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        # do it
        V[a, 0], V[a, 1], V[a, 2] = px, py, pz
        L[a] = min(L[a], L[b])
        for r in range(vf_start[b], vf_start[b + 1]):
            f = vf[r]
            for s in range(3):
                if F[f, s] == b:
                    F[f, s] = a
        alive[f0] = False
        alive[f1] = False
        for side in range(2):
            v0 = a if side == 0 else b
            for r in range(vf_start[v0], vf_start[v0 + 1]):
                f = vf[r]
                for s in range(3):
                    lock[F[f, s]] = True
        lock[b] = True
        collapses += 1
    return collapses


@njit(i64(FN, FN_ID, A2i, A1, A2, A2i, A2i, A1i, A1i, A1i, A1b, A1i, f64, f64, b1, A1b),
      cache=True, nogil=True)
def _flip_batch(run, run_id, code, P, V, F, E, valence, vf_start, vf, lock, vcrease, h,
                crease_cos, features_only, far):
    """Flip edges.  mid_ab / mid_cd are |SDF| at the midpoints of the edge and
    of the other diagonal: an edge that cuts across a crease has a midpoint
    well off the surface, and is turned to lie along it (a feature flip).
    Otherwise flip when it clearly improves the smallest angle of the two
    triangles (this undoes caps), or when it brings valences closer to 6
    without making the smallest angle worse; never onto a diagonal that cuts."""
    ps = np.empty((STACK_DEPTH, 3))
    ds = np.empty(STACK_DEPTH)
    aux = np.empty(STACK_DEPTH)
    idstack = np.empty(STACK_DEPTH, dtype=np.int64)
    flips = 0
    for e in range(E.shape[0]):
        a, b, f0, f1, c, d = E[e, 0], E[e, 1], E[e, 2], E[e, 3], E[e, 4], E[e, 5]
        if lock[a] or lock[b] or lock[c] or lock[d]:
            continue
        if valence[a] <= 3 or valence[b] <= 3:
            continue
        ex, ey, ez = V[a, 0] - V[b, 0], V[a, 1] - V[b, 1], V[a, 2] - V[b, 2]
        length = math.sqrt(ex * ex + ey * ey + ez * ez)
        # open surface (see _open_surface): no crease to cut or follow, and
        # the triangles' own normals are a safe guide, so the SDF isn't asked
        open_ = far[a] and far[b] and far[c] and far[d]
        if open_:
            # ... and only while the two triangles are nearly coplanar
            p0 = _normal(V, a, b, c)
            p1 = _normal(V, b, a, d)
            pp = p0[0] * p1[0] + p0[1] * p1[1] + p0[2] * p1[2]
            l0 = p0[0] * p0[0] + p0[1] * p0[1] + p0[2] * p0[2]
            l1 = p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2]
            open_ = pp > OPEN_COS * math.sqrt(l0 * l1)
        if open_ and features_only:
            continue
        flat = ((_min_angle(V, a, b, c) < 0.02 and _cap_apex_t(V, a, b, c) >= 0.0) or
                (_min_angle(V, b, a, d) < 0.02 and _cap_apex_t(V, b, a, d) >= 0.0))
        cuts_ab = False if open_ else _cuts_crease(run, code, P, V, a, b, h, crease_cos, ps, ds, aux)
        cuts_cd = False if open_ else _cuts_crease(run, code, P, V, c, d, h, crease_cos, ps, ds, aux)
        if flat:
            # a cap flattened onto its long edge (its apex on the same crease):
            # the path through the apex already follows the crease, so this
            # edge can go even if it is a crease edge
            if min(_min_angle(V, a, d, c), _min_angle(V, d, b, c)) <= \
                    min(_min_angle(V, a, b, c), _min_angle(V, b, a, d)):
                continue
        elif cuts_ab and not cuts_cd:
            pass                      # feature flip: the other diagonal runs along the crease
        elif features_only:
            continue
        elif not open_ and _is_crease_edge(run, run_id, code, P, V, F, a, b, f0, f1, vcrease, h,
                                           crease_cos, ps, ds, aux, idstack):
            continue
        else:
            q_before = min(_min_angle(V, a, b, c), _min_angle(V, b, a, d))
            q_after = min(_min_angle(V, a, d, c), _min_angle(V, d, b, c))
            better_shape = q_after > q_before + 0.035 and q_before < 0.52
            before = ((valence[a] - 6) ** 2 + (valence[b] - 6) ** 2 +
                      (valence[c] - 6) ** 2 + (valence[d] - 6) ** 2)
            after = ((valence[a] - 7) ** 2 + (valence[b] - 7) ** 2 +
                     (valence[c] - 5) ** 2 + (valence[d] - 5) ** 2)
            better_valence = after < before and q_after >= min(q_before, 0.52)
            if not (better_shape or better_valence):
                continue
            if cuts_cd and not cuts_ab:
                continue              # would cut a crease
        # c and d must not already be connected
        connected = False
        for r in range(vf_start[c], vf_start[c + 1]):
            f = vf[r]
            if F[f, 0] == d or F[f, 1] == d or F[f, 2] == d:
                connected = True
                break
        if connected:
            continue
        # the two new faces must face the same way as the old ones
        n0 = _normal(V, a, b, c)
        n1 = _normal(V, b, a, d)
        m0 = _normal(V, a, d, c)
        m1 = _normal(V, d, b, c)
        sx, sy, sz = n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2]
        if (m0[0] * sx + m0[1] * sy + m0[2] * sz) <= 0.0 or (m1[0] * sx + m1[1] * sy + m1[2] * sz) <= 0.0:
            continue
        # ... and the way the surface does (on a crease the old normals are no guide)
        if not open_ and not (_faces_out(run, code, P, V, a, d, c, h, ps, ds, aux) and
                              _faces_out(run, code, P, V, d, b, c, h, ps, ds, aux)):
            continue
        F[f0, 0], F[f0, 1], F[f0, 2] = a, d, c
        F[f1, 0], F[f1, 1], F[f1, 2] = d, b, c
        lock[a] = lock[b] = lock[c] = lock[d] = True
        flips += 1
    return flips


@njit(parallel=True, cache=True, nogil=True)
def _relax(V, F, vf_start, vf, crease_count, E, crease_e, out):
    """Tangential smoothing target for each vertex (area-weighted centroid
    of its one-ring, moved within the tangent plane)."""
    n = V.shape[0]
    # crease neighbours, for vertices on a crease line
    cn = np.full((n, 2), -1, dtype=np.int64)
    for e in range(E.shape[0]):
        if crease_e[e]:
            a, b = E[e, 0], E[e, 1]
            if cn[a, 0] < 0:
                cn[a, 0] = b
            else:
                cn[a, 1] = b
            if cn[b, 0] < 0:
                cn[b, 0] = a
            else:
                cn[b, 1] = a
    for i in prange(n):
        out[i, 0], out[i, 1], out[i, 2] = V[i, 0], V[i, 1], V[i, 2]
        k = crease_count[i]
        if k > 2 or vf_start[i] == vf_start[i + 1]:
            continue          # corner, or unused vertex
        if k == 2:
            # move halfway towards the midpoint of its two crease neighbours;
            # the crease snap that follows puts it back on the crease
            p, q = cn[i, 0], cn[i, 1]
            out[i, 0] = 0.5 * (V[p, 0] + V[q, 0]) * 0.5 + V[i, 0] * 0.5
            out[i, 1] = 0.5 * (V[p, 1] + V[q, 1]) * 0.5 + V[i, 1] * 0.5
            out[i, 2] = 0.5 * (V[p, 2] + V[q, 2]) * 0.5 + V[i, 2] * 0.5
            continue
        if k == 1:
            continue          # crease end inside a smooth area: leave it
        cx = cy = cz = 0.0
        wsum = 0.0
        nx = ny = nz = 0.0
        for r in range(vf_start[i], vf_start[i + 1]):
            f = vf[r]
            a, b, c = F[f, 0], F[f, 1], F[f, 2]
            fx, fy, fz = _normal(V, a, b, c)
            area = 0.5 * math.sqrt(fx * fx + fy * fy + fz * fz)
            nx += fx
            ny += fy
            nz += fz
            cx += area * (V[a, 0] + V[b, 0] + V[c, 0]) / 3.0
            cy += area * (V[a, 1] + V[b, 1] + V[c, 1]) / 3.0
            cz += area * (V[a, 2] + V[b, 2] + V[c, 2]) / 3.0
            wsum += area
        if wsum == 0.0:
            continue
        cx, cy, cz = cx / wsum, cy / wsum, cz / wsum
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl == 0.0:
            continue
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        dx, dy, dz = cx - V[i, 0], cy - V[i, 1], cz - V[i, 2]
        t = dx * nx + dy * ny + dz * nz
        out[i, 0] = V[i, 0] + dx - t * nx
        out[i, 1] = V[i, 1] + dy - t * ny
        out[i, 2] = V[i, 2] + dz - t * nz


# --- driver ----------------------------------------------------------------------

OPEN_RINGS = 5       # edge hops from a crease or thin part before the surface is open


def _open_surface(E, n, near):
    """Vertices at least OPEN_RINGS edges away from any `near` vertex
    (on a crease, or where the shape is thin).  There a flip or a small move
    can't cut a crease or reach the far side of the shape, so the cheap
    geometric checks are enough and the SDF needn't be sampled."""
    near = near.copy()
    a, b = E[:, 0], E[:, 1]
    for _ in range(OPEN_RINGS):
        grow = near.copy()
        grow[a[near[b]]] = True
        grow[b[near[a]]] = True
        near = grow
    return ~near


def _flips(program, V, F, vcrease, h, ccos, features_only, passes, order, thin=None):
    for _ in range(passes):
        mesh = Mesh(V, F)
        lock = np.zeros(len(V), dtype=bool)
        far = (np.zeros(len(V), dtype=bool) if thin is None    # None: no shortcuts
               else _open_surface(mesh.E, len(V), (vcrease > 0) | thin))
        done = _flip_batch(program.run, program.run_id, program.code, program.params,
                           V, F, order(mesh.E), mesh.valence, mesh.vf_start, mesh.vf, lock,
                           vcrease, h, ccos, features_only, far)
        if done == 0:
            break


def _compact(V, F, L, extra=None):
    used = np.zeros(len(V), dtype=bool)
    used[F.ravel()] = True
    remap = np.cumsum(used) - 1
    out = (V[used], remap[F].astype(np.int64), L[used])
    if extra is not None:
        out += (extra[used],)
    return out


def _normals_of(program, V, h):
    out = np.empty_like(V)
    _normals(program.run, program.code, program.params, np.ascontiguousarray(V), out, h)
    return out


def _folded(program, V, F, h):
    """Triangles facing against the surface."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    grad = _normals_of(program, (a + b + c) / 3, h)
    return (np.cross(b - a, c - a) * grad).sum(1) < 0


def _undo_folds(program, V, F, before, h, rounds=8, far=None, ref=None):
    """Relaxing and snapping move vertices without checking the triangles
    around them; where a triangle got turned over, put its corners back
    where they were before the move (in place), until none is left.  If
    that keeps tipping neighbours over, the undo widens to their corners
    too; at worst everything is put back, which had no new folds.

    Only triangles with a moved corner are checked.  On open surface (`far`,
    see _open_surface) they are checked against `ref`, the SDF normals at
    the corners before the move, which needs no new SDF samples; elsewhere
    against the SDF at the triangle."""
    moved = np.any(V != before, axis=1)
    sel = np.flatnonzero(moved[F].any(1))
    if not len(sel):
        return
    T = F[sel]
    use_ref = far is not None and ref is not None

    def align(X, Q):
        """cos between each triangle's normal and its corners' reference normals."""
        n = np.cross(X[Q[:, 1]] - X[Q[:, 0]], X[Q[:, 2]] - X[Q[:, 0]])
        r = ref[Q[:, 0]] + ref[Q[:, 1]] + ref[Q[:, 2]]
        return (n * r).sum(1) / (np.linalg.norm(n, axis=1) * np.linalg.norm(r, axis=1) + 1e-300)

    def split(T):
        """Open-surface triangles well aligned before the move get the cheap
        check; the rest are checked against the SDF."""
        if not use_ref or not len(T):
            return np.zeros(len(T), dtype=bool), T
        geo = far[T].all(1)
        geo[geo] = align(before, T[geo]) > 0.5
        return geo, T[~geo]

    geo, sdf_t = split(T)
    was_folded = _folded(program, before, sdf_t, h)

    def new_folds():
        bad = np.zeros(len(T), dtype=bool)
        bad[~geo] = _folded(program, V, sdf_t, h) & ~was_folded
        if geo.any():
            # cheap: still well aligned is fine; anything doubtful asks the SDF
            # (it wasn't folded before: it was well aligned)
            g = np.flatnonzero(geo)
            doubtful = g[align(V, T[g]) < 0.3]
            if len(doubtful):
                bad[doubtful] = _folded(program, V, T[doubtful], h)
        return bad

    for r in range(rounds):
        bad = new_folds()
        if not bad.any():
            return
        undo = np.zeros(len(V), dtype=bool)
        undo[T[bad].ravel()] = True
        if r >= 2:                       # widen: every triangle touching those corners
            undo[F[undo[F].any(1)].ravel()] = True
            sel = np.flatnonzero((moved | undo)[F].any(1))   # checked triangles grow with it
            T = F[sel]
            geo, sdf_t = split(T)
            was_folded = _folded(program, before, sdf_t, h)
        V[undo] = before[undo]
    if new_folds().any():
        V[:] = before


def quality(sdf, verts, tris):
    """How good a triangle mesh of `sdf` is.  Returns a dict:
        triangles    number of triangles
        needles      share with one edge much shorter than the others
        caps         share that are nearly flat (an angle over 120 degrees)
        folded       triangles facing against the surface
        min_angle    smallest angle in the mesh, degrees
        max_offset   furthest a vertex is from the surface (vertices are left
                     off it where projecting would turn a triangle over)
    """
    program = sdf.program() if hasattr(sdf, 'program') else sdf
    V = np.asarray(verts, dtype=np.float64)
    T = np.asarray(tris, dtype=np.int64)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    L = np.stack([np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1),
                  np.linalg.norm(a - b, axis=1)], 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos = np.empty_like(L)
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            cos[:, i] = (L[:, j] ** 2 + L[:, k] ** 2 - L[:, i] ** 2) / (2 * L[:, j] * L[:, k])
    ang = np.degrees(np.arccos(np.clip(np.nan_to_num(cos, nan=1.0), -1, 1)))
    amin, amax = ang.min(1), ang.max(1)
    normal = np.cross(b - a, c - a)
    grad = _normals_of(program, (a + b + c) / 3, 1e-5 * float(np.median(L)))
    return {
        'triangles': len(T),
        'needles': float(((amin < 20) & (amax <= 120)).mean()) if len(T) else 0.0,
        'caps': float((amax > 120).mean()) if len(T) else 0.0,
        'folded': int(((normal * grad).sum(1) < 0).sum()),
        'min_angle': float(amin.min()) if len(T) else 0.0,
        'max_offset': float(np.abs(_distances(program, V)).max()) if len(V) else 0.0,
    }


def advice(report):
    """Plain-language warnings for a remesh report (empty: nothing to say)."""
    out = []
    if report.get('floor_conflict', 0) > 0 and (report['folded'] or report.get('runaway')):
        out.append("Shortest edge is longer than parts of the shape are thick, so those "
                   "parts fold. Lower Shortest edge.")
    elif report.get('runaway') or report.get('input_folded', 0) > 0:
        n = report.get('input_folded', 0)
        folded = f' ({n} folded triangles in it)' if n else ''
        out.append(f"The starting mesh is too coarse for this shape's thinnest parts{folded}. "
                   "Start from a finer mesh: raise Resolution on SDF Mesh, or Start resolution.")
    elif report['folded'] > 0:
        out.append(f"{report['folded']} triangles are turned over, usually where the shape "
                   "narrows to a sharp edge. Try another Seed, or a smaller Edge length.")
    thin = report['needles'] + report['caps']
    if thin > 0.01:
        out.append(f"{thin * 100:.1f}% of the triangles are thin (needles or caps). "
                   "Try another Seed, or a higher Detail.")
    return out


def _distances(program, V):
    from .grid import evaluate
    return evaluate(program, V).ravel()


def _expected_triangles(V, F, L):
    """How many triangles the sizing asks for: each triangle's area over the
    area of an equilateral triangle with its corners' mean target length."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    target = L[F].mean(1)
    return float((area / (0.433 * target ** 2)).sum())




def remesh(sdf, verts, faces, max_len=None, min_len=None, tolerance=None,
           creases=True, crease_angle=30.0, crease_len=None, gradation=0.4,
           iterations=6, project=True, seed=0, shortest=None, report=False):
    """Remesh a closed mesh of `sdf` adaptively.  Returns (verts, triangles),
    or (verts, triangles, quality dict) with report=True (see quality()).

    max_len       longest edge wanted, on flat areas (default: bounding
                  box diagonal / 40)
    min_len       shortest edge wanted (default: max_len / 10)
    tolerance     how far a flat triangle may sit from a curved surface;
                  smaller = denser on curves (default: min_len / 4)
    creases       find creases and keep them (with finer triangles along them);
                  False treats the surface as smooth everywhere
    crease_angle  where the surface normal turns more than this (degrees)
                  across an edge, that edge is a crease
    crease_len    edge length along creases (default: max_len / 4)
    gradation     how quickly sizes may grow away from creases and tight
                  curves (0.4 = 40% of the distance travelled)
    iterations    remeshing passes
    shortest      hard floor on edge length, whatever curvature, creases or
                  thin parts ask for (default: none).  Above the thickness of
                  thin parts it can fold them: the report will say so
    seed          0: edges are visited in their natural order; any other
                  value shuffles that order, giving a different (but equally
                  valid) result.  The same seed always gives the same mesh.
    report        also return quality(), plus:
                    input_folded  folded triangles in the mesh passed in
                    floor_conflict  vertices where `shortest` overrode the
                                  shape's thickness
                    runaway       True if splitting was stopped because the
                                  mesh grew far past what the sizing asks for
                  Both mean the starting mesh is too coarse for the shape's
                  thinnest parts: start from a finer one.

    Edges are also kept shorter than the shape is thin, and needles and caps
    are collapsed, flipped or split away; whatever remains is in the report.
    """
    program = sdf.program() if hasattr(sdf, 'program') else sdf
    # run on this shape's compiled code from the start, so the result doesn't
    # depend on whether the background compile had finished yet
    try:
        program.compile()
    except RuntimeError:
        pass                                    # fall back to the interpreter
    # copies: everything below works in place, and must not touch the caller's arrays
    V = np.array(verts, dtype=np.float64, order='C', copy=True)
    F = np.array(faces, dtype=np.int64, order='C', copy=True)
    if F.shape[1] == 4:
        from .mesh import triangulate
        F = triangulate(F, V, program).astype(np.int64)
    input_folded = int(_folded(program, V, F, 1e-4 * np.linalg.norm(V.max(0) - V.min(0)) / 400).sum()) \
        if report else 0
    runaway = False
    floor_conflict = 0
    diag = np.linalg.norm(V.max(0) - V.min(0))
    max_len = max_len or diag / 40
    min_len = min_len or max_len / 10
    tolerance = tolerance or min_len / 4
    crease_len = crease_len or max_len / 4
    shortest = shortest or 0.0
    max_len, min_len, crease_len = (max(x, shortest) for x in (max_len, min_len, crease_len))
    crease_cos = math.cos(math.radians(crease_angle)) if creases else None
    h_grad = 1e-4 * min_len
    code, P = program.code, program.params
    run, run_id = program.run, program.run_id

    rng = np.random.default_rng(seed) if seed else None

    def order(E):
        return np.ascontiguousarray(E[rng.permutation(len(E))]) if rng is not None else E

    ccos = crease_cos if crease_cos is not None else -2.0   # -2: no creases
    for _ in range(iterations):
        if runaway:
            break
        # the one full crease scan per iteration; flags then travel with
        # the vertices through splits and collapses
        mesh = Mesh(V, F)
        _, vcrease = _crease_info(mesh, crease_cos, program, h_grad)
        L = _sizing(mesh, program, min_len, max_len, tolerance, crease_len,
                    vcrease, gradation, shortest)
        floor_conflict = max(floor_conflict, mesh.floor_conflict)
        # a brake: a starting mesh too coarse for thin parts can make splits
        # feed on themselves.  Splitting never goes far past what the sizing
        # asks for (or twice where it started), and if an iteration still
        # ends well above it, remeshing stops there
        expected = _expected_triangles(V, F, L)
        most = max(4.0 * expected + 1000, 2.0 * len(F))

        # where edges are held short (thin parts, tight curves) the SDF is
        # always consulted; see _open_surface
        def held(L):
            return L <= 1.01 * min_len

        # turn edges that cut across creases (the input may have some)
        _flips(program, V, F, vcrease, h_grad, ccos, True, 3, order, held(L))

        # split until nothing is too long
        for _ in range(8):
            if len(F) > most:
                break
            mesh = Mesh(V, F)
            ne = len(mesh.E)
            Vb = np.empty((len(V) + ne, 3)); Vb[:len(V)] = V
            Fb = np.empty((len(F) + 2 * ne, 3), dtype=np.int64); Fb[:len(F)] = F
            Lb = np.empty(len(V) + ne); Lb[:len(V)] = L
            cb = np.zeros(len(V) + ne, dtype=np.int64); cb[:len(V)] = vcrease
            parents = np.zeros((ne, 2), dtype=np.int64)
            lock = np.zeros(len(F), dtype=bool)
            nv0 = len(V)
            nv, nf, done = _split_batch(run, run_id, code, P, Vb, Fb, nv0, len(F),
                                        order(mesh.E), Lb, lock, cb, h_grad, ccos, parents)
            V, F, L, vcrease = Vb[:nv], Fb[:nf], Lb[:nv], cb[:nv]
            if project and done:
                unprojected = V.copy()
                new = np.zeros(nv, dtype=bool); new[nv0:] = True
                # each new vertex should end up facing like its two parents
                par = parents[:done]
                n_par = _normals_of(program, V[par.ravel()], h_grad).reshape(-1, 2, 3)
                ref = np.zeros_like(V)
                ref[nv0:] = n_par.sum(1)
                ref[nv0:] /= np.linalg.norm(ref[nv0:], axis=1, keepdims=True) + 1e-300
                _project(run, code, P, V, new & (vcrease == 0), ref, h_grad, 3)
                snap = new & (vcrease > 0)
                if snap.any():
                    m2 = Mesh(V, F)
                    _project_features(run, code, P, V, F, m2.vf_start, m2.vf, snap,
                                      np.full(len(V), 2), h_grad, 3)
                _undo_folds(program, V, F, unprojected, h_grad)
            if done == 0:
                break

        # collapse until nothing is too short and no needle is left (or enough passes)
        for _ in range(6):
            mesh = Mesh(V, F)
            lock = np.zeros(len(V), dtype=bool)
            alive = np.ones(len(F), dtype=bool)
            stamp = np.zeros(len(V), dtype=np.int64)
            done = _collapse_batch(run, run_id, code, P, V, F, order(mesh.E), L, mesh.valence,
                                   mesh.vf_start, mesh.vf, lock, alive, stamp,
                                   vcrease, h_grad, ccos)
            F = F[alive]
            V, F, L, vcrease = _compact(V, F, L, vcrease)
            if done == 0:
                break
        if len(F) > 3.0 * expected + 1000:
            runaway = True                      # past this point it can't end well

        # flip away caps and towards valence 6, and turn any edge that cuts a crease
        _flips(program, V, F, vcrease, h_grad, ccos, False, 3, order, held(L))

        # relax and project (crease edges looked up afresh for the sliding)
        mesh = Mesh(V, F)
        crease_e, vcrease = _crease_info(mesh, crease_cos, program, h_grad)
        ref = _normals_of(program, V, h_grad)          # the side each vertex is on now
        before = V.copy()
        target = np.empty_like(V)
        _relax(V, F, mesh.vf_start, mesh.vf, vcrease, mesh.E, crease_e, target)
        V = target
        if project:
            on_crease = vcrease > 0
            _project(run, code, P, V, ~on_crease, ref, h_grad, 3)
            rank = np.where(vcrease > 2, 3, 2)
            _project_features(run, code, P, V, F, mesh.vf_start, mesh.vf, on_crease, rank,
                              h_grad, 3)
        _undo_folds(program, V, F, before, h_grad, ref=ref,
                    far=_open_surface(mesh.E, len(V), (vcrease > 0) | held(L)))

    # clean-up: crease snapping can pull neighbouring vertices onto the same
    # point (zero-area triangles); merge those and any needle left, and flip
    # away caps, without snapping anything again
    for _ in range(3):
        mesh = Mesh(V, F)
        lock = np.zeros(len(V), dtype=bool)
        alive = np.ones(len(F), dtype=bool)
        stamp = np.zeros(len(V), dtype=np.int64)
        L = np.full(len(V), 1e-300)            # only tiny edges and needles qualify
        done = _collapse_batch(run, run_id, code, P, V, F, order(mesh.E), L, mesh.valence,
                               mesh.vf_start, mesh.vf, lock, alive, stamp,
                               np.zeros(len(V), dtype=np.int64), h_grad, ccos)
        F = F[alive]
        V, F, L = _compact(V, F, L)
        if done == 0:
            break
    mesh = Mesh(V, F)
    _, vcrease = _crease_info(mesh, crease_cos, program, h_grad)
    _flips(program, V, F, vcrease, h_grad, ccos, False, 2, order)

    T = F.astype(np.int32)
    if report:
        q = quality(program, V, T)
        q['input_folded'] = input_folded
        q['runaway'] = runaway
        q['floor_conflict'] = floor_conflict
        return V, T, q
    return V, T
