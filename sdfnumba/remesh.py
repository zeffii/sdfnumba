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

from .interp import run, STACK_DEPTH
from .dc import _jacobi_eigen


# --- SDF queries ---------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=True)
def _project(code, P, V, mask, h, steps):
    """Newton steps onto the zero level: p -= d * g / |g|^2."""
    for i in prange(V.shape[0]):
        if not mask[i]:
            continue
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        x, y, z = V[i, 0], V[i, 1], V[i, 2]
        for _ in range(steps):
            d = run(code, P, x, y, z, ps, ds, aux)
            gx = (run(code, P, x + h, y, z, ps, ds, aux) - run(code, P, x - h, y, z, ps, ds, aux)) / (2 * h)
            gy = (run(code, P, x, y + h, z, ps, ds, aux) - run(code, P, x, y - h, z, ps, ds, aux)) / (2 * h)
            gz = (run(code, P, x, y, z + h, ps, ds, aux) - run(code, P, x, y, z - h, ps, ds, aux)) / (2 * h)
            g2 = gx * gx + gy * gy + gz * gz
            if g2 < 1e-12:
                break
            x -= d * gx / g2
            y -= d * gy / g2
            z -= d * gz / g2
        V[i, 0], V[i, 1], V[i, 2] = x, y, z


@njit(parallel=True, cache=True, fastmath=True)
def _project_features(code, P, V, F, vf_start, vf, mask, rank, h, steps):
    """Snap vertices onto creases and corners.

    Each face around the vertex gives a sample point a little way into that
    face; the SDF value and gradient there define that side's tangent plane.
    The vertex moves to the point nearest to it that lies on all those planes
    (a small QEF, as in dual contouring): two distinct planes put it on the
    crease line, three put it on the corner, one is a plain projection."""
    for i in prange(V.shape[0]):
        if not mask[i]:
            continue
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        A = np.empty((3, 3))
        Q = np.empty((3, 3))
        w = np.empty(3)
        for _ in range(steps):
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
            V[i, 0], V[i, 1], V[i, 2] = x + mx, y + my, z + mz


@njit(parallel=True, cache=True, fastmath=True)
def _curvature(code, P, V, h):
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

@njit(cache=True)
def _normal(V, i, j, k):
    ux, uy, uz = V[j, 0] - V[i, 0], V[j, 1] - V[i, 1], V[j, 2] - V[i, 2]
    vx, vy, vz = V[k, 0] - V[i, 0], V[k, 1] - V[i, 1], V[k, 2] - V[i, 2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return nx, ny, nz



@njit(cache=True)
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


@njit(parallel=True, cache=True, fastmath=True)
def _face_normals_sdf(code, P, V, F, h):
    """Unit SDF gradient at each face centre.  Unlike the face's own normal,
    this cannot be thrown off by a badly shaped triangle."""
    m = F.shape[0]
    out = np.empty((m, 3))
    for f in prange(m):
        ps = np.empty((STACK_DEPTH, 3))
        ds = np.empty(STACK_DEPTH)
        aux = np.empty(STACK_DEPTH)
        a, b, c = F[f, 0], F[f, 1], F[f, 2]
        x = (V[a, 0] + V[b, 0] + V[c, 0]) / 3.0
        y = (V[a, 1] + V[b, 1] + V[c, 1]) / 3.0
        z = (V[a, 2] + V[b, 2] + V[c, 2]) / 3.0
        gx = run(code, P, x + h, y, z, ps, ds, aux) - run(code, P, x - h, y, z, ps, ds, aux)
        gy = run(code, P, x, y + h, z, ps, ds, aux) - run(code, P, x, y - h, z, ps, ds, aux)
        gz = run(code, P, x, y, z + h, ps, ds, aux) - run(code, P, x, y, z - h, ps, ds, aux)
        gl = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-300
        out[f, 0], out[f, 1], out[f, 2] = gx / gl, gy / gl, gz / gl
    return out


def _crease_info(mesh, crease_cos, program, h):
    """Crease edges (the SDF's normal turns sharply across them), and per
    vertex the number of crease edges through it.  crease_cos=None: no creases."""
    n_e = len(mesh.E)
    if crease_cos is None:
        return np.zeros(n_e, dtype=bool), np.zeros(len(mesh.V), dtype=np.int64)
    nrm = _face_normals_sdf(program.code, program.params, mesh.V, mesh.F, h)
    cos = np.einsum('ij,ij->i', nrm[mesh.E[:, 2]], nrm[mesh.E[:, 3]])
    crease = cos < crease_cos
    count = np.bincount(mesh.E[crease][:, :2].ravel(), minlength=len(mesh.V))
    return crease, count


# --- sizing ------------------------------------------------------------------

def _sizing(mesh, program, min_len, max_len, tolerance, crease_len, crease_count,
            gradation):
    V = mesh.V
    kappa = _curvature(program.code, program.params, V, 0.1 * min_len)
    eps = tolerance
    with np.errstate(divide='ignore', invalid='ignore'):
        L = np.sqrt(np.maximum(6 * eps / kappa - 3 * eps * eps, 0.0))
    L = np.where(kappa > 1e-12, L, max_len)
    L = np.clip(L, min_len, max_len)
    L[crease_count > 0] = crease_len
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
    return L


# --- per-edge tests, evaluated only for edges an operation is considering ----

@njit(cache=True, fastmath=True)
def _face_grad(code, P, V, F, f, h, ps, ds, aux):
    a, b, c = F[f, 0], F[f, 1], F[f, 2]
    x = (V[a, 0] + V[b, 0] + V[c, 0]) / 3.0
    y = (V[a, 1] + V[b, 1] + V[c, 1]) / 3.0
    z = (V[a, 2] + V[b, 2] + V[c, 2]) / 3.0
    gx = run(code, P, x + h, y, z, ps, ds, aux) - run(code, P, x - h, y, z, ps, ds, aux)
    gy = run(code, P, x, y + h, z, ps, ds, aux) - run(code, P, x, y - h, z, ps, ds, aux)
    gz = run(code, P, x, y, z + h, ps, ds, aux) - run(code, P, x, y, z - h, ps, ds, aux)
    gl = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-300
    return gx / gl, gy / gl, gz / gl


@njit(cache=True, fastmath=True)
def _is_crease_edge(code, P, V, F, a, b, f0, f1, vcrease, h, crease_cos, ps, ds, aux):
    """Both ends on a crease, and the SDF normal turns sharply across the edge."""
    if crease_cos < -1.5 or vcrease[a] == 0 or vcrease[b] == 0:
        return False
    n0 = _face_grad(code, P, V, F, f0, h, ps, ds, aux)
    n1 = _face_grad(code, P, V, F, f1, h, ps, ds, aux)
    return n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2] < crease_cos


@njit(cache=True, fastmath=True)
def _mid_dist(code, P, V, a, b, ps, ds, aux):
    return abs(run(code, P, 0.5 * (V[a, 0] + V[b, 0]), 0.5 * (V[a, 1] + V[b, 1]),
                   0.5 * (V[a, 2] + V[b, 2]), ps, ds, aux))


# --- topology batches ----------------------------------------------------------

@njit(cache=True)
def _split_batch(code, P, V, F, nv, nf, E, L, lock, vcrease, h, crease_cos):
    """Split long edges.  An edge's two faces must be untouched this batch
    (`lock` is per face).  New vertices are appended at V[nv...], new faces
    at F[nf...]; a new vertex splitting a crease edge, or an edge that cuts
    across a crease, is marked as a crease vertex (vcrease = 2) so that it is
    snapped onto the crease.  Returns (nv, nf, splits)."""
    ps = np.empty((STACK_DEPTH, 3))
    ds = np.empty(STACK_DEPTH)
    aux = np.empty(STACK_DEPTH)
    splits = 0
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
        on_crease = crease_cos > -1.5 and (
            _mid_dist(code, P, V, a, b, ps, ds, aux) > 0.02 * math.sqrt(len2) or
            _is_crease_edge(code, P, V, F, a, b, f0, f1, vcrease, h, crease_cos, ps, ds, aux))
        m = nv
        V[m, 0] = 0.5 * (V[a, 0] + V[b, 0])
        V[m, 1] = 0.5 * (V[a, 1] + V[b, 1])
        V[m, 2] = 0.5 * (V[a, 2] + V[b, 2])
        L[m] = 0.5 * (L[a] + L[b])
        vcrease[m] = 2 if on_crease else 0
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


@njit(cache=True)
def _collapse_batch(code, P, V, F, E, L, valence, vf_start, vf, lock, alive, stamp,
                    vcrease, h, crease_cos):
    """Collapse short edges, b into a.  Faces f0, f1 die; b is left unused."""
    ps = np.empty((STACK_DEPTH, 3))
    ds = np.empty(STACK_DEPTH)
    aux = np.empty(STACK_DEPTH)
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
        if dx * dx + dy * dy + dz * dz >= (0.8 * target) ** 2:
            continue
        if valence[c] <= 3 or valence[d] <= 3:
            continue
        # creases: corners never move; crease vertices only along a crease
        # (an edge of practically no length is merged whatever it is)
        ka, kb = vcrease[a], vcrease[b]
        tiny = dx * dx + dy * dy + dz * dz < (1e-6 * target) ** 2
        if not tiny and ka > 0 and kb > 0 and not _is_crease_edge(
                code, P, V, F, a, b, f0, f1, vcrease, h, crease_cos, ps, ds, aux):
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
                for s in range(3):
                    u = F[f, s]
                    if u == v0:
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


@njit(cache=True)
def _flip_batch(code, P, V, F, E, valence, vf_start, vf, lock, vcrease, h,
                crease_cos, features_only):
    """Flip edges.  mid_ab / mid_cd are |SDF| at the midpoints of the edge and
    of the other diagonal: an edge that cuts across a crease has a midpoint
    well off the surface, and is turned to lie along it (a feature flip).
    Otherwise flip towards valence 6, but never onto a diagonal that cuts."""
    ps = np.empty((STACK_DEPTH, 3))
    ds = np.empty(STACK_DEPTH)
    aux = np.empty(STACK_DEPTH)
    flips = 0
    for e in range(E.shape[0]):
        a, b, f0, f1, c, d = E[e, 0], E[e, 1], E[e, 2], E[e, 3], E[e, 4], E[e, 5]
        if lock[a] or lock[b] or lock[c] or lock[d]:
            continue
        if valence[a] <= 3 or valence[b] <= 3:
            continue
        ex, ey, ez = V[a, 0] - V[b, 0], V[a, 1] - V[b, 1], V[a, 2] - V[b, 2]
        length = math.sqrt(ex * ex + ey * ey + ez * ez)
        if crease_cos < -1.5:
            mid_ab = mid_cd = 0.0
        else:
            mid_ab = _mid_dist(code, P, V, a, b, ps, ds, aux)
            mid_cd = _mid_dist(code, P, V, c, d, ps, ds, aux)
        cuts = mid_ab > 0.02 * length
        if cuts and mid_cd < 0.5 * mid_ab:
            pass                      # feature flip: the other diagonal fits better
        elif features_only:
            continue
        elif _is_crease_edge(code, P, V, F, a, b, f0, f1, vcrease, h, crease_cos, ps, ds, aux):
            continue
        else:
            before = ((valence[a] - 6) ** 2 + (valence[b] - 6) ** 2 +
                      (valence[c] - 6) ** 2 + (valence[d] - 6) ** 2)
            after = ((valence[a] - 7) ** 2 + (valence[b] - 7) ** 2 +
                     (valence[c] - 5) ** 2 + (valence[d] - 5) ** 2)
            if after >= before:
                continue
            if mid_cd > mid_ab + 0.01 * length:
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
        F[f0, 0], F[f0, 1], F[f0, 2] = a, d, c
        F[f1, 0], F[f1, 1], F[f1, 2] = d, b, c
        lock[a] = lock[b] = lock[c] = lock[d] = True
        flips += 1
    return flips


@njit(parallel=True, cache=True)
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

def _flips(code, P, V, F, vcrease, h, ccos, features_only, passes):
    for _ in range(passes):
        mesh = Mesh(V, F)
        lock = np.zeros(len(V), dtype=bool)
        done = _flip_batch(code, P, V, F, mesh.E, mesh.valence, mesh.vf_start,
                           mesh.vf, lock, vcrease, h, ccos, features_only)
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


def remesh(sdf, verts, faces, max_len=None, min_len=None, tolerance=None,
           creases=True, crease_angle=30.0, crease_len=None, gradation=0.4,
           iterations=6, project=True):
    """Remesh a closed mesh of `sdf` adaptively.  Returns (verts, triangles).

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
    """
    program = sdf.program() if hasattr(sdf, 'program') else sdf
    V = np.ascontiguousarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if F.shape[1] == 4:
        from .mesh import triangulate
        F = triangulate(F, V, program).astype(np.int64)
    diag = np.linalg.norm(V.max(0) - V.min(0))
    max_len = max_len or diag / 40
    min_len = min_len or max_len / 10
    tolerance = tolerance or min_len / 4
    crease_len = crease_len or max_len / 4
    crease_cos = math.cos(math.radians(crease_angle)) if creases else None
    h_grad = 1e-4 * min_len
    code, P = program.code, program.params

    ccos = crease_cos if crease_cos is not None else -2.0   # -2: no creases
    for _ in range(iterations):
        # the one full crease scan per iteration; flags then travel with
        # the vertices through splits and collapses
        mesh = Mesh(V, F)
        _, vcrease = _crease_info(mesh, crease_cos, program, h_grad)
        L = _sizing(mesh, program, min_len, max_len, tolerance, crease_len,
                    vcrease, gradation)

        # turn edges that cut across creases (the input may have some)
        _flips(code, P, V, F, vcrease, h_grad, ccos, True, 3)

        # split until nothing is too long
        for _ in range(8):
            mesh = Mesh(V, F)
            ne = len(mesh.E)
            Vb = np.empty((len(V) + ne, 3)); Vb[:len(V)] = V
            Fb = np.empty((len(F) + 2 * ne, 3), dtype=np.int64); Fb[:len(F)] = F
            Lb = np.empty(len(V) + ne); Lb[:len(V)] = L
            cb = np.zeros(len(V) + ne, dtype=np.int64); cb[:len(V)] = vcrease
            lock = np.zeros(len(F), dtype=bool)
            nv, nf, done = _split_batch(code, P, Vb, Fb, len(V), len(F), mesh.E, Lb,
                                        lock, cb, h_grad, ccos)
            new = np.zeros(nv, dtype=bool); new[len(V):] = True
            V, F, L, vcrease = Vb[:nv], Fb[:nf], Lb[:nv], cb[:nv]
            if project and done:
                _project(code, P, V, new & (vcrease == 0), h_grad, 3)
                snap = new & (vcrease > 0)
                if snap.any():
                    m2 = Mesh(V, F)
                    _project_features(code, P, V, F, m2.vf_start, m2.vf, snap,
                                      np.full(len(V), 2), h_grad, 3)
            if done == 0:
                break

        # collapse until nothing is too short (or enough passes)
        for _ in range(6):
            mesh = Mesh(V, F)
            lock = np.zeros(len(V), dtype=bool)
            alive = np.ones(len(F), dtype=bool)
            stamp = np.zeros(len(V), dtype=np.int64)
            done = _collapse_batch(code, P, V, F, mesh.E, L, mesh.valence,
                                   mesh.vf_start, mesh.vf, lock, alive, stamp,
                                   vcrease, h_grad, ccos)
            F = F[alive]
            V, F, L, vcrease = _compact(V, F, L, vcrease)
            if done == 0:
                break

        # flip towards valence 6, and turn any edge that cuts a crease
        _flips(code, P, V, F, vcrease, h_grad, ccos, False, 3)

        # relax and project (crease edges looked up afresh for the sliding)
        mesh = Mesh(V, F)
        crease_e, vcrease = _crease_info(mesh, crease_cos, program, h_grad)
        target = np.empty_like(V)
        _relax(V, F, mesh.vf_start, mesh.vf, vcrease, mesh.E, crease_e, target)
        V = target
        if project:
            on_crease = vcrease > 0
            _project(code, P, V, ~on_crease, h_grad, 3)
            rank = np.where(vcrease > 2, 3, 2)
            _project_features(code, P, V, F, mesh.vf_start, mesh.vf, on_crease, rank,
                              h_grad, 3)

    return V, F.astype(np.int32)
