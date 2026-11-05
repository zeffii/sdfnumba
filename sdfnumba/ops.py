"""Every SDF operation, defined once.

Each op is a small numba function with a fixed signature so that both the
bytecode interpreter (interp.py) and, later, the code generator can call it
the same way.  Parameters live in one flat float array `P`; an op reads its
own values starting at offset `o`.

Kinds of op (what the function receives and returns):

    PRIM        f(x, y, z, P, o)        -> d          push a distance
    XFORM       f(x, y, z, P, o)        -> x, y, z    move the current point
    XFORM_AUX   f(x, y, z, P, o)        -> x, y, z, a also push `a` to the aux stack
    DUNARY      f(d, P, o)              -> d          change the top distance
    DUNARY_AUX  f(d, a, P, o)           -> d          ... using a popped aux value
    DBINARY     f(d1, d2, P, o)         -> d          combine the top two distances

To add an op: write the function, decorate it with @op(KIND, nparams), and
give the API in api.py a way to build it.  Nothing else needs changing.
"""

import math
from numba import njit

PRIM, XFORM, XFORM_AUX, DUNARY, DUNARY_AUX, DBINARY = range(6)

# Control opcodes, handled by the interpreter itself.
PUSHP, POPP = 0, 1
FIRST_OP_CODE = 2


class Op:
    def __init__(self, name, kind, nparams, fn, code, winner=None):
        self.name = name
        self.kind = kind
        self.nparams = nparams
        self.fn = fn
        self.code = code
        self.winner = winner   # DBINARY only: name of f(d1, d2, P, o) -> True if d1's side won


REGISTRY = {}          # name -> Op, in definition order
REGISTRY_BY_CODE = {}  # opcode -> Op


def op(kind, nparams, winner=None):
    """Register an op.  Binary ops also name a `winner` function, which says
    whether the first or second operand decided the result at that point;
    that is how each point knows which primitive it belongs to (used for
    crease detection and, in the viewer, colouring)."""
    def decorate(py_fn):
        fn = njit(cache=True, fastmath=True)(py_fn)
        code = FIRST_OP_CODE + len(REGISTRY)
        REGISTRY[py_fn.__name__] = Op(py_fn.__name__, kind, nparams, fn, code,
                                      winner.__name__ if winner is not None else None)
        REGISTRY_BY_CODE[code] = REGISTRY[py_fn.__name__]
        return fn
    return decorate


# --- helpers -----------------------------------------------------------

@njit(cache=True, fastmath=True)
def _clamp(v, lo, hi):
    return min(max(v, lo), hi)


@njit(cache=True, fastmath=True)
def _len3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


# --- primitives --------------------------------------------------------

@op(PRIM, 1)
def sphere(x, y, z, P, o):
    return _len3(x, y, z) - P[o]


@op(PRIM, 6)
def plane(x, y, z, P, o):
    # normal (unit), point on plane
    nx, ny, nz, px, py, pz = P[o], P[o + 1], P[o + 2], P[o + 3], P[o + 4], P[o + 5]
    return (px - x) * nx + (py - y) * ny + (pz - z) * nz


@op(PRIM, 3)
def box(x, y, z, P, o):
    qx = abs(x) - P[o] * 0.5
    qy = abs(y) - P[o + 1] * 0.5
    qz = abs(z) - P[o + 2] * 0.5
    outside = _len3(max(qx, 0.0), max(qy, 0.0), max(qz, 0.0))
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside


@op(PRIM, 4)
def rounded_box(x, y, z, P, o):
    r = P[o + 3]
    qx = abs(x) - P[o] * 0.5 + r
    qy = abs(y) - P[o + 1] * 0.5 + r
    qz = abs(z) - P[o + 2] * 0.5 + r
    outside = _len3(max(qx, 0.0), max(qy, 0.0), max(qz, 0.0))
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside - r


@op(PRIM, 2)
def torus(x, y, z, P, o):
    a = math.sqrt(x * x + y * y) - P[o]
    return math.sqrt(a * a + z * z) - P[o + 1]


@op(PRIM, 7)
def capsule(x, y, z, P, o):
    ax, ay, az, bx, by, bz, r = (P[o], P[o + 1], P[o + 2],
                                 P[o + 3], P[o + 4], P[o + 5], P[o + 6])
    pax, pay, paz = x - ax, y - ay, z - az
    bax, bay, baz = bx - ax, by - ay, bz - az
    h = _clamp((pax * bax + pay * bay + paz * baz) /
               (bax * bax + bay * bay + baz * baz), 0.0, 1.0)
    return _len3(pax - bax * h, pay - bay * h, paz - baz * h) - r


@op(PRIM, 1)
def cylinder(x, y, z, P, o):
    # infinite, along z
    return math.sqrt(x * x + y * y) - P[o]


@op(PRIM, 7)
def capped_cylinder(x, y, z, P, o):
    ax, ay, az, bx, by, bz, r = (P[o], P[o + 1], P[o + 2],
                                 P[o + 3], P[o + 4], P[o + 5], P[o + 6])
    bax, bay, baz = bx - ax, by - ay, bz - az
    pax, pay, paz = x - ax, y - ay, z - az
    baba = bax * bax + bay * bay + baz * baz
    paba = pax * bax + pay * bay + paz * baz
    dx = _len3(pax * baba - bax * paba, pay * baba - bay * paba,
               paz * baba - baz * paba) - r * baba
    dy = abs(paba - baba * 0.5) - baba * 0.5
    x2 = dx * dx
    y2 = dy * dy * baba
    if max(dx, dy) < 0.0:
        d = -min(x2, y2)
    else:
        d = (x2 if dx > 0.0 else 0.0) + (y2 if dy > 0.0 else 0.0)
    s = 1.0 if d > 0.0 else (-1.0 if d < 0.0 else 0.0)
    return s * math.sqrt(abs(d)) / baba


@op(PRIM, 3)
def rounded_cylinder(x, y, z, P, o):
    ra, rb, h = P[o], P[o + 1], P[o + 2]
    dx = math.sqrt(x * x + y * y) - ra + rb
    dy = abs(z) - h * 0.5 + rb
    return min(max(dx, dy), 0.0) + math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2) - rb


@op(PRIM, 8)
def capped_cone(x, y, z, P, o):
    ax, ay, az, bx, by, bz, ra, rb = (P[o], P[o + 1], P[o + 2], P[o + 3],
                                      P[o + 4], P[o + 5], P[o + 6], P[o + 7])
    rba = rb - ra
    bax, bay, baz = bx - ax, by - ay, bz - az
    pax, pay, paz = x - ax, y - ay, z - az
    baba = bax * bax + bay * bay + baz * baz
    papa = pax * pax + pay * pay + paz * paz
    paba = (pax * bax + pay * bay + paz * baz) / baba
    q = math.sqrt(max(papa - paba * paba * baba, 0.0))
    cax = max(0.0, q - (ra if paba < 0.5 else rb))
    cay = abs(paba - 0.5) - 0.5
    k = rba * rba + baba
    f = _clamp((rba * (q - ra) + paba * baba) / k, 0.0, 1.0)
    cbx = q - ra - f * rba
    cby = paba - f
    s = -1.0 if (cbx < 0.0 and cay < 0.0) else 1.0
    return s * math.sqrt(min(cax * cax + cay * cay * baba,
                             cbx * cbx + cby * cby * baba))


@op(PRIM, 3)
def rounded_cone(x, y, z, P, o):
    r1, r2, h = P[o], P[o + 1], P[o + 2]
    qx = math.sqrt(x * x + y * y)
    qy = z
    b = (r1 - r2) / h
    a = math.sqrt(1.0 - b * b)
    k = qx * -b + qy * a
    if k < 0.0:
        return math.sqrt(qx * qx + qy * qy) - r1
    if k > a * h:
        return math.sqrt(qx * qx + (qy - h) ** 2) - r2
    return qx * a + qy * b - r1


@op(PRIM, 3)
def ellipsoid(x, y, z, P, o):
    sx, sy, sz = P[o], P[o + 1], P[o + 2]
    k0 = _len3(x / sx, y / sy, z / sz)
    k1 = _len3(x / (sx * sx), y / (sy * sy), z / (sz * sz))
    if k1 == 0.0:
        return -min(sx, min(sy, sz))
    return k0 * (k0 - 1.0) / k1


@op(PRIM, 1)
def tetrahedron(x, y, z, P, o):
    return (max(abs(x + y) - z, abs(x - y) + z) - P[o]) / math.sqrt(3.0)


@op(PRIM, 1)
def octahedron(x, y, z, P, o):
    return (abs(x) + abs(y) + abs(z) - P[o]) * 0.5773502691896258


# --- point transforms --------------------------------------------------

@op(XFORM, 3)
def translate(x, y, z, P, o):
    return x - P[o], y - P[o + 1], z - P[o + 2]


@op(XFORM, 3)
def scale_point(x, y, z, P, o):
    # divide by the scale; the distance is fixed up by `scale_dist`
    return x / P[o], y / P[o + 1], z / P[o + 2]


@op(XFORM, 9)
def matrix(x, y, z, P, o):
    # row vector times 3x3 matrix (rotation), as fogleman's np.dot(p, m)
    return (x * P[o] + y * P[o + 3] + z * P[o + 6],
            x * P[o + 1] + y * P[o + 4] + z * P[o + 7],
            x * P[o + 2] + y * P[o + 5] + z * P[o + 8])


@op(XFORM, 1)
def twist(x, y, z, P, o):
    c = math.cos(P[o] * z)
    s = math.sin(P[o] * z)
    return c * x - s * y, s * x + c * y, z


@op(XFORM, 1)
def bend(x, y, z, P, o):
    c = math.cos(P[o] * x)
    s = math.sin(P[o] * x)
    return c * x - s * y, s * x + c * y, z


@op(XFORM, 9)
def repeat(x, y, z, P, o):
    # spacing (0 = no repeat on that axis), count (<0 = unlimited),
    # neighbour offset (for padding > 0)
    sx, sy, sz = P[o], P[o + 1], P[o + 2]
    cx, cy, cz = P[o + 3], P[o + 4], P[o + 5]
    nx, ny, nz = P[o + 6], P[o + 7], P[o + 8]
    if sx != 0.0:
        i = round(x / sx)
        if cx >= 0.0:
            i = _clamp(i, -cx, cx)
        x = x - sx * (i + nx)
    if sy != 0.0:
        i = round(y / sy)
        if cy >= 0.0:
            i = _clamp(i, -cy, cy)
        y = y - sy * (i + ny)
    if sz != 0.0:
        i = round(z / sz)
        if cz >= 0.0:
            i = _clamp(i, -cz, cz)
        z = z - sz * (i + nz)
    return x, y, z


@op(XFORM, 2)
def polar_fold(x, y, z, P, o):
    # one of the two copies used by circular_array: angle step, shift (0 or 1)
    da, shift = P[o], P[o + 1]
    d = math.hypot(x, y)
    a = math.atan2(y, x) % da - shift * da
    return math.cos(a) * d, math.sin(a) * d, z


@op(XFORM_AUX, 3)
def elongate(x, y, z, P, o):
    qx = abs(x) - P[o]
    qy = abs(y) - P[o + 1]
    qz = abs(z) - P[o + 2]
    w = min(max(qx, max(qy, qz)), 0.0)
    return max(qx, 0.0), max(qy, 0.0), max(qz, 0.0), w


# --- distance modifiers ------------------------------------------------

@op(DUNARY, 1)
def scale_dist(d, P, o):
    return d * P[o]


@op(DUNARY, 0)
def negate(d, P, o):
    return -d


@op(DUNARY, 1)
def offset(d, P, o):
    # dilate uses -r, erode uses +r
    return d + P[o]


@op(DUNARY, 1)
def shell(d, P, o):
    return abs(d) - P[o] * 0.5


@op(DUNARY_AUX, 0)
def add_aux(d, a, P, o):
    return d + a


# --- combiners (k = 0 means hard) --------------------------------------

@njit(cache=True, fastmath=True)
def first_if_smaller(d1, d2, P, o):
    return d1 <= d2


@njit(cache=True, fastmath=True)
def first_if_larger(d1, d2, P, o):
    return d1 >= d2


@njit(cache=True, fastmath=True)
def first_if_not_cut(d1, d2, P, o):
    return d1 >= -d2


@njit(cache=True, fastmath=True)
def first_if_heavier(d1, d2, P, o):
    return P[o] <= 0.5


@op(DBINARY, 1, winner=first_if_smaller)
def union(d1, d2, P, o):
    k = P[o]
    if k == 0.0:
        return min(d1, d2)
    h = _clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 + (d1 - d2) * h - k * h * (1.0 - h)


@op(DBINARY, 1, winner=first_if_not_cut)
def difference(d1, d2, P, o):
    k = P[o]
    if k == 0.0:
        return max(d1, -d2)
    h = _clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0)
    return d1 + (-d2 - d1) * h + k * h * (1.0 - h)


@op(DBINARY, 1, winner=first_if_larger)
def intersection(d1, d2, P, o):
    k = P[o]
    if k == 0.0:
        return max(d1, d2)
    h = _clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
    return d2 + (d1 - d2) * h + k * h * (1.0 - h)


@op(DBINARY, 1, winner=first_if_heavier)
def blend(d1, d2, P, o):
    k = P[o]
    return k * d2 + (1.0 - k) * d1
