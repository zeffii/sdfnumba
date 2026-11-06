"""The user-facing API, modelled on fogleman's sdf.

    from sdfnumba import *
    f = sphere(1) & box(1.5)
    f -= cylinder(0.5).orient(X) | cylinder(0.5)
    verts, faces = f.generate()

Building a shape only builds a tree of nodes; nothing is evaluated until the
tree is compiled to bytecode (ir.py) and run.
"""

import itertools

import numpy as np

from . import ir

ORIGIN = np.array((0.0, 0.0, 0.0))
X = np.array((1.0, 0.0, 0.0))
Y = np.array((0.0, 1.0, 0.0))
Z = np.array((0.0, 0.0, 1.0))
UP = Z


def _vec3(v):
    a = np.asarray(v, dtype=float)
    if a.ndim == 0:
        a = np.array((a, a, a), dtype=float)
    return a


def _normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# --- nodes -------------------------------------------------------------

class SDF:
    _k = None

    def emit(self, e):
        raise NotImplementedError

    def program(self):
        return ir.compile_tree(self)

    def __call__(self, points):
        from .grid import evaluate
        return evaluate(self.program(), points)

    def ids(self, points):
        """Which primitive (0, 1, 2 ... in build order) the surface nearest
        each point belongs to; boolean seams are where it changes."""
        from .grid import primitive_ids
        return primitive_ids(self.program(), points)

    def compile(self):
        """Compile this shape's generated code now and wait for it.  Without
        this it compiles in the background on first use."""
        self.program().compile()
        return self

    def __or__(self, other):
        return union(self, other)

    def __and__(self, other):
        return intersection(self, other)

    def __sub__(self, other):
        return difference(self, other)

    def k(self, k=None):
        self._k = k
        return self

    def generate(self, **kwargs):
        from .mesh import generate
        return generate(self, **kwargs)


class Prim(SDF):
    def __init__(self, name, *values):
        self.name = name
        self.values = values

    def emit(self, e):
        e.op(self.name, *self.values)


class Transform(SDF):
    """Move the point, evaluate the child, restore the point, then apply any
    distance fix-ups (`after`)."""

    def __init__(self, child, before, after=()):
        self.child = child
        self.before = before
        self.after = after

    def emit(self, e):
        e.push_point()
        for name, values in self.before:
            e.op(name, *values)
        self.child.emit(e)
        e.pop_point()
        for name, values in self.after:
            e.op(name, *values)


class Modify(SDF):
    def __init__(self, child, name, *values):
        self.child = child
        self.name = name
        self.values = values

    def emit(self, e):
        self.child.emit(e)
        e.op(self.name, *self.values)


class Combine(SDF):
    def __init__(self, name, first, rest):
        self.name = name
        self.first = first
        self.rest = rest   # [(node, k)]

    def emit(self, e):
        self.first.emit(e)
        for node, k in self.rest:
            node.emit(e)
            e.op(self.name, k)


class Copies(SDF):
    """Several transformed copies of one child, hard-unioned."""

    def __init__(self, child, copies):
        self.child = child
        self.copies = copies   # [[(name, values), ...], ...]

    def emit(self, e):
        for i, steps in enumerate(self.copies):
            Transform(self.child, steps).emit(e)
            if i:
                e.op('union', 0.0)


# --- primitives --------------------------------------------------------

def _at(node, center):
    center = _vec3(center)
    return translate(node, center) if np.any(center) else node


def sphere(radius=1, center=ORIGIN):
    return _at(Prim('sphere', radius), center)


def plane(normal=UP, point=ORIGIN):
    return Prim('plane', *_normalize(normal), *_vec3(point))


def slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None):
    fs = []
    if x0 is not None: fs.append(plane(X, (x0, 0, 0)))
    if x1 is not None: fs.append(plane(-X, (x1, 0, 0)))
    if y0 is not None: fs.append(plane(Y, (0, y0, 0)))
    if y1 is not None: fs.append(plane(-Y, (0, y1, 0)))
    if z0 is not None: fs.append(plane(Z, (0, 0, z0)))
    if z1 is not None: fs.append(plane(-Z, (0, 0, z1)))
    return intersection(*fs, k=k)


def box(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a, b = _vec3(a), _vec3(b)
        return box(b - a, a + (b - a) / 2)
    return _at(Prim('box', *_vec3(size)), center)


def rounded_box(size, radius):
    return Prim('rounded_box', *_vec3(size), radius)


def torus(r1, r2):
    return Prim('torus', r1, r2)


def capsule(a, b, radius):
    return Prim('capsule', *_vec3(a), *_vec3(b), radius)


def cylinder(radius):
    return Prim('cylinder', radius)


def capped_cylinder(a, b, radius):
    return Prim('capped_cylinder', *_vec3(a), *_vec3(b), radius)


def rounded_cylinder(ra, rb, h):
    return Prim('rounded_cylinder', ra, rb, h)


def capped_cone(a, b, ra, rb):
    return Prim('capped_cone', *_vec3(a), *_vec3(b), ra, rb)


def rounded_cone(r1, r2, h):
    return Prim('rounded_cone', r1, r2, h)


def ellipsoid(size):
    return Prim('ellipsoid', *_vec3(size))


def tetrahedron(r):
    return Prim('tetrahedron', r)


def octahedron(r):
    return Prim('octahedron', r)


# --- 2D profiles (x and y only; see revolve) -------------------------------

def arc(radius, start_angle, end_angle):
    """A circular arc in the xy plane, angles in radians from the x axis.
    It has no thickness: use .dilate(t) for a wall of 2t."""
    return Prim('arc', radius, start_angle, end_angle)


# --- booleans and distance modifiers ------------------------------------

def _combine(name, a, bs, k):
    rest = [(b, k or b._k or 0.0) for b in bs]
    return Combine(name, a, rest)


def union(a, *bs, k=None):
    return _combine('union', a, bs, k)


def difference(a, *bs, k=None):
    return _combine('difference', a, bs, k)


def intersection(a, *bs, k=None):
    return _combine('intersection', a, bs, k)


def blend(a, *bs, k=0.5):
    return _combine('blend', a, bs, k)


def negate(other):
    return Modify(other, 'negate')


def dilate(other, r):
    return Modify(other, 'offset', -r)


def erode(other, r):
    return Modify(other, 'offset', r)


def shell(other, thickness):
    return Modify(other, 'shell', thickness)


# --- positioning and alterations ----------------------------------------

def translate(other, offset):
    return Transform(other, [('translate', _vec3(offset))])


def scale(other, factor):
    s = _vec3(factor)
    return Transform(other, [('scale_point', s)], [('scale_dist', (s.min(),))])


def _rotation(angle, vector):
    x, y, z = _normalize(vector)
    s, c = np.sin(angle), np.cos(angle)
    m = 1 - c
    return np.array([
        [m*x*x + c, m*x*y + z*s, m*z*x - y*s],
        [m*x*y - z*s, m*y*y + c, m*y*z + x*s],
        [m*z*x + y*s, m*y*z - x*s, m*z*z + c],
    ]).T


def rotate(other, angle, vector=Z):
    return Transform(other, [('matrix', _rotation(angle, vector).ravel())])


def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        return np.cross(v, (0, 1, 0))
    return np.cross(v, (1, 0, 0))


def rotate_to(other, a, b):
    a, b = _normalize(a), _normalize(b)
    dot = np.dot(b, a)
    if dot >= 1:
        return other
    if dot <= -1:
        return rotate(other, np.pi, _perpendicular(a))
    return rotate(other, np.arccos(dot), _normalize(np.cross(b, a)))


def orient(other, axis):
    return rotate_to(other, UP, axis)


def circular_array(other, count, offset=0):
    if offset:
        other = translate(other, X * offset)
    da = 2 * np.pi / count
    return Copies(other, [[('polar_fold', (da, 1.0))], [('polar_fold', (da, 0.0))]])


def repeat(other, spacing, count=None, padding=0):
    spacing = _vec3(spacing)
    count = np.full(3, -1.0) if count is None else _vec3(count)
    pad = [0 if s == 0 else int(p) for s, p in zip(spacing, _vec3(padding))]
    axes = [range(-p, p + 1) for p in pad]
    copies = [[('repeat', (*spacing, *count, *n))] for n in itertools.product(*axes)]
    return Copies(other, copies)


def elongate(other, size):
    return Transform(other, [('elongate', _vec3(size))], [('add_aux', ())])


def revolve(other, offset=0):
    """Spin a 2D profile (drawn in the xy plane) around the z axis, `offset`
    away from it.  Transform the profile first to place it; transform the
    result to place the solid."""
    return Transform(other, [('revolve', (offset,))])


def twist(other, k):
    return Transform(other, [('twist', (k,))])


def bend(other, k):
    return Transform(other, [('bend', (k,))])


# Methods, so shapes chain like fogleman's: sphere(1).translate(X).k(0.2)
for _f in (translate, scale, rotate, rotate_to, orient, circular_array, repeat,
           elongate, revolve, twist, bend, negate, dilate, erode, shell,
           union, difference, intersection, blend):
    setattr(SDF, _f.__name__, _f)
