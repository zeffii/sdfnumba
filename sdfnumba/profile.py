"""Profiles drawn as vertices, revolved into solids.

A profile is one or more closed loops of points lying in a plane.  It can be
given three ways, matching Sverchok's mesh data:

    verts               one loop, the points in order (closed automatically)
    verts + edges       any order; the edges are chained into closed loops,
                        then dropped.  Several loops are allowed: islands, or
                        holes when one loop lies inside another
    verts + polygons    each polygon's outline; where polygons share an edge
                        it is inside the profile and dropped, so adjacent
                        faces merge into one region.  Several separate
                        polygons give islands

All the points must lie in one plane.  The best-fitting plane is found and
any point off it is moved onto it; the result says how many moved and how
far, counting only offsets well beyond float32 rounding.

    shape, info = revolve_profile(loops, axis='Z')

revolves the profile around a world axis.  The profile's plane should
contain that axis (a profile in the XZ or YZ plane for Z): the solid is then
what Blender's Spin/Screw around the axis through the origin would give.  A
tilted plane is accepted too: the profile is revolved around the axis's
projection onto its plane, through the point of the plane nearest the origin.
"""

from dataclasses import dataclass, field

import numpy as np

from . import api

AXES = {'X': np.array((1.0, 0.0, 0.0)), 'Y': np.array((0.0, 1.0, 0.0)),
        'Z': np.array((0.0, 0.0, 1.0))}

# offsets below this share of the profile's size are float32 rounding, not
# a vertex genuinely off the plane (float32 is good to ~1e-7 of a value)
NOISE = 1e-5
AXIS_SNAP = np.cos(np.radians(1.0))


# --- loops --------------------------------------------------------------------------

def loops_from_points(verts):
    """One closed loop, the points in order."""
    pts = np.asarray(verts, dtype=float).reshape(-1, 3)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]                    # the closing point was repeated
    if len(pts) < 3:
        raise ValueError('a profile needs at least 3 vertices')
    return [pts]


def loops_from_edges(verts, edges):
    """Chain edges into closed loops.  Every vertex used must have exactly
    two edges; unused vertices are ignored."""
    pts = np.asarray(verts, dtype=float).reshape(-1, 3)
    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    E = E[E[:, 0] != E[:, 1]]
    E = np.unique(np.sort(E, axis=1), axis=0)      # drop repeated edges
    neighbours = {}
    for a, b in E:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    odd = [v for v, n in neighbours.items() if len(n) != 2]
    if odd:
        v = odd[0]
        raise ValueError(f'vertex {v} has {len(neighbours[v])} edges; a profile must be '
                         f'closed loops, where every vertex has exactly 2')
    loops, seen = [], set()
    for start in sorted(neighbours):
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            a, b = neighbours[cur]
            nxt = b if a == prev else a
            if nxt == start:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) < 3:
            raise ValueError('a loop in the profile has fewer than 3 vertices')
        loops.append(pts[loop])
    if not loops:
        raise ValueError('no closed loops in the profile')
    return loops


def loops_from_polygons(verts, polygons):
    """The outline of the polygons: edges used by exactly one polygon."""
    edges = []
    for poly in polygons:
        poly = list(poly)
        edges += [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    E = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    keys, counts = np.unique(E, axis=0, return_counts=True)
    return loops_from_edges(verts, keys[counts == 1])


# --- plane --------------------------------------------------------------------------

@dataclass
class ProfileInfo:
    normal: np.ndarray            # the profile plane's unit normal
    moved: int = 0                # vertices moved onto the plane
    max_offset: float = 0.0       # how far the furthest of them was
    tilted: bool = False          # plane not perpendicular to a world axis
    loops: int = 0
    mirrored: bool = False        # the profile crosses the axis
    notes: list = field(default_factory=list)


def fit_plane(points):
    """(centre, unit normal, signed offset of each point) of the best plane.

    Profiles are nearly always drawn in a world plane (XZ, YZ ...), so a fit
    within about a degree of one is taken to be it, placed at the median
    point: then a stray vertex is the only one moved, rather than the plane
    tilting a little towards it and every vertex moving."""
    P = np.asarray(points, dtype=float)
    centre = P.mean(0)
    _, sv, vt = np.linalg.svd(P - centre, full_matrices=False)
    n = vt[-1]
    k = np.argmax(np.abs(n))
    if abs(n[k]) > AXIS_SNAP:
        n = np.zeros(3)
        n[k] = 1.0
        centre = centre.copy()
        centre[k] = np.median(P[:, k])
    return centre, n, (P - centre) @ n


def flatten(loops):
    """Move every point onto the profile's plane.  Returns the flattened
    loops, the plane (centre, normal) and a ProfileInfo."""
    allp = np.concatenate(loops)
    centre, n, off = fit_plane(allp)
    size = np.linalg.norm(allp.max(0) - allp.min(0))
    if size == 0 or np.linalg.svd(allp - centre, compute_uv=False)[1] < NOISE * size:
        raise ValueError('the profile has no area: all its points lie on a line')
    tol = NOISE * size
    info = ProfileInfo(normal=n, loops=len(loops))
    big = np.abs(off) > tol
    info.moved = int(big.sum())
    info.max_offset = float(np.abs(off).max()) if info.moved else 0.0
    info.tilted = bool(np.abs(n).max() < 1.0)
    out, i = [], 0
    for loop in loops:
        k = len(loop)
        out.append(loop - np.outer(off[i:i + k], n))
        i += k
    return out, centre, n, info


# --- revolving ----------------------------------------------------------------------

def _segments(loops2d):
    segs = []
    for loop in loops2d:
        loop = np.asarray(loop, dtype=float)
        segs.append(np.concatenate([loop, np.roll(loop, -1, axis=0)], axis=1))
    return np.concatenate(segs)


def _prim(segs):
    return api.Prim('polygon', len(segs), *segs.ravel())


def polygon(loops2d):
    """A 2D profile from closed loops of (x, y) points (even-odd inside)."""
    return _prim(_segments(loops2d))


def _mirrored(loops2d, tol):
    """Segments of the profile plus its mirror image across x = 0, without
    the segments lying on x = 0: an edge drawn along the revolve axis is not
    a surface of the solid, and must not count as one."""
    segs = _segments(loops2d)
    segs = np.concatenate([segs, segs * (-1.0, 1.0, -1.0, 1.0)])
    on_axis = (np.abs(segs[:, 0]) <= tol) & (np.abs(segs[:, 2]) <= tol)
    return segs[~on_axis]


def revolve_profile(loops, axis='Z'):
    """Revolve closed loops of 3D points around a world axis ('X', 'Y' or
    'Z', or a direction).  Returns (shape, ProfileInfo)."""
    flat, centre, n, info = flatten([np.asarray(l, dtype=float).reshape(-1, 3) for l in loops])
    a = AXES[axis.upper()] if isinstance(axis, str) else np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    u = a - (a @ n) * n                          # the axis, laid into the plane
    if np.linalg.norm(u) < 1e-3:
        raise ValueError('the profile lies across the revolve axis (its plane is '
                         'perpendicular to it): draw it in a plane containing the axis')
    u /= np.linalg.norm(u)
    w = np.cross(n, u)                           # in the plane, away from the axis
    origin = (centre @ n) * n                    # the plane's point nearest the world origin
    loops2d = [np.stack([(l - origin) @ w, (l - origin) @ u], 1) for l in flat]
    r = np.concatenate([l[:, 0] for l in loops2d])
    tol = NOISE * max(np.abs(r).max(), 1e-12)
    if info.tilted:
        info.notes.append("the profile's plane is tilted: it is revolved around the "
                          "axis's projection onto that plane")
    if r.min() >= -tol or r.max() <= tol:
        # one side of the axis (either side sweeps the same solid)
        solid = api.Transform(_prim(_mirrored(loops2d, tol)), [('revolve', (0.0,))])
    else:
        # both sides: each side sweeps its own solid, the result is both
        info.mirrored = True
        profile = polygon(loops2d)
        mirror = np.diag((-1.0, 1.0, 1.0)).ravel()
        solid = api.union(api.Transform(profile, [('revolve', (0.0,))]),
                          api.Transform(profile, [('revolve', (0.0,)), ('matrix', mirror)]))
    # local frame -> world: local x along w, local z along the axis u
    R = np.stack([w, np.cross(u, w), u], axis=1)
    shape = api.Transform(solid, [('translate', origin), ('matrix', R.ravel())])
    return shape, info


def profile_loops(verts, edges=None, polygons=None):
    """Loops from Sverchok-style mesh data: polygons if given, else edges,
    else the vertices in order."""
    if polygons is not None and len(polygons):
        return loops_from_polygons(verts, polygons)
    if edges is not None and len(edges):
        return loops_from_edges(verts, edges)
    return loops_from_points(verts)
