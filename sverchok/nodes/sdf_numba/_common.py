# This file is part of sdfnumba (https://github.com/zeffii/sdfnumba), MIT licence.

"""Small helpers shared by the sdf_numba nodes.  Not a node module."""

import numpy as np


def import_sdfnumba():
    """Import sdfnumba on first use (the very first import compiles for a while)."""
    try:
        import sdfnumba
    except ImportError as e:
        raise ImportError('sdfnumba is not installed in Blender\'s Python '
                          '(see sverchok/README.md in the sdfnumba repo)') from e
    return sdfnumba


def first_shape(socket):
    """The first sdfnumba shape on a socket, however nested."""
    stack = [socket.sv_get(deepcopy=False, default=[])]
    while stack:
        item = stack.pop(0)
        if hasattr(item, 'program'):
            return item
        if isinstance(item, (list, tuple)):
            stack.extend(item)
    return None


def shape_signature(shape):
    """Identifies a shape by its structure and parameter values."""
    program = shape.program()
    return program.key + program.params.tobytes()


def unique_edges(faces):
    """(n, 2) edges of a mesh of equal-sized faces, each edge once."""
    faces = np.asarray(faces)
    e = np.stack([faces, np.roll(faces, -1, axis=1)], -1).reshape(-1, 2)
    return np.unique(np.sort(e, axis=1), axis=0)


def mesh_outputs(verts, faces):
    """Sverchok-style outputs: one object each of verts, edges, polygons."""
    return [np.asarray(verts)], [unique_edges(faces).tolist()], [np.asarray(faces).tolist()]


def faces_array(polygons):
    """Polygons from a socket as an (m, 3) or (m, 4) int array; mixed
    polygons are split into triangle fans."""
    try:
        arr = np.asarray(polygons, dtype=np.int64)
        if arr.ndim == 2 and arr.shape[1] in (3, 4):
            return arr
    except ValueError:
        pass
    tris = [(p[0], p[i], p[i + 1]) for p in polygons for i in range(1, len(p) - 1)]
    return np.asarray(tris, dtype=np.int64)
