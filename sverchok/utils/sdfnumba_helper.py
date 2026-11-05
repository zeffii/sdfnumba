# This file is part of sdfnumba (https://github.com/zeffii/sdfnumba), MIT licence.
# Symlink or copy it into sverchok/utils/; the nodes in nodes/sdf_numba import it as
# sverchok.utils.sdfnumba_helper.

"""Helpers shared by the sdf_numba nodes.

Shapes and meshes on sockets:
    first_shape(socket)        the first sdfnumba shape on a socket
    shape_signature(shape)     identifies a shape (structure and values)
    mesh_outputs(verts, faces) verts, edges, polygons in Sverchok's layout
    faces_array(polygons)      polygons from a socket as an int array

Settle-then-refine (SDF Mesh, SDF Remesh):
    request(node, signature, quick, full, delay)
    forget(node)

While a node's inputs keep changing (a slider being dragged), it hands out a
quick result straight away and arms a timer.  Each change pushes the timer
back; once nothing has changed for `delay` seconds the full result is built,
cached, and the node is re-run so it can output it.

    signature   anything comparable that identifies the inputs and settings
    quick       function returning the stand-in result (or None: keep the last one)
    full        function returning the real result
    delay       seconds of quiet before building the full result (0: always now)
"""

import time
from functools import partial

import bpy
import numpy as np

from sverchok.data_structure import node_id


# --- shapes and meshes ---------------------------------------------------------

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


# --- settle-then-refine ----------------------------------------------------------

_state = {}     # node id -> dict


def request(node, signature, quick, full, delay):
    n_id = node_id(node)
    st = _state.setdefault(n_id, {'armed': False})
    st['tree'], st['node'] = node.id_data.name, node.name

    if st.get('result_sig') == signature:
        return st['result']                     # settled, already built
    if delay <= 0:
        st['result'], st['result_sig'] = full(), signature
        return st['result']

    st['sig'], st['full'], st['delay'] = signature, full, delay
    st['changed'] = time.monotonic()
    if not st['armed']:
        st['armed'] = True
        bpy.app.timers.register(partial(_tick, n_id), first_interval=delay)
    if quick is not None:
        return quick()
    return st.get('result')                     # last full result, until the new one


def _tick(n_id):
    st = _state.get(n_id)
    if st is None:
        return None
    wait = st['delay'] - (time.monotonic() - st['changed'])
    if wait > 0:
        return wait                             # still changing: check again later
    st['armed'] = False
    try:
        st['result'], st['result_sig'] = st['full'](), st['sig']
    except Exception as err:
        print(f"sdf_numba: building the full result failed: {err}")
        return None
    tree = bpy.data.node_groups.get(st['tree'])
    node = tree.nodes.get(st['node']) if tree else None
    if node is not None:
        node.process_node(None)                 # outputs the cached result
    return None


def forget(node):
    _state.pop(node_id(node), None)
