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
    request(node, signature, quick, full, delay, background=False)
    forget(node)

While a node's inputs keep changing (a slider being dragged), it hands out a
quick result straight away and arms a timer.  Each change pushes the timer
back; once nothing has changed for `delay` seconds the full result is built,
cached, and the node is re-run so it can output it.

    signature   anything comparable that identifies the inputs and settings
    quick       function returning the stand-in result (or None: keep the last one)
    full        function returning the real result
    delay       seconds of quiet before building the full result (0: always now)
    background  build the full result on a separate thread, so Blender stays
                responsive; if the inputs change while it builds, the result
                is thrown away instead of shown.  `full` must not touch bpy.
                Falls back to building on the main thread when numba's
                threading layer can't run kernels from two threads at once.
"""

import threading
import time
from functools import cache, partial

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
POLL = 0.05     # seconds between checks on a background build


@cache
def background_safe():
    import sdfnumba
    return sdfnumba.background_safe()


class _Job:
    """A full build running on its own thread."""

    def __init__(self, signature, full):
        self.signature = signature
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self._run, args=(full,), daemon=True)
        self.thread.start()

    def _run(self, full):
        try:
            self.result = full()
        except Exception as err:
            self.error = err

    @property
    def done(self):
        return not self.thread.is_alive()


def request(node, signature, quick, full, delay, background=False):
    n_id = node_id(node)
    st = _state.setdefault(n_id, {'armed': False, 'job': None})
    st['tree'], st['node'] = node.id_data.name, node.name

    if st.get('result_sig') == signature:
        return st['result']                     # settled, already built
    if delay <= 0:
        st['result'], st['result_sig'] = full(), signature
        return st['result']

    if st.get('sig') != signature:             # a real change: restart the wait
        st['changed'] = time.monotonic()
    st['sig'], st['full'], st['delay'] = signature, full, delay
    st['background'] = background
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

    job = st['job']
    if job is not None:
        if not job.done:
            return POLL
        st['job'] = None
        if job.error is not None:
            print(f"sdf_numba: building the full result failed: {job.error}")
        elif job.signature == st['sig']:
            st['armed'] = False                 # nothing changed while it built
            _deliver(st, job.result, job.signature)
            return None
        # otherwise the inputs changed while it built: drop it, wait again

    wait = st['delay'] - (time.monotonic() - st['changed'])
    if wait > 0:
        return wait                             # still changing: check again later

    if st['background'] and background_safe():
        st['job'] = _Job(st['sig'], st['full'])
        return POLL

    st['armed'] = False
    try:
        result = st['full']()
    except Exception as err:
        print(f"sdf_numba: building the full result failed: {err}")
        return None
    _deliver(st, result, st['sig'])
    return None


def _deliver(st, result, signature):
    """Cache the full result and re-run the node so it outputs it."""
    st['result'], st['result_sig'] = result, signature
    tree = bpy.data.node_groups.get(st['tree'])
    node = tree.nodes.get(st['node']) if tree else None
    if node is not None:
        node.process_node(None)


def forget(node):
    _state.pop(node_id(node), None)
