# This file is part of sdfnumba (https://github.com/zeffii/sdfnumba), MIT licence.

"""Settle-then-refine, shared by the SDF Mesh and SDF Remesh nodes.

While a node's inputs keep changing (a slider being dragged), it hands out a
quick result straight away and arms a timer.  Each change pushes the timer
back; once nothing has changed for `delay` seconds the full result is built,
cached, and the node is re-run so it can output it.

    result = request(node, signature, quick, full, delay)

signature   anything comparable that identifies the inputs and settings
quick       function returning the stand-in result (or None: keep the last one)
full        function returning the real result
delay       seconds of quiet before building the full result (0: always now)

Not a node module: it has no classes and no register(), so Sverchok's node
collector imports it and moves on.
"""

import time
from functools import partial

import bpy

from sverchok.data_structure import node_id

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
