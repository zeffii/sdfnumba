"""The Sverchok mesh and remesh nodes, run against small stand-ins for bpy
and Sverchok (just enough of them), to check the settle-then-refine logic
and the outputs.  The viewer needs a real GPU context, so it isn't covered."""

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

import sdfnumba as sn

NODE_DIR = os.path.join(os.path.dirname(__file__), '..', 'sverchok', 'nodes', 'sdf_numba')
UTILS_DIR = os.path.join(os.path.dirname(__file__), '..', 'sverchok', 'utils')


# --- stand-ins ------------------------------------------------------------------

class Timers:
    def __init__(self):
        self.pending = []

    def register(self, fn, first_interval=0.0):
        self.pending.append(fn)

    def run_all(self):
        """Run every armed timer as if its time had come (and gone)."""
        while self.pending:
            fn = self.pending.pop(0)
            again = fn()
            if again is not None:
                self.pending.append(fn)


class Socket:
    def __init__(self, name):
        self.name = name
        self.data = None
        self.is_linked = False

    def sv_get(self, deepcopy=False, default=None):
        return self.data if self.data is not None else default

    def sv_set(self, data):
        self.data = data
        self.is_linked = True


class Sockets(list):
    def new(self, kind, name):
        s = Socket(name)
        self.append(s)
        return s

    def __getitem__(self, key):
        if isinstance(key, str):
            return next(s for s in self if s.name == key)
        return list.__getitem__(self, key)


def _install_stubs(timers, clock):
    bpy = types.ModuleType('bpy')
    bpy.app = types.SimpleNamespace(timers=timers, background=False)
    bpy.types = types.SimpleNamespace(Node=type('Node', (), {}))
    bpy.utils = types.SimpleNamespace(register_classes_factory=lambda c: (lambda: None, lambda: None))
    bpy.data = types.SimpleNamespace(node_groups={})
    props = types.ModuleType('bpy.props')
    for name in ('BoolProperty', 'FloatProperty', 'IntProperty', 'EnumProperty',
                 'FloatVectorProperty', 'StringProperty'):
        setattr(props, name, lambda **kw: kw.get('default'))
    bpy.props = props
    sys.modules['bpy'] = bpy
    sys.modules['bpy.props'] = props

    sv = types.ModuleType('sverchok')
    sv.__path__ = []
    ds = types.ModuleType('sverchok.data_structure')
    ds.updateNode = lambda self, context: None
    ds.node_id = lambda node: node.node_id
    nt = types.ModuleType('sverchok.node_tree')
    class SverchCustomTreeNode:
        def warning(self, message):            # Sverchok shows these on the node
            self.__dict__.setdefault('warnings', []).append(message)
    nt.SverchCustomTreeNode = SverchCustomTreeNode
    nodes = types.ModuleType('sverchok.nodes')
    nodes.__path__ = []
    utils = types.ModuleType('sverchok.utils')
    utils.__path__ = [UTILS_DIR]          # where sdfnumba_helper.py is symlinked to
    for m in (sv, ds, nt, nodes, utils):
        sys.modules[m.__name__] = m

    spec = importlib.util.spec_from_file_location(
        'sverchok.nodes.sdf_numba', os.path.join(NODE_DIR, '__init__.py'),
        submodule_search_locations=[NODE_DIR])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules['sverchok.nodes.sdf_numba'] = pkg
    spec.loader.exec_module(pkg)
    settle = importlib.import_module('sverchok.utils.sdfnumba_helper')
    settle.time = types.SimpleNamespace(monotonic=lambda: clock[0])
    return bpy


@pytest.fixture
def env():
    saved = {k: sys.modules.get(k) for k in list(sys.modules)
             if k == 'bpy' or k.startswith(('bpy.', 'sverchok'))}
    timers, clock = Timers(), [0.0]
    bpy = _install_stubs(timers, clock)
    yield types.SimpleNamespace(bpy=bpy, timers=timers, clock=clock)
    for k in [k for k in sys.modules if k == 'bpy' or k.startswith(('bpy.', 'sverchok'))]:
        del sys.modules[k]
    sys.modules.update({k: v for k, v in saved.items() if v is not None})


def make_node(env, module, cls, name, **values):
    mod = importlib.import_module(f'sverchok.nodes.sdf_numba.{module}')
    node_cls = getattr(mod, cls)
    node = node_cls.__new__(node_cls)
    node.inputs, node.outputs = Sockets(), Sockets()
    node.sv_init(None)
    node.name, node.node_id = name, name
    tree = types.SimpleNamespace(name='tree', nodes={})
    tree.nodes[name] = node
    node.id_data = tree
    env.bpy.data.node_groups['tree'] = tree
    for k, v in type(node).__annotations__.items():
        setattr(node, k, v)                    # the property defaults
    for k, v in values.items():
        setattr(node, k, v)
    node.runs = 0

    def process_node(context):
        node.runs += 1
        node.process()
    node.process_node = process_node
    for s in node.outputs:
        s.is_linked = True
    return node


def csg(radius):
    f = sn.sphere(radius) & sn.box(1.5 * radius)
    c = sn.cylinder(0.5)
    return f - (c.orient(sn.X) | c.orient(sn.Y) | c)


def feed(node, shape):
    node.inputs['SDF'].sv_set([shape])


# --- tests ------------------------------------------------------------------------

def test_mesh_node_settles_to_full_resolution(env):
    node = make_node(env, 'sdf_mesh', 'SvSDFMeshNode', 'mesh', resolution=64)
    # a drag: several changes in quick succession
    for i, r in enumerate((1.0, 1.02, 1.04, 1.06)):
        env.clock[0] = 0.05 * i
        feed(node, csg(r))
        node.process()
        coarse = len(node.outputs['Vertices'].data[0])
    assert len(env.timers.pending) == 1          # one timer, pushed back each time
    env.clock[0] = 10.0                          # ... and then nothing for a while
    env.timers.run_all()
    assert node.runs == 1                        # the node re-ran itself
    full = len(node.outputs['Vertices'].data[0])
    assert full > 4 * coarse
    # the settled result is the one for the last value dragged to
    verts = np.asarray(node.outputs['Vertices'].data[0])
    assert np.abs(csg(1.06)(verts)).max() < 0.02
    # processing again with nothing changed reuses it without rebuilding
    node.process()
    assert len(env.timers.pending) == 0
    assert len(node.outputs['Vertices'].data[0]) == full


def test_mesh_node_settle_zero_builds_immediately(env):
    node = make_node(env, 'sdf_mesh', 'SvSDFMeshNode', 'mesh', resolution=48, settle=0.0)
    feed(node, csg(1.0))
    node.process()
    assert not env.timers.pending
    polys = node.outputs['Polygons'].data[0]
    edges = node.outputs['Edges'].data[0]
    assert len(polys) and len(edges) == 2 * len(polys)       # closed quad mesh: E = 2F


def test_mesh_node_triangles(env):
    node = make_node(env, 'sdf_mesh', 'SvSDFMeshNode', 'mesh', resolution=40,
                     settle=0.0, triangles=True)
    feed(node, sn.box(1.0))
    node.process()
    polys = node.outputs['Polygons'].data[0]
    assert all(len(p) == 3 for p in polys)


def test_remesh_node_passes_through_then_remeshes(env):
    shape = sn.box(1.5)
    verts, faces = shape.generate(samples=2 ** 15, sharp=True)
    node = make_node(env, 'sdf_remesh', 'SvSDFRemeshNode', 'remesh')
    feed(node, shape)
    node.inputs['Vertices'].sv_set([verts])
    node.inputs['Polygons'].sv_set([faces.tolist()])
    node.process()
    # still settling: the input comes straight through
    assert len(node.outputs['Vertices'].data[0]) == len(verts)
    env.clock[0] = 10.0
    env.timers.run_all()
    out_v = np.asarray(node.outputs['Vertices'].data[0])
    out_p = node.outputs['Polygons'].data[0]
    assert len(out_v) != len(verts) and all(len(p) == 3 for p in out_p)
    assert np.abs(shape(out_v)).max() < 1e-6


def test_remesh_node_without_mesh_inputs_makes_its_own(env):
    shape = sn.box(1.5)
    node = make_node(env, 'sdf_remesh', 'SvSDFRemeshNode', 'remesh_own')
    feed(node, shape)                              # only the shape is connected
    node.process()
    coarse = np.asarray(node.outputs['Vertices'].data[0])
    assert len(coarse) > 0                         # a coarse stand-in while settling
    env.clock[0] = 10.0
    env.timers.run_all()
    out_v = np.asarray(node.outputs['Vertices'].data[0])
    out_p = node.outputs['Polygons'].data[0]
    assert all(len(p) == 3 for p in out_p)
    assert np.abs(shape(out_v)).max() < 1e-6
    # edges and corners kept: the box's corners are vertices of the result
    corner = np.array([0.75, 0.75, 0.75])
    assert np.linalg.norm(out_v - corner, axis=1).min() < 1e-6


# --- building in the background ------------------------------------------------

import threading


def test_mesh_node_background_build(env):
    node = make_node(env, 'sdf_mesh', 'SvSDFMeshNode', 'mesh_bg', resolution=64,
                     background=True)
    feed(node, csg(1.0))
    node.process()
    coarse = len(node.outputs['Vertices'].data[0])
    env.clock[0] = 10.0
    env.timers.run_all()                 # starts the thread, then polls it to the end
    assert node.runs == 1
    assert len(node.outputs['Vertices'].data[0]) > 4 * coarse


class FakeNode:
    def __init__(self, env, name):
        self.name = self.node_id = name
        self.id_data = types.SimpleNamespace(name='tree', nodes={name: self})
        env.bpy.data.node_groups['tree'] = self.id_data
        self.delivered = []

    def process_node(self, context):
        self.delivered.append(self.last)


def test_background_result_made_stale_is_dropped(env):
    helper = importlib.import_module('sverchok.utils.sdfnumba_helper')
    node = FakeNode(env, 'fake')
    gate = threading.Event()

    def slow_full(value):
        def full():
            gate.wait(5)                 # held until the test lets it finish
            return value
        return full

    def ask(value):
        node.last = helper.request(node, value, lambda: 'coarse', slow_full(value),
                                   0.5, background=True)
        return node.last

    assert ask('A') == 'coarse'
    tick = env.timers.pending[0]
    env.clock[0] = 1.0
    assert tick() == helper.POLL           # quiet long enough: build 'A' starts
    env.clock[0] = 1.1
    assert ask('B') == 'coarse'          # the drag resumes while 'A' builds
    gate.set()
    helper._state['fake']['job'].thread.join()
    wait = tick()                        # 'A' is finished but stale: dropped
    assert node.delivered == [] and 0 < wait <= 0.5
    env.clock[0] = 2.0
    assert tick() == helper.POLL           # quiet again: build 'B'
    helper._state['fake']['job'].thread.join()
    assert tick() is None
    assert ask('B') == 'B'               # only 'B' was ever delivered
    assert helper._state['fake']['result_sig'] == 'B'


def test_background_falls_back_when_threads_unsafe(env, monkeypatch):
    helper = importlib.import_module('sverchok.utils.sdfnumba_helper')
    monkeypatch.setattr(helper, 'background_safe', lambda: False)
    node = FakeNode(env, 'fake2')
    node.last = helper.request(node, 'A', lambda: 'coarse', lambda: 'A', 0.5,
                               background=True)
    env.clock[0] = 1.0
    assert env.timers.pending[0]() is None   # built right away on this thread
    assert helper._state['fake2']['job'] is None
    assert helper._state['fake2']['result_sig'] == 'A'



def test_remesh_node_warns_about_artifacts(env):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_sdfnumba import lamp_shade
    node = make_node(env, 'sdf_remesh', 'SvSDFRemeshNode', 'remesh_warn', settle=0.0,
                     start_resolution=32)      # far too coarse for the 0.04 wall
    feed(node, lamp_shade())
    node.process()
    assert any('too coarse' in w for w in node.warnings)


def test_remesh_node_quiet_when_clean(env):
    node = make_node(env, 'sdf_remesh', 'SvSDFRemeshNode', 'remesh_ok', settle=0.0)
    feed(node, sn.box(1.5))
    node.process()
    assert not getattr(node, 'warnings', [])
