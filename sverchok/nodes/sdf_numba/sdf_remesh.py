# This file is part of sdfnumba (https://github.com/zeffii/sdfnumba), MIT licence.

"""SDF Remesh: adaptive isotropic remeshing of a mesh of an sdfnumba shape,
projected onto the exact surface.  Triangles follow the curvature, get finer
along creases (optional) and grow gradually away from them.

Needs only the shape.  The Vertices and Polygons inputs are optional: a
closed mesh of the same shape to start from (normally from SDF Mesh), which
gives more control.  Without them the node makes its own starting mesh
(Start resolution, sharp edges).

While the inputs keep changing it outputs the starting mesh untouched (or,
without mesh inputs, a coarse one); once they have been still for `Settle`
seconds it remeshes (see sdfnumba_helper.py).
"""

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty

from sverchok.data_structure import updateNode
from sverchok.node_tree import SverchCustomTreeNode

import numpy as np

from sverchok.utils.sdfnumba_helper import faces_array, first_shape, forget, mesh_outputs, request, shape_signature

try:
    import sdfnumba as sn      # cheap: sdfnumba loads its parts on first use
except ImportError:
    sn = None                  # Sverchok shows the node as needing sdfnumba


class SvSDFRemeshNode(SverchCustomTreeNode, bpy.types.Node):
    """
    Triggers: sdf remesh adaptive isotropic
    Tooltip: Remesh a mesh of an SDF shape: curvature-adaptive triangles that keep creases
    """
    bl_idname = 'SvSDFRemeshNode'
    bl_label = 'SDF Remesh'
    bl_icon = 'MOD_REMESH'
    sv_dependencies = {'sdfnumba'}

    max_len: FloatProperty(
        name='Edge length', default=0.0, min=0.0, update=updateNode,
        description='Longest edge, on flat areas (0: bounding box diagonal / 40)')
    detail: FloatProperty(
        name='Detail', default=10.0, min=1.0, max=100.0, update=updateNode,
        description='How much shorter edges may get on tight curves (max / min length)')
    creases: BoolProperty(
        name='Creases', default=True, update=updateNode,
        description='Find creases and keep them, with finer triangles along them')
    crease_angle: FloatProperty(
        name='Crease angle', default=30.0, min=1.0, max=179.0, update=updateNode,
        description='Where the surface turns more than this (degrees), it is a crease')
    crease_len: FloatProperty(
        name='Crease length', default=0.0, min=0.0, update=updateNode,
        description='Edge length along creases (0: edge length / 4)')
    gradation: FloatProperty(
        name='Gradation', default=0.4, min=0.05, max=2.0, update=updateNode,
        description='How quickly triangles may grow away from creases and tight curves')
    iterations: IntProperty(name='Iterations', default=6, min=1, max=30, update=updateNode)
    start_resolution: IntProperty(
        name='Start resolution', default=64, min=8, max=512, update=updateNode,
        description='Without mesh inputs: grid cells along the longest side of the '
                    'starting mesh the node makes itself')
    settle: FloatProperty(
        name='Settle', default=1.0, min=0.0, max=5.0, subtype='TIME', unit='TIME',
        update=updateNode,
        description='Seconds without changes before remeshing; 0 remeshes on every change')

    def sv_init(self, context):
        self.inputs.new('SvStringsSocket', 'SDF')
        self.inputs.new('SvVerticesSocket', 'Vertices')
        self.inputs.new('SvStringsSocket', 'Polygons')
        self.outputs.new('SvVerticesSocket', 'Vertices')
        self.outputs.new('SvStringsSocket', 'Edges')
        self.outputs.new('SvStringsSocket', 'Polygons')

    def _has_mesh_input(self):
        return self.inputs['Vertices'].is_linked and self.inputs['Polygons'].is_linked

    def draw_buttons(self, context, layout):
        col = layout.column(align=True)
        if not self._has_mesh_input():
            col.prop(self, 'start_resolution')
        col.prop(self, 'max_len')
        col.prop(self, 'detail')
        col.prop(self, 'creases', toggle=True)

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)
        col = layout.column(align=True)
        if self.creases:
            col.prop(self, 'crease_angle')
            col.prop(self, 'crease_len')
        col.prop(self, 'gradation')
        col.prop(self, 'iterations')
        layout.prop(self, 'settle')

    def _start_mesh(self, shape, resolution, sharp):
        """A closed mesh of the shape to start from, when none is connected."""
        program = shape.program()
        lo, hi = sn.estimate_bounds(program)
        step = max(h - l for l, h in zip(lo, hi)) / resolution
        return sn.generate(program, step=step, bounds=(lo, hi), sharp=sharp,
                           triangles=True)

    def process(self):
        inputs = self.inputs
        if not inputs['SDF'].is_linked or not any(s.is_linked for s in self.outputs):
            return
        shape = first_shape(inputs['SDF'])
        if shape is None:
            return

        if self._has_mesh_input():
            verts = inputs['Vertices'].sv_get(deepcopy=False, default=[[]])[0]
            polys = inputs['Polygons'].sv_get(deepcopy=False, default=[[]])[0]
            if not len(verts) or not len(polys):
                return
            V = np.array(verts, dtype=np.float64).reshape(-1, 3)
            F = faces_array(polys)
            mesh_key = (V.tobytes(), F.tobytes())

            def start():
                return V, F

            def quick():
                return mesh_outputs(V, F)       # the input, untouched
        else:
            res = self.start_resolution
            mesh_key = ('own', res)

            def start():
                return self._start_mesh(shape, res, True)

            def quick():
                return mesh_outputs(*self._start_mesh(shape, min(res, 24), False))

        options = dict(max_len=self.max_len or None,
                       creases=self.creases, crease_angle=self.crease_angle,
                       crease_len=self.crease_len or None,
                       gradation=self.gradation, iterations=self.iterations)
        detail = self.detail
        signature = (shape_signature(shape), mesh_key,
                     tuple(sorted(options.items())), detail)

        def full():
            V0, F0 = start()
            max_len = options['max_len'] or np.linalg.norm(V0.max(0) - V0.min(0)) / 40
            out_v, out_f = sn.remesh(shape, V0, F0, **dict(options, max_len=max_len,
                                                           min_len=max_len / detail))
            return mesh_outputs(out_v, out_f)

        result = request(self, signature, quick, full, self.settle)
        if result is None:
            return
        out_v, out_e, out_p = result
        self.outputs['Vertices'].sv_set(out_v)
        self.outputs['Edges'].sv_set(out_e)
        self.outputs['Polygons'].sv_set(out_p)

    def sv_free(self):
        forget(self)


classes = [SvSDFRemeshNode]
register, unregister = bpy.utils.register_classes_factory(classes)
