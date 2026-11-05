# This file is part of sdfnumba (https://github.com/zeffii/sdfnumba), MIT licence.

"""SDF Mesh: turns an sdfnumba shape into a closed mesh.

Surface Nets for smooth shapes, or dual contouring (Sharp) to keep edges and
corners.  While the shape keeps changing (a slider being dragged) the node
outputs a quick coarse mesh; once it has been still for `Settle` seconds it
builds the full one (see _settle.py).  The SDF Viewer is the live view; this
node is for when you want the actual geometry.
"""

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty

from sverchok.data_structure import updateNode
from sverchok.node_tree import SverchCustomTreeNode

from . import _settle
from ._common import first_shape, import_sdfnumba, mesh_outputs, shape_signature

PREVIEW_RESOLUTION = 24


def _mesh(sn, shape, resolution, sharp, triangles):
    program = shape.program()
    lo, hi = sn.estimate_bounds(program)
    step = max(h - l for l, h in zip(lo, hi)) / resolution
    verts, faces = sn.generate(program, step=step, bounds=(lo, hi), sharp=sharp,
                               triangles=triangles)
    return mesh_outputs(verts, faces)


class SvSDFMeshNode(SverchCustomTreeNode, bpy.types.Node):
    """
    Triggers: sdf mesh generate polygonize
    Tooltip: Make a closed mesh from an SDF shape (Surface Nets, or dual contouring for sharp edges)
    """
    bl_idname = 'SvSDFMeshNode'
    bl_label = 'SDF Mesh'
    bl_icon = 'MESH_ICOSPHERE'

    resolution: IntProperty(
        name='Resolution', default=96, min=8, max=1024, update=updateNode,
        description='Grid cells along the longest side of the shape')
    sharp: BoolProperty(
        name='Sharp', default=True, update=updateNode,
        description='Dual contouring: keep sharp edges and corners (slower)')
    triangles: BoolProperty(
        name='Triangles', default=False, update=updateNode,
        description='Output triangles instead of quads')
    settle: FloatProperty(
        name='Settle', default=0.3, min=0.0, max=5.0, subtype='TIME', unit='TIME',
        update=updateNode,
        description='Seconds without changes before the full mesh is built; '
                    '0 builds it on every change')
    preview: BoolProperty(
        name='Preview while changing', default=True, update=updateNode,
        description='Output a coarse mesh while the shape keeps changing '
                    '(otherwise keep the last full mesh)')

    def sv_init(self, context):
        self.inputs.new('SvStringsSocket', 'SDF')
        self.outputs.new('SvVerticesSocket', 'Vertices')
        self.outputs.new('SvStringsSocket', 'Edges')
        self.outputs.new('SvStringsSocket', 'Polygons')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'resolution')
        row = layout.row(align=True)
        row.prop(self, 'sharp', toggle=True)
        row.prop(self, 'triangles', toggle=True)

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)
        layout.prop(self, 'settle')
        layout.prop(self, 'preview')

    def process(self):
        if not self.inputs['SDF'].is_linked or not any(s.is_linked for s in self.outputs):
            return
        shape = first_shape(self.inputs['SDF'])
        if shape is None:
            return
        sn = import_sdfnumba()
        res, sharp, tris = self.resolution, self.sharp, self.triangles

        signature = (shape_signature(shape), res, sharp, tris)
        quick = None
        if self.preview and res > PREVIEW_RESOLUTION:
            def quick():
                return _mesh(sn, shape, PREVIEW_RESOLUTION, False, tris)

        def full():
            return _mesh(sn, shape, res, sharp, tris)

        result = _settle.request(self, signature, quick, full, self.settle)
        if result is None:
            return
        verts, edges, polys = result
        self.outputs['Vertices'].sv_set(verts)
        self.outputs['Edges'].sv_set(edges)
        self.outputs['Polygons'].sv_set(polys)

    def sv_free(self):
        _settle.forget(self)


classes = [SvSDFMeshNode]
register, unregister = bpy.utils.register_classes_factory(classes)
