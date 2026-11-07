# This file is part of sdfnumba (https://github.com/zeffii/sdfnumba), MIT licence.

"""SDF Revolve: spins a profile drawn as vertices around a world axis.

The profile is closed loops of vertices in one plane, given as
    Vertices                one loop, the vertices in order
    Vertices + Edges        any order; edges are chained into closed loops
    Vertices + Polygons     the polygons' outline (shared edges are dropped)
Several loops make holes (a loop inside another) or islands; every object on
the sockets adds its loops to the same profile.

Vertices off the profile's plane are moved onto it; the node warns when that
moved something by more than float32 rounding.  The plane should contain the
axis (XZ or YZ for Z); a tilted plane is revolved around the axis's
projection onto it.
"""

import bpy
from bpy.props import EnumProperty

from sverchok.data_structure import updateNode
from sverchok.node_tree import SverchCustomTreeNode

try:
    import sdfnumba as sn
except ImportError:
    sn = None


class SvSDFRevolveNode(SverchCustomTreeNode, bpy.types.Node):
    """
    Triggers: sdf revolve lathe spin profile
    Tooltip: Revolve a profile (vertices, edges or polygons) around an axis into an SDF shape
    """
    bl_idname = 'SvSDFRevolveNode'
    bl_label = 'SDF Revolve'
    bl_icon = 'MOD_SCREW'
    sv_dependencies = {'sdfnumba'}

    axis: EnumProperty(
        name='Axis', default='Z', update=updateNode,
        items=[(a, a, f'Revolve around the world {a} axis') for a in 'XYZ'])

    def sv_init(self, context):
        self.inputs.new('SvVerticesSocket', 'Vertices')
        self.inputs.new('SvStringsSocket', 'Edges')
        self.inputs.new('SvStringsSocket', 'Polygons')
        self.outputs.new('SvStringsSocket', 'SDF')

    def draw_buttons(self, context, layout):
        layout.row().prop(self, 'axis', expand=True)

    def process(self):
        if not self.inputs['Vertices'].is_linked or not self.outputs['SDF'].is_linked:
            return
        verts = self.inputs['Vertices'].sv_get(deepcopy=False)
        polys = self.inputs['Polygons'].sv_get(deepcopy=False, default=[])
        edges = self.inputs['Edges'].sv_get(deepcopy=False, default=[])
        use = 'polygons' if polys else 'edges' if edges else 'points'
        loops = []
        for i, v in enumerate(verts):
            if not len(v):
                continue
            if use == 'polygons':
                loops += sn.profile_loops(v, polygons=polys[min(i, len(polys) - 1)])
            elif use == 'edges':
                loops += sn.profile_loops(v, edges=edges[min(i, len(edges) - 1)])
            else:
                loops += sn.profile_loops(v)
        if not loops:
            return
        shape, info = sn.revolve_profile(loops, self.axis)
        if info.moved:
            self.warning(f'{info.moved} vertices were off the profile plane (up to '
                         f'{info.max_offset:.3g}) and were moved onto it')
        self.outputs['SDF'].sv_set([shape])


classes = [SvSDFRevolveNode]
register, unregister = bpy.utils.register_classes_factory(classes)
