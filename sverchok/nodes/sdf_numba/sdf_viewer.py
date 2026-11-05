# This file is part of sdfnumba (https://github.com/zeffii/sdfnumba), MIT licence.
# Copy this folder into sverchok/nodes/ to use it; see sverchok/README.md.

"""SDF Viewer: draws an sdfnumba shape in the 3D view by raymarching it on
the GPU.  No mesh is made, so edges stay exact and parameter changes are
instant.  Use the SDF Mesh node when you want actual geometry.

Follows the conventions of Viewer Draw Mk5:
- shaders use GPUShaderCreateInfo (OpenGL, Metal and Vulkan), compiled once
  per shape structure and cached at module level
- nothing touching the GPU happens in process(): shaders, batches and the
  parameter texture are made on the first redraw, which has a GPU context
- colours go through to_framebuffer() for Blender's sRGB overlay target
- GPU state is reset after drawing

Unlike Mk5's lines, the raymarcher must write gl_FragDepth (the depth comes
from where each ray hits the shape), so early-z is off for these fragments.

How it fits together:
  process()   prepares the shader sources, the box (or slice) geometry and
              the texture data: shape parameters plus viewer settings
  draw_sdf()  every redraw: builds what's missing, passes the current view,
              and draws the box; the fragment shader marches the rays
"""

from types import SimpleNamespace

import bpy
import gpu
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from sverchok.data_structure import node_id, updateNode
from sverchok.node_tree import SverchCustomTreeNode
from sverchok.ui.bgl_callback_3dview import callback_disable, callback_enable

_SHADERS = {}   # (program key, 'shape' | 'slice') -> GPUShader, shared by all viewers


def _sdfnumba():
    """Import sdfnumba on first use (the very first import compiles for a while)."""
    try:
        import sdfnumba
        from sdfnumba import glsl
    except ImportError as e:
        raise ImportError('sdfnumba is not installed in Blender\'s Python '
                          '(see sverchok/README.md in the sdfnumba repo)') from e
    return sdfnumba, glsl


def _build_shader(vertex, fragment):
    iface = gpu.types.GPUStageInterfaceInfo('sv_sdf_numba_iface')
    iface.smooth('VEC3', 'world_pos')

    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('BOOL', 'srgbTarget')
    info.push_constant('MAT4', 'ViewProjection')
    info.push_constant('VEC4', 'cam')
    info.push_constant('VEC4', 'view_dir')
    info.sampler(0, 'FLOAT_2D', 'prm_tex')
    info.vertex_in(0, 'VEC3', 'pos')
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', 'FragColor')
    if hasattr(info, 'depth_write'):      # Blender 4.2+; needed on Metal and Vulkan
        info.depth_write('ANY')
    info.vertex_source(vertex)
    info.fragment_source(fragment)
    return gpu.shader.create_from_info(info)


def get_shader(key, vertex, fragment):
    """compile once per shape structure; needs a GPU context, so only call
    from the draw callback"""
    shader = _SHADERS.get(key)
    if shader is None:
        shader = _SHADERS[key] = _build_shader(vertex, fragment)
    return shader


def view_info(rd, region):
    """(camera position, perspective flag), view direction, world size of a
    pixel (at distance 1 in perspective views)"""
    inv = rd.view_matrix.inverted()
    persp = abs(rd.window_matrix[3][3]) < 0.5
    cam = inv.translation
    view_dir = (-inv.col[2].xyz).normalized()
    pixel = 2.0 / (rd.window_matrix[1][1] * max(region.height, 1))
    return (cam.x, cam.y, cam.z, 1.0 if persp else 0.0), (*view_dir, pixel)


def _uniform(shader, name, value):
    # a uniform the shader doesn't use is compiled away; setting it would fail
    try:
        shader.uniform_float(name, value)
    except ValueError:
        pass


def draw_sdf(context, state):
    rd = context.region_data
    region = context.region
    if rd is None or region is None:
        return

    if state.texture is None:
        data = state.texture_data
        buf = gpu.types.Buffer('FLOAT', data.size, data.ravel().tolist())
        state.texture = gpu.types.GPUTexture((data.shape[1], 2), format='RGBA32F', data=buf)

    cam, view_dir = view_info(rd, region)
    vp = rd.perspective_matrix

    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    gpu.state.face_culling_set('NONE')
    try:
        for part in state.parts:
            if part.failed:
                continue
            try:
                shader = get_shader(part.key, state.vertex, part.fragment)
            except Exception as err:
                # report once, not on every redraw
                print(f"SDF Viewer: shader failed to compile: {err}")
                part.failed = True
                continue
            if part.batch is None:
                part.batch = batch_for_shader(shader, 'TRIS', {'pos': part.tris})
            gpu.state.blend_set(part.blend)
            shader.bind()
            _uniform(shader, 'ViewProjection', vp)
            _uniform(shader, 'cam', cam)
            _uniform(shader, 'view_dir', view_dir)
            shader.uniform_sampler('prm_tex', state.texture)
            part.batch.draw(shader)
    finally:
        gpu.state.face_culling_set('NONE')
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')


class SvSDFViewerNode(SverchCustomTreeNode, bpy.types.Node):
    """
    Triggers: sdf raymarch preview viewer
    Tooltip: Draw an SDF shape in the 3D view on the GPU, without meshing it
    """
    bl_idname = 'SvSDFViewerNode'
    bl_label = 'SDF Viewer'
    bl_icon = 'SHADING_RENDERED'

    activate: BoolProperty(name='Show', default=True, update=updateNode)
    show_shape: BoolProperty(name='Shape', default=True, update=updateNode)

    color_by_primitive: BoolProperty(
        name='Colour by primitive', default=False, update=updateNode,
        description='Tint each part by the primitive it comes from')
    base_color: FloatVectorProperty(
        name='Colour', subtype='COLOR', size=3, min=0, max=1,
        default=(0.8, 0.8, 0.8), update=updateNode)
    line_color: FloatVectorProperty(
        name='Lines', subtype='COLOR', size=3, min=0, max=1,
        default=(0.05, 0.05, 0.05), update=updateNode)

    show_seams: BoolProperty(
        name='Seams', default=True, update=updateNode,
        description='Lines where one primitive meets another')
    show_creases: BoolProperty(
        name='Creases', default=True, update=updateNode,
        description='Lines where the surface turns sharply')
    crease_threshold: FloatProperty(
        name='Crease', default=0.35, min=0.01, max=2.0, update=updateNode,
        description='How sharply the surface must turn (per pixel) to draw a crease line')
    show_outline: BoolProperty(name='Outline', default=True, update=updateNode)
    line_width: FloatProperty(name='Outline width', default=1.5, min=0.5, max=8,
                              update=updateNode)

    show_slice: BoolProperty(
        name='Slice', default=False, update=updateNode,
        description='A plane through the shape showing distance bands')
    slice_axis: EnumProperty(
        name='Axis', items=[('0', 'X', ''), ('1', 'Y', ''), ('2', 'Z', '')],
        default='2', update=updateNode)
    slice_offset: FloatProperty(name='Offset', default=0.0, update=updateNode)
    slice_spacing: FloatProperty(name='Spacing', default=0.1, min=0.001,
                                 update=updateNode)

    max_steps: IntProperty(name='Steps', default=160, min=8, max=1024,
                           update=updateNode,
                           description='Most raymarching steps per pixel')
    step_scale: FloatProperty(
        name='Step scale', default=1.0, min=0.05, max=1.0, update=updateNode,
        description='Lower this if twisted or bent shapes show holes')
    padding: FloatProperty(name='Padding', default=0.05, min=0.0, update=updateNode,
                           description='Margin added around the shape\'s bounds')

    def sv_init(self, context):
        self.inputs.new('SvStringsSocket', 'SDF')

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, 'activate', text='', icon='HIDE_' + ('OFF' if self.activate else 'ON'))
        row.prop(self, 'show_shape', toggle=True)
        row.prop(self, 'show_slice', toggle=True)
        col = layout.column(align=True)
        if self.show_shape:
            col.prop(self, 'color_by_primitive')
            if not self.color_by_primitive:
                col.prop(self, 'base_color', text='')
            row = col.row(align=True)
            row.prop(self, 'show_seams', toggle=True)
            row.prop(self, 'show_creases', toggle=True)
            row.prop(self, 'show_outline', toggle=True)
            col.prop(self, 'line_color', text='')
        if self.show_slice:
            col.separator()
            col.row(align=True).prop(self, 'slice_axis', expand=True)
            col.prop(self, 'slice_offset')
            col.prop(self, 'slice_spacing')

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)
        col = layout.column(align=True)
        col.prop(self, 'crease_threshold')
        col.prop(self, 'line_width')
        col.prop(self, 'max_steps')
        col.prop(self, 'step_scale')
        col.prop(self, 'padding')

    def _shape(self):
        """The first sdfnumba shape found on the input, however nested."""
        stack = [self.inputs['SDF'].sv_get(deepcopy=False, default=[])]
        while stack:
            item = stack.pop(0)
            if hasattr(item, 'program'):
                return item
            if isinstance(item, (list, tuple)):
                stack.extend(item)
        return None

    def process(self):
        if bpy.app.background:
            return
        n_id = node_id(self)
        callback_disable(n_id)
        if not (self.id_data.sv_show and self.activate):
            return
        if not self.inputs['SDF'].is_linked:
            return
        shape = self._shape()
        if shape is None:
            return
        sn, glsl = _sdfnumba()

        program = shape.program()
        lo, hi = sn.estimate_bounds(program)
        lo, hi = Vector(lo), Vector(hi)
        pad = (hi - lo).length * self.padding
        lo -= Vector((pad, pad, pad))
        hi += Vector((pad, pad, pad))

        settings = glsl.settings_array(
            box_lo=tuple(lo), box_hi=tuple(hi),
            base_color=tuple(self.base_color), line_color=tuple(self.line_color),
            color_by_primitive=float(self.color_by_primitive),
            show_seams=float(self.show_seams), show_creases=float(self.show_creases),
            crease_threshold=self.crease_threshold,
            show_outline=float(self.show_outline), line_width=self.line_width,
            max_steps=float(self.max_steps), step_scale=self.step_scale,
            slice_axis=float(self.slice_axis), slice_offset=self.slice_offset,
            slice_spacing=self.slice_spacing)

        parts = []
        if self.show_shape:
            parts.append(SimpleNamespace(
                key=(program.key, 'shape'), fragment=glsl.fragment_source(program),
                tris=glsl.box_triangles(lo, hi), blend='NONE', batch=None, failed=False))
        if self.show_slice:
            parts.append(SimpleNamespace(
                key=(program.key, 'slice'), fragment=glsl.slice_fragment_source(program),
                tris=glsl.slice_quad(lo, hi, int(self.slice_axis), self.slice_offset),
                blend='ALPHA', batch=None, failed=False))
        if not parts:
            return

        state = SimpleNamespace(
            vertex=glsl.VERTEX, parts=parts, texture=None,
            texture_data=glsl.parameter_texture_data(program.params, settings))
        callback_enable(n_id, {'tree_name': self.id_data.name[:],
                               'custom_function': draw_sdf, 'args': state})

    def sv_free(self):
        callback_disable(node_id(self))

    def show_viewport(self, is_show: bool):
        if not self.activate:
            return
        if is_show:
            self.process()
        else:
            callback_disable(node_id(self))


classes = [SvSDFViewerNode]
register, unregister = bpy.utils.register_classes_factory(classes)
