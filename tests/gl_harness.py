"""Run the viewer's GLSL offscreen (moderngl; a software renderer is fine).

Declares the same interface the Blender node provides (see glsl.py), so the
code under test is exactly what the viewer runs.
"""

import numpy as np

from sdfnumba import glsl

try:
    import moderngl
except ImportError:
    moderngl = None

HEADER = '#version 330\n'

INTERFACE_FRAG = '''
uniform sampler2D prm_tex;
uniform mat4 ViewProjection;
uniform vec4 cam;
uniform vec4 view_dir;
uniform bool srgbTarget;
in vec3 world_pos;
out vec4 FragColor;
'''

INTERFACE_VERT = '''
uniform mat4 ViewProjection;
in vec3 pos;
out vec3 world_pos;
'''


def context():
    if moderngl is None:
        return None
    try:
        return moderngl.create_standalone_context(backend='egl')
    except Exception:
        try:
            return moderngl.create_standalone_context()
        except Exception:
            return None


def _param_texture(ctx, program, settings):
    data = glsl.parameter_texture_data(program.params, settings)
    tex = ctx.texture((data.shape[1], 2), 4, data.tobytes(), dtype='f4')
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return tex


def evaluate(ctx, program, points):
    """sdf(p) and sdf_id(p) on the GPU for each point."""
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    w = 256
    h = (n + w - 1) // w
    pts = np.zeros((h * w, 4), dtype=np.float32)
    pts[:n, :3] = points
    pts_tex = ctx.texture((w, h), 4, pts.tobytes(), dtype='f4')
    pts_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    frag = (HEADER + 'uniform sampler2D prm_tex;\nuniform sampler2D pts_tex;\nout vec4 FragColor;\n'
            + glsl.PRM_FROM_TEXTURE + glsl.shape_source(program) + '''
void main() {
    vec3 p = texelFetch(pts_tex, ivec2(gl_FragCoord.xy), 0).xyz;
    FragColor = vec4(sdf(p), float(sdf_id(p)), 0.0, 1.0);
}''')
    vert = HEADER + 'in vec2 v; void main() { gl_Position = vec4(v, 0.0, 1.0); }'
    prog = ctx.program(vertex_shader=vert, fragment_shader=frag)
    quad = np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype='f4')
    vao = ctx.vertex_array(prog, [(ctx.buffer(quad.tobytes()), '2f', 'v')])
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4, dtype='f4')])
    fbo.use()
    _param_texture(ctx, program, glsl.settings_array()).use(0)
    prog['prm_tex'] = 0
    pts_tex.use(1)
    prog['pts_tex'] = 1
    vao.render()
    out = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype='f4').reshape(-1, 4)[:n]
    return out[:, 0], out[:, 1].astype(np.int64)


def look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_y, aspect, near, far):
    f = 1 / np.tan(fov_y / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def render(ctx, program, bounds, size=(320, 240), eye=(2.6, -3.2, 2.2), slice_=False,
           **settings):
    """Raymarch the shape like the viewer does.  Returns (rgba, depth, vp)."""
    lo, hi = (np.asarray(b, float) for b in bounds)
    w, h = size
    view = look_at(eye, (lo + hi) / 2)
    proj = perspective(np.radians(40), w / h, 0.05, 50)
    vp = proj @ view
    settings = glsl.settings_array(box_lo=lo, box_hi=hi, **settings)
    frag_src = (glsl.slice_fragment_source(program) if slice_
                else glsl.fragment_source(program))
    prog = ctx.program(vertex_shader=HEADER + INTERFACE_VERT + glsl.VERTEX,
                       fragment_shader=HEADER + INTERFACE_FRAG + frag_src)
    geom = (glsl.slice_quad(lo, hi, int(settings[glsl.LAYOUT['slice_axis'][0]]),
                            settings[glsl.LAYOUT['slice_offset'][0]])
            if slice_ else glsl.box_triangles(lo, hi))
    vao = ctx.vertex_array(prog, [(ctx.buffer(geom.tobytes()), '3f', 'pos')])
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((w, h), 4, dtype='f4')],
                          depth_attachment=ctx.depth_texture((w, h)))
    fbo.use()
    fbo.clear(0.92, 0.92, 0.9, 1.0, depth=1.0)
    ctx.enable(moderngl.DEPTH_TEST)
    _param_texture(ctx, program, settings).use(0)
    prog['prm_tex'] = 0
    prog['ViewProjection'].write(vp.T.astype('f4').tobytes())
    if not slice_:
        prog['cam'].value = (*eye, 1.0)
        prog['view_dir'].value = (0.0, 0.0, 0.0, 2.0 / (proj[1, 1] * h))
    vao.render()
    rgba = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype='f4').reshape(h, w, 4)
    depth = np.frombuffer(fbo.read(attachment=-1, components=1, dtype='f4'), dtype='f4').reshape(h, w)
    return rgba[::-1], depth[::-1], vp


def unproject(depth, vp):
    """World positions of the pixels in a depth image (top row first)."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    ndc = np.stack([(xs + 0.5) / w * 2 - 1, 1 - (ys + 0.5) / h * 2, depth * 2 - 1,
                    np.ones_like(depth)], -1).reshape(-1, 4)
    world = ndc @ np.linalg.inv(vp).T
    return (world[:, :3] / world[:, 3:]).reshape(h, w, 3)
