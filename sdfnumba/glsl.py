"""GLSL for the GPU viewer: the shape as shader code, and the raymarcher.

shape_source(program) writes the program out as two straight-line GLSL
functions, exactly like codegen.py does for numba:

    float sdf(vec3 p)       distance
    int   sdf_id(vec3 p)    which primitive won (row in program.code)

Parameter values are read through PRM(i), never baked in, so a slider
change only re-uploads the parameter texture; the shader is rebuilt only
when the shape's structure changes.

The raymarching shader (VERTEX, FRAGMENT) draws the shape by sphere tracing
from the camera, inside a box around it.  It is written against a small
interface so the same code runs in Blender (sverchok/nodes/sdf_numba) and in
the offscreen tests:

    push constants   mat4 ViewProjection, vec4 cam (xyz, w=1 if perspective),
                     vec4 view_dir (xyz, w=world size of a pixel at distance 1,
                     or of any pixel in an orthographic view), bool srgbTarget
                     (Blender sets it; its overlay framebuffer is sRGB, so
                     colours are converted to linear before writing)
    sampler2D prm_tex   row 0: shape parameters, row 1: viewer settings,
                        4 floats per texel
    vec3 pos  ->  vec3 world_pos  ->  vec4 FragColor (+ gl_FragDepth)
"""

import numpy as np

from . import ops
from .glsl_ops import HELPERS, OPS, WINNERS, SEAM_MARGINS, EDGE_MARGINS
from .ops import PRIM, XFORM, XFORM_AUX, DUNARY, DUNARY_AUX, DBINARY, PUSHP, POPP

# --- the op library ------------------------------------------------------------

_SIGNATURES = {
    PRIM: ('float', 'vec3 p, int o'),
    XFORM: ('vec3', 'vec3 p, int o'),
    XFORM_AUX: ('vec4', 'vec3 p, int o'),
    DUNARY: ('float', 'float d, int o'),
    DUNARY_AUX: ('float', 'float d, float a, int o'),
    DBINARY: ('float', 'float d1, float d2, int o'),
}

PRM_FROM_TEXTURE = '''
float PRM(int i) { return texelFetch(prm_tex, ivec2(i >> 2, 0), 0)[i & 3]; }
float SET(int i) { return texelFetch(prm_tex, ivec2(i >> 2, 1), 0)[i & 3]; }
'''


def library(used=None):
    """GLSL functions for the ops (all of them, or just the names in `used`)."""
    missing = set(ops.REGISTRY) - set(OPS)
    if missing:
        raise NotImplementedError(f'no GLSL for: {", ".join(sorted(missing))}')
    out = [HELPERS]
    for name, body in WINNERS.items():
        out.append(f'bool {name}(float d1, float d2, int o) {{ {body.strip()} }}')
    for name, body in SEAM_MARGINS.items():
        out.append(f'float s_{name}(float d1, float d2, int o) {{ {body.strip()} }}')
    for name, body in EDGE_MARGINS.items():
        if used is None or name in used:
            out.append(f'float e_{name}(vec3 p, int o) {{\n    {body.strip()}\n}}')
    for op in ops.REGISTRY.values():
        if used is not None and op.name not in used:
            continue
        ret, args = _SIGNATURES[op.kind]
        out.append(f'{ret} op_{op.name}({args}) {{\n    {OPS[op.name].strip()}\n}}')
    return '\n\n'.join(out) + '\n'


def _body(code, with_ids, with_edges=False):
    """GLSL statements for one program.  with_ids: return the winning
    primitive.  with_edges: return vec2(seam margin, primitive edge margin)
    carried along the winning branch of every boolean."""
    lines = []
    pt = dt = at = 0
    declared = set()

    def var(kind, name):
        if name in declared:
            return name
        declared.add(name)
        return f'{kind} {name}'

    for i, (c, o) in enumerate(code):
        c, o = int(c), int(o)
        if c == PUSHP:
            lines.append(f'{var("vec3", f"s{pt}")} = p;')
            pt += 1
            continue
        if c == POPP:
            pt -= 1
            lines.append(f'p = s{pt};')
            continue
        op = ops.REGISTRY_BY_CODE[c]
        f = f'op_{op.name}'
        if op.kind == PRIM:
            lines.append(f'{var("float", f"d{dt}")} = {f}(p, {o});')
            if with_ids:
                lines.append(f'{var("int", f"i{dt}")} = {i};')
            if with_edges:
                edge = f'e_{op.name}(p, {o})' if op.name in EDGE_MARGINS else '1e30'
                lines.append(f'{var("vec2", f"m{dt}")} = vec2(1e30, {edge});')
            dt += 1
        elif op.kind == XFORM:
            lines.append(f'p = {f}(p, {o});')
        elif op.kind == XFORM_AUX:
            lines.append(f'{var("vec4", f"t{at}")} = {f}(p, {o});')
            lines.append(f'p = t{at}.xyz;')
            lines.append(f'{var("float", f"a{at}")} = t{at}.w;')
            at += 1
        elif op.kind == DUNARY:
            lines.append(f'd{dt - 1} = {f}(d{dt - 1}, {o});')
        elif op.kind == DUNARY_AUX:
            at -= 1
            lines.append(f'd{dt - 1} = {f}(d{dt - 1}, a{at}, {o});')
        elif op.kind == DBINARY:
            dt -= 1
            a, b = dt - 1, dt
            if with_ids:
                lines.append(f'if (!{op.winner}(d{a}, d{b}, {o})) i{a} = i{b};')
            if with_edges:
                lines.append(f'if (!{op.winner}(d{a}, d{b}, {o})) m{a} = m{b};')
                lines.append(f'm{a}.x = min(m{a}.x, s_{op.name}(d{a}, d{b}, {o}));')
            lines.append(f'd{a} = {f}(d{a}, d{b}, {o});')
    if with_edges:
        lines.append('return m0;')
    else:
        lines.append('return i0;' if with_ids else 'return d0;')
    return lines


def shape_source(program):
    """`sdf`, `sdf_id` and `sdf_edges` for this program, plus the op
    functions it uses."""
    used = {ops.REGISTRY_BY_CODE[int(c)].name for c in program.code[:, 0]
            if int(c) >= ops.FIRST_OP_CODE}
    out = [library(used),
           'float sdf(vec3 p) {',
           *[f'    {s}' for s in _body(program.code, False)],
           '}', '',
           'int sdf_id(vec3 p) {',
           *[f'    {s}' for s in _body(program.code, True)],
           '}', '',
           'vec2 sdf_edges(vec3 p) {',
           *[f'    {s}' for s in _body(program.code, False, True)],
           '}', '']
    return '\n'.join(out)


# --- viewer settings: row 1 of the parameter texture -----------------------------

SETTINGS = [
    # name, default
    ('box_lo', (0.0, 0.0, 0.0)),
    ('box_hi', (0.0, 0.0, 0.0)),
    ('base_color', (0.8, 0.8, 0.8)),
    ('line_color', (0.05, 0.05, 0.05)),
    ('color_by_primitive', 0.0),
    ('show_seams', 1.0),
    ('show_creases', 1.0),
    ('shading_creases', 0.0),       # also find creases from the shading (fallback)
    ('crease_threshold', 0.35),     # for that: change in unit normal per pixel
    ('show_outline', 1.0),
    ('line_width', 1.5),            # pixels, for all lines
    ('smooth_edges', 1.0),          # anti-alias the silhouette (4 rays per edge pixel)
    ('max_steps', 160.0),
    ('step_scale', 1.0),            # < 1 for shapes that bend or twist space
    ('slice_axis', 2.0),            # 0, 1, 2 = x, y, z
    ('slice_offset', 0.0),
    ('slice_spacing', 0.1),
]


def settings_layout():
    """Name -> (first index, size) in the settings row."""
    layout, i = {}, 0
    for name, default in SETTINGS:
        n = len(default) if isinstance(default, tuple) else 1
        layout[name] = (i, n)
        i += n
    return layout


LAYOUT = settings_layout()
SETTINGS_SIZE = sum(n for _, n in LAYOUT.values())


def settings_array(**values):
    out = np.zeros(SETTINGS_SIZE, dtype=np.float32)
    for name, default in SETTINGS:
        v = values.get(name, default)
        i, n = LAYOUT[name]
        out[i:i + n] = v
    return out


def _settings_glsl():
    lines = []
    for name, (i, n) in LAYOUT.items():
        if n == 1:
            lines.append(f'#define {name.upper()} SET({i})')
        else:
            lines.append(f'#define {name.upper()} vec3(SET({i}), SET({i + 1}), SET({i + 2}))')
    return '\n'.join(lines) + '\n'


def parameter_texture_data(params, settings):
    """Both rows, padded to the same width, as float32 (2, width, 4)."""
    width = max(len(params), len(settings), 4)
    width = (width + 3) // 4
    data = np.zeros((2, width * 4), dtype=np.float32)
    data[0, :len(params)] = params
    data[1, :len(settings)] = settings
    return data.reshape(2, width, 4)


# --- the raymarching shader --------------------------------------------------------

VERTEX = '''
void main() {
    world_pos = pos;
    gl_Position = ViewProjection * vec4(pos, 1.0);
}
'''

TO_FRAMEBUFFER = '''
vec4 to_framebuffer(vec4 c) {
    // colours here are sRGB; an sRGB target expects linear values
    if (srgbTarget) {
        vec3 v = max(c.rgb, vec3(0.0));
        vec3 lo = v / 12.92;
        vec3 hi = pow((v + 0.055) / 1.055, vec3(2.4));
        c.rgb = mix(lo, hi, step(vec3(0.04045), v));
    }
    return c;
}
'''

FRAGMENT_MAIN = '''
float pixels_to_edge(float m) {
    // margin divided by its change per pixel = distance to the edge in pixels
    float g = length(vec2(dFdx(m), dFdy(m)));
    if (m >= 1e29 || g <= 0.0) return 1e30;
    return m / g;
}

float line_at(float px) {
    return 1.0 - smoothstep(0.5 * LINE_WIDTH - 0.5, 0.5 * LINE_WIDTH + 0.5, px);
}

float pixel_size(float t) {
    // world size of one pixel at distance t along the ray
    return (cam.w > 0.5) ? view_dir.w * t : view_dir.w;
}

vec3 sdf_normal(vec3 p, float h) {
    const vec2 k = vec2(1.0, -1.0);
    return normalize(k.xyy * sdf(p + k.xyy * h) + k.yyx * sdf(p + k.yyx * h) +
                     k.yxy * sdf(p + k.yxy * h) + k.xxx * sdf(p + k.xxx * h));
}

vec3 primitive_color(int id) {
    // spread hues by the golden ratio, keep them soft
    float h = fract(float(id) * 0.6180339887 + 0.13);
    vec3 c = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(0.85), c, 0.45);
}

bool march(vec3 ro, vec3 rd, vec3 lo, vec3 hi,
           out float t, out float closest, out float t_closest) {
    // only march where the ray is inside the box around the shape
    vec3 inv = 1.0 / rd;
    vec3 ta = (lo - ro) * inv, tb = (hi - ro) * inv;
    vec3 tmin = min(ta, tb), tmax = max(ta, tb);
    float t0 = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
    float t1 = min(min(tmax.x, tmax.y), tmax.z);
    t = t0;
    closest = 1e30;      // nearest miss, in pixels
    t_closest = t0;
    int steps = int(MAX_STEPS);
    for (int i = 0; i < steps; i++) {
        if (t > t1) break;
        float d = sdf(ro + rd * t);
        float px = pixel_size(t);
        if (d < 0.5 * px) return true;
        if (d / px < closest) { closest = d / px; t_closest = t; }
        t += d * STEP_SCALE;
    }
    return false;
}

vec3 shade(vec3 p, vec3 n, vec3 rd, int id) {
    vec3 base = (COLOR_BY_PRIMITIVE > 0.5) ? primitive_color(id) : BASE_COLOR;
    // headlight plus a little sky, so shapes read without scene lights
    float key = max(dot(n, -rd), 0.0);
    float sky = 0.5 + 0.5 * n.z;
    return base * (0.25 + 0.6 * key + 0.15 * sky) + 0.12 * pow(1.0 - key, 3.0);
}

void main() {
    vec3 ro, rd;
    vec3 lo = BOX_LO, hi = BOX_HI;
    if (cam.w > 0.5) {
        ro = cam.xyz;
        rd = normalize(world_pos - cam.xyz);
    } else {
        rd = normalize(view_dir.xyz);
        ro = world_pos - rd * 2.0 * length(hi - lo);
    }
    float t, closest, t_closest;
    bool hit = march(ro, rd, lo, hi, t, closest, t_closest);

    // everything below until the edge samples runs for every pixel, so the
    // screen-space derivatives used for lines and edges are well defined
    vec3 dxo = dFdx(ro), dyo = dFdy(ro), dxd = dFdx(rd), dyd = dFdy(rd);
    float hitf = hit ? 1.0 : 0.0;
    float t_end = hit ? t : t_closest;
    vec3 p = ro + rd * t_end;
    float px = pixel_size(t_end);
    vec3 n = sdf_normal(p, max(px, 1e-6));
    int id = sdf_id(p);

    // lines from edge margins: margin / (its change per pixel) = distance
    // to the edge in pixels, which gives smooth lines of a fixed width
    vec2 m = sdf_edges(p);
    float to_seam = pixels_to_edge(m.x), to_crease = pixels_to_edge(m.y);
    float to_rim = pixels_to_edge(max(dot(n, -rd), 0.0));
    float seam = (SHOW_SEAMS > 0.5) ? line_at(to_seam) : 0.0;
    float crease = (SHOW_CREASES > 0.5) ? line_at(to_crease) : 0.0;
    if (SHOW_CREASES > 0.5 && SHADING_CREASES > 0.5) {
        // fallback for edges no primitive reports: where the shading turns
        float turn = length(fwidth(n));
        crease = max(crease, smoothstep(CREASE_THRESHOLD, 2.0 * CREASE_THRESHOLD, turn));
    }
    // inner half of the outline: n.v falls to zero at a smooth silhouette
    float rim = (SHOW_OUTLINE > 0.5) ? line_at(to_rim) : 0.0;
    // a pixel the silhouette may cross: hit/miss differs within its 2x2
    // block, a missed ray came within a pixel, or an edge of the shape (which
    // is where sharp silhouettes are) lies within a pixel or so
    bool edge = fwidth(hitf) > 0.0 || (!hit && closest < 1.5) ||
                (hit && min(to_rim, min(to_seam, to_crease)) < 1.5);

    // how much of the pixel the shape covers: 4 sub-pixel rays, but only on
    // the silhouette, which is a small part of the screen
    float cover = hitf;
    vec3 p_shape = p;
    if (edge && SMOOTH_EDGES > 0.5) {
        const vec2 offs[4] = vec2[4](vec2(-0.125, -0.375), vec2(0.375, -0.125),
                                     vec2(0.125, 0.375), vec2(-0.375, 0.125));
        int hits = 0;
        bool found = hit;
        for (int k = 0; k < 4; k++) {
            vec3 o2 = ro + dxo * offs[k].x + dyo * offs[k].y;
            vec3 d2 = normalize(rd + dxd * offs[k].x + dyd * offs[k].y);
            float tk, ck, tck;
            if (march(o2, d2, lo, hi, tk, ck, tck)) {
                hits++;
                if (!found) { p_shape = o2 + d2 * tk; found = true; }
            }
        }
        cover = float(hits) / 4.0;
    }
    // the outline fills the background part of pixels near the silhouette
    float outline = 0.0;
    if (SHOW_OUTLINE > 0.5) {
        outline = hit ? (edge ? 1.0 : 0.0)
                      : 1.0 - smoothstep(LINE_WIDTH - 1.0, LINE_WIDTH, closest);
    }
    float alpha = cover + (1.0 - cover) * outline;
    if (alpha <= 0.0) discard;

    vec3 col;
    if (hit) {
        col = shade(p, n, rd, id);
        col = mix(col, LINE_COLOR, max(max(seam, crease), rim));
    } else {
        // a missed pixel the shape still partly covers: shade a sample that hit
        col = shade(p_shape, sdf_normal(p_shape, max(px, 1e-6)), rd, sdf_id(p_shape));
        if (SHOW_OUTLINE > 0.5) col = LINE_COLOR;
    }
    col = (cover * col + (1.0 - cover) * outline * LINE_COLOR) / alpha;
    FragColor = to_framebuffer(vec4(col, alpha));
    vec4 clip = ViewProjection * vec4(cover > 0.0 ? p_shape : p, 1.0);
    gl_FragDepth = 0.5 + 0.5 * clip.z / clip.w;
}
'''

SLICE_MAIN = '''
void main() {
    float d = sdf(world_pos);
    float s = SLICE_SPACING;
    float band = abs(fract(d / s + 0.5) - 0.5) * s;       // distance to nearest iso-line
    float w = fwidth(d);
    float line = 1.0 - smoothstep(0.0, 1.5 * w, band);
    float zero = 1.0 - smoothstep(0.0, 2.0 * w, abs(d));
    vec3 inside = vec3(0.85, 0.45, 0.35), outside = vec3(0.35, 0.55, 0.85);
    vec3 col = mix(outside, inside, step(d, 0.0)) * (0.75 + 0.25 * line);
    col = mix(col, LINE_COLOR, zero);
    FragColor = to_framebuffer(vec4(col, 0.9));
}
'''


def fragment_source(program):
    return (PRM_FROM_TEXTURE + _settings_glsl() + TO_FRAMEBUFFER +
            shape_source(program) + FRAGMENT_MAIN)


def slice_fragment_source(program):
    return (PRM_FROM_TEXTURE + _settings_glsl() + TO_FRAMEBUFFER +
            shape_source(program) + SLICE_MAIN)


def box_triangles(lo, hi):
    """The proxy box the raymarcher is drawn on: (36, 3) float32 positions."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[(hi if (i >> k) & 1 else lo)[k] for k in range(3)] for i in range(8)])
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = []
    for a, b, cc, d in quads:
        tris += [c[a], c[b], c[cc], c[a], c[cc], c[d]]
    return np.array(tris, dtype=np.float32)


def slice_quad(lo, hi, axis, offset):
    """Two triangles across the box at `offset` along `axis`."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    u, v = [k for k in range(3) if k != axis]
    corners = []
    for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
        q = np.zeros(3)
        q[axis] = offset
        q[u] = hi[u] if a else lo[u]
        q[v] = hi[v] if b else lo[v]
        corners.append(q)
    c = corners
    return np.array([c[0], c[1], c[2], c[0], c[2], c[3]], dtype=np.float32)
