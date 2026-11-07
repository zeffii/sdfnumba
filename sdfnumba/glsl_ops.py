"""GLSL versions of the ops in ops.py, for the GPU viewer.

Same names, same order, same maths.  Each entry is the body of a GLSL
function whose signature follows from the op's kind:

    PRIM        float op_NAME(vec3 p, int o)
    XFORM       vec3  op_NAME(vec3 p, int o)
    XFORM_AUX   vec4  op_NAME(vec3 p, int o)        xyz = point, w = aux
    DUNARY      float op_NAME(float d, int o)
    DUNARY_AUX  float op_NAME(float d, float a, int o)
    DBINARY     float op_NAME(float d1, float d2, int o)
    winners     bool  NAME(float d1, float d2, int o)

PRM(i) reads parameter i; the host shader defines it (in the viewer it
reads a texture, so changing parameter values only re-uploads a texture).

tests/test_glsl.py renders every op on the GPU (a software renderer is
enough) and compares it with the numba version, so the two can't drift.
"""

HELPERS = '''
vec3 PRM3(int o) { return vec3(PRM(o), PRM(o + 1), PRM(o + 2)); }
'''

OPS = {
    # --- primitives
    'sphere': 'return length(p) - PRM(o);',
    'plane': 'return dot(PRM3(o + 3) - p, PRM3(o));',
    'box': '''
        vec3 q = abs(p) - 0.5 * PRM3(o);
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);''',
    'rounded_box': '''
        float r = PRM(o + 3);
        vec3 q = abs(p) - 0.5 * PRM3(o) + r;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;''',
    'torus': '''
        float a = length(p.xy) - PRM(o);
        return length(vec2(a, p.z)) - PRM(o + 1);''',
    'capsule': '''
        vec3 a = PRM3(o), b = PRM3(o + 3);
        vec3 pa = p - a, ba = b - a;
        float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h) - PRM(o + 6);''',
    'cylinder': 'return length(p.xy) - PRM(o);',
    'capped_cylinder': '''
        vec3 a = PRM3(o), b = PRM3(o + 3);
        float r = PRM(o + 6);
        vec3 ba = b - a, pa = p - a;
        float baba = dot(ba, ba), paba = dot(pa, ba);
        float x = length(pa * baba - ba * paba) - r * baba;
        float y = abs(paba - baba * 0.5) - baba * 0.5;
        float x2 = x * x, y2 = y * y * baba;
        float d = (max(x, y) < 0.0) ? -min(x2, y2)
                : ((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0);
        return sign(d) * sqrt(abs(d)) / baba;''',
    'rounded_cylinder': '''
        float ra = PRM(o), rb = PRM(o + 1), h = PRM(o + 2);
        vec2 d = vec2(length(p.xy) - ra + rb, abs(p.z) - h * 0.5 + rb);
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - rb;''',
    'capped_cone': '''
        vec3 a = PRM3(o), b = PRM3(o + 3);
        float ra = PRM(o + 6), rb = PRM(o + 7);
        float rba = rb - ra;
        vec3 ba = b - a, pa = p - a;
        float baba = dot(ba, ba), papa = dot(pa, pa);
        float paba = dot(pa, ba) / baba;
        float q = sqrt(max(papa - paba * paba * baba, 0.0));
        float cax = max(0.0, q - ((paba < 0.5) ? ra : rb));
        float cay = abs(paba - 0.5) - 0.5;
        float k = rba * rba + baba;
        float f = clamp((rba * (q - ra) + paba * baba) / k, 0.0, 1.0);
        float cbx = q - ra - f * rba, cby = paba - f;
        float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
        return s * sqrt(min(cax * cax + cay * cay * baba,
                            cbx * cbx + cby * cby * baba));''',
    'rounded_cone': '''
        float r1 = PRM(o), r2 = PRM(o + 1), h = PRM(o + 2);
        vec2 q = vec2(length(p.xy), p.z);
        float b = (r1 - r2) / h;
        float a = sqrt(1.0 - b * b);
        float k = dot(q, vec2(-b, a));
        if (k < 0.0) return length(q) - r1;
        if (k > a * h) return length(q - vec2(0.0, h)) - r2;
        return dot(q, vec2(a, b)) - r1;''',
    'ellipsoid': '''
        vec3 s = PRM3(o);
        float k0 = length(p / s), k1 = length(p / (s * s));
        if (k1 == 0.0) return -min(s.x, min(s.y, s.z));
        return k0 * (k0 - 1.0) / k1;''',
    'tetrahedron': '''
        return (max(abs(p.x + p.y) - p.z, abs(p.x - p.y) + p.z) - PRM(o)) / sqrt(3.0);''',
    'octahedron': '''
        return (abs(p.x) + abs(p.y) + abs(p.z) - PRM(o)) * 0.5773502691896258;''',

    # --- 2D profiles
    'arc': '''
        float r = PRM(o), a0 = PRM(o + 1), a1 = PRM(o + 2);
        float c = 0.5 * (a0 + a1), span = abs(a1 - c);   // `half` is reserved in GLSL
        float qx = cos(c) * p.x + sin(c) * p.y;
        float qy = abs(-sin(c) * p.x + cos(c) * p.y);
        float psi = clamp(atan(qy, qx), 0.0, span);
        return length(vec2(qx - r * cos(psi), qy - r * sin(psi)));''',

    'polygon': '''
        int m = int(PRM(o));
        float d2 = 1e30, s = 1.0;
        for (int i = 0; i < m; i++) {
            int k = o + 1 + 4 * i;
            vec2 a = vec2(PRM(k), PRM(k + 1)), b = vec2(PRM(k + 2), PRM(k + 3));
            vec2 e = b - a, w = p.xy - a;
            float ee = dot(e, e);
            float t = (ee == 0.0) ? 0.0 : clamp(dot(w, e) / ee, 0.0, 1.0);
            vec2 q = w - e * t;
            d2 = min(d2, dot(q, q));
            bvec3 c = bvec3(p.y >= a.y, p.y < b.y, e.x * w.y > e.y * w.x);
            if (all(c) || all(not(c))) s = -s;
        }
        return s * sqrt(d2);''',

    # --- point transforms
    'translate': 'return p - PRM3(o);',
    'scale_point': 'return p / PRM3(o);',
    'matrix': '''
        return vec3(dot(p, vec3(PRM(o), PRM(o + 3), PRM(o + 6))),
                    dot(p, vec3(PRM(o + 1), PRM(o + 4), PRM(o + 7))),
                    dot(p, vec3(PRM(o + 2), PRM(o + 5), PRM(o + 8))));''',
    'twist': '''
        float c = cos(PRM(o) * p.z), s = sin(PRM(o) * p.z);
        return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);''',
    'bend': '''
        float c = cos(PRM(o) * p.x), s = sin(PRM(o) * p.x);
        return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);''',
    'repeat': '''
        vec3 s = PRM3(o), c = PRM3(o + 3), n = PRM3(o + 6);
        for (int k = 0; k < 3; k++) {
            if (s[k] != 0.0) {
                float i = roundEven(p[k] / s[k]);   // Python rounds half to even
                if (c[k] >= 0.0) i = clamp(i, -c[k], c[k]);
                p[k] -= s[k] * (i + n[k]);
            }
        }
        return p;''',
    'polar_fold': '''
        float da = PRM(o), shift = PRM(o + 1);
        float d = length(p.xy);
        float a = mod(atan(p.y, p.x), da) - shift * da;
        return vec3(cos(a) * d, sin(a) * d, p.z);''',
    'revolve': 'return vec3(length(p.xy) - PRM(o), p.z, 0.0);',
    'elongate': '''
        vec3 q = abs(p) - PRM3(o);
        float w = min(max(q.x, max(q.y, q.z)), 0.0);
        return vec4(max(q, 0.0), w);''',

    # --- distance modifiers
    'scale_dist': 'return d * PRM(o);',
    'negate': 'return -d;',
    'offset': 'return d + PRM(o);',
    'shell': 'return abs(d) - PRM(o) * 0.5;',
    'add_aux': 'return d + a;',

    # --- combiners (k = 0 means hard)
    'union': '''
        float k = PRM(o);
        if (k == 0.0) return min(d1, d2);
        float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
        return d2 + (d1 - d2) * h - k * h * (1.0 - h);''',
    'difference': '''
        float k = PRM(o);
        if (k == 0.0) return max(d1, -d2);
        float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
        return d1 + (-d2 - d1) * h + k * h * (1.0 - h);''',
    'intersection': '''
        float k = PRM(o);
        if (k == 0.0) return max(d1, d2);
        float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
        return d2 + (d1 - d2) * h + k * h * (1.0 - h);''',
    'blend': '''
        float k = PRM(o);
        return k * d2 + (1.0 - k) * d1;''',
}

WINNERS = {
    'first_if_smaller': 'return d1 <= d2;',
    'first_if_larger': 'return d1 >= d2;',
    'first_if_not_cut': 'return d1 >= -d2;',
    'first_if_heavier': 'return PRM(o) <= 0.5;',
}

# --- edge margins, for anti-aliased contour lines in the viewer
#
# A margin is zero exactly on an edge and grows away from it; the viewer
# divides it by its screen-space rate of change to get a distance in pixels.
#
# Hard booleans: how close the op was to picking the other side.  Smooth ones
# (k > 0) blend without an edge, so they report none (1e30).
#     float s_NAME(float d1, float d2, int o)
SEAM_MARGINS = {
    'union': 'return (PRM(o) == 0.0) ? abs(d1 - d2) : 1e30;',
    'intersection': 'return (PRM(o) == 0.0) ? abs(d1 - d2) : 1e30;',
    'difference': 'return (PRM(o) == 0.0) ? abs(d1 + d2) : 1e30;',
    'blend': 'return 1e30;',
}

# A primitive's own sharp edges, where it has them.  Primitives not listed
# are smooth (or their edges are left to the optional shading-based test).
#     float e_NAME(vec3 p, int o)
EDGE_MARGINS = {
    'box': '''
        vec3 q = abs(p) - 0.5 * PRM3(o);
        // an edge is where the largest two components of q are equal
        float a = max(q.x, max(q.y, q.z));
        float b = (a == q.x) ? max(q.y, q.z) : ((a == q.y) ? max(q.x, q.z) : max(q.x, q.y));
        return a - b;''',
    'capped_cylinder': '''
        vec3 a = PRM3(o), b = PRM3(o + 3);
        vec3 ba = b - a, pa = p - a;
        float baba = dot(ba, ba), paba = dot(pa, ba);
        float x = length(pa * baba - ba * paba) - PRM(o + 6) * baba;
        float y = abs(paba - baba * 0.5) - baba * 0.5;
        return abs(x - y);''',
    'tetrahedron': '''
        float s = abs(p.x + p.y) - p.z, t = abs(p.x - p.y) + p.z;
        return min(abs(s - t), (s >= t) ? abs(p.x + p.y) : abs(p.x - p.y));''',
    'octahedron': 'return min(abs(p.x), min(abs(p.y), abs(p.z)));',
}
