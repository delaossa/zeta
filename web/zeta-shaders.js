// Display shaders for the web viewer (ports of the ones in zeta_viewer.py).
"use strict";

// colour schemes: 6 cyclic key colours each, Catmull-Rom interpolated (arg 0 = first colour)
const SCHEMES = {
    "Rainbow":    [[1, .15, .15], [1, .85, .1], [.15, .85, .25], [.1, .8, 1], [.25, .3, 1], [.95, .2, .9]],
    "Neon":       [[1, .1, .55], [1, .55, .1], [.95, 1, .3], [.1, 1, .75], [.1, .5, 1], [.55, .15, 1]],
    "Twilight":   [[.89, .85, .88], [.55, .66, .80], [.34, .30, .62], [.19, .07, .23], [.55, .17, .25], [.80, .55, .45]],
    "Fire & Ice": [[1, .97, .75], [1, .55, .08], [.55, .05, .08], [.04, .04, .12], [.08, .35, .8], [.6, .92, 1]],
    "Pastel":     [[1, .55, .55], [1, .92, .5], [.55, .92, .6], [.5, .9, 1], [.6, .62, 1], [.97, .58, .95]],
    "Monochrome": [[.95, .95, .95], [.75, .75, .75], [.4, .4, .4], [.1, .1, .1], [.4, .4, .4], [.75, .75, .75]],
    "HSV":        [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]],
    "Phase":      [[.66, .47, .04], [.86, .24, .34], [.75, .17, .75], [.38, .35, .95], [.09, .56, .66], [.36, .62, .12]],
    "Ocean":      [[.02, .10, .30], [.05, .35, .60], [.20, .70, .80], [.85, .95, .95], [.40, .75, .60], [.10, .35, .35]],
    "Sunset":     [[.15, .05, .30], [.55, .10, .45], [.90, .30, .35], [1, .65, .30], [1, .90, .60], [.50, .35, .60]],
    "Aurora":     [[.05, .10, .20], [.10, .60, .50], [.50, 1, .60], [.30, .50, 1], [.70, .30, .90], [.30, .10, .40]],
    "Copper":     [[.95, .75, .50], [.80, .45, .20], [.40, .18, .08], [.08, .20, .22], [.20, .55, .55], [.70, .85, .75]],
    "Earth":      [[.20, .35, .15], [.55, .60, .25], [.90, .85, .60], [.70, .50, .30], [.35, .20, .15], [.15, .25, .35]],
    "Blue-Orange": [[.10, .30, .80], [.50, .70, 1], [.95, .95, .95], [1, .70, .40], [.85, .35, .05], [.35, .15, .40]],
};
const SCHEME_NAMES = Object.keys(SCHEMES);
const DEFAULT_SCHEME = "Aurora";

// same interpolation as the shader, for the colour bar (drawn on the 2D overlay)
function paletteJS(scheme, x) {
    const P = SCHEMES[SCHEME_NAMES[scheme]];
    const y = (x - Math.floor(x)) * 6;
    const i = Math.min(5, Math.floor(y)), f = y - i;
    const p0 = P[(i + 5) % 6], p1 = P[i], p2 = P[(i + 1) % 6], p3 = P[(i + 2) % 6];
    return [0, 1, 2].map((k) => Math.min(1, Math.max(0, 0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * f
        + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * f * f + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * f * f * f))));
}

const PALETTE_GLSL = `
uniform vec3 u_pal[6];
vec3 palette(float x) {
    float y = fract(x) * 6.0;
    int i = min(int(floor(y)), 5);
    float f = y - float(i);
    vec3 p0 = u_pal[(i + 5) % 6], p1 = u_pal[i], p2 = u_pal[(i + 1) % 6], p3 = u_pal[(i + 2) % 6];
    vec3 c = 0.5 * (2.0 * p1 + (-p0 + p2) * f + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f * f
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f * f * f);
    return clamp(c, 0.0, 1.0);
}
vec3 hue(float a) { return palette(a / TAU); }
`;

const SHADE_GLSL = PALETTE_GLSL + `
uniform int u_plain;        // 1: colours off -> neutral background with contour lines only
uniform int u_style;        // 0 hue + brightness   1 hue only (arg)   2 brightness only (|zeta|)   3 lines only
uniform int u_markers;      // 1: small dark marker at each zero
float isoline(float x, float fw, float w) {
    float d = abs(fract(x + 0.5) - 0.5);
    float l = 1.0 - smoothstep(0.0, max(fw * w, 1e-6), d);
    return l * (1.0 - smoothstep(0.08, 0.3, fw));
}
vec3 domain_color(float logm, vec2 ph, float flow, int contours) {
    float a = atan(ph.y, ph.x);
    vec3 col = u_style == 2 ? vec3(0.95) : hue(a);
    float lx = clamp(logm, -60.0, 60.0) * 1.4426950408889634;
    // for large sigma zeta ~ 1 + 2^-s: log|zeta| and arg zeta sink into float noise -> treat as flat
    if (abs(lx) < 1e-4) lx = 1e-4;
    float x  = lx - flow;
    float fx = fwidth(x);
    float px = (abs(a) < 1e-4 ? 1e-4 : a) * (6.0 / PI) + flow;
    float fp = length(fwidth(ph)) * (6.0 / PI);
    if (u_plain != 0) {
        vec3 c = vec3(0.94, 0.94, 0.92);
        if (contours != 0) {
            c = mix(c, vec3(0.5), 0.75 * isoline(x, fx, 1.2));          // |zeta| = 2^k
            c = mix(c, vec3(0.05), 0.85 * isoline(px, fp, 1.2));        // arg zeta = k pi/6
        }
        return c;
    }
    if (u_style != 1) {                     // brightness bands: one dark-to-bright ramp per doubling of |zeta|
        float saw = fract(x);
        saw *= 1.0 - smoothstep(1.0 - min(fx, 0.5), 1.0, saw);
        float fade = 1.0 - smoothstep(0.15, 0.6, fx);
        col *= mix(0.72, 0.45 + 0.55 * saw, fade);
    }
    if (contours != 0 && u_style != 2)
        col = mix(col, col * 0.12, 0.7 * isoline(px, fp, 1.3));
    col = mix(col, vec3(1.0), 0.55 * smoothstep(5.0, 40.0, logm));
    return col;
}
float zero_mask(float logm) {
    if (u_markers == 0) return 0.0;
    float m = exp(min(logm, 10.0));
    float r = m / max(fwidth(m), 1e-20);              // ~ distance to the zero in pixels
    return (1.0 - smoothstep(2.5, 4.5, r)) * (1.0 - smoothstep(-1.5, -1.0, logm));
}
vec3 apply_zero(vec3 col, float logm) {
    return mix(col, vec3(0.0), zero_mask(logm));
}
vec3 grid_overlay(vec3 col, vec2 s, vec2 step, float axis_t) {
    vec2 g = s / step;
    vec2 fw = fwidth(g);
    float l = max(isoline(g.x, fw.x, 1.0), isoline(g.y, fw.y, 1.0));
    vec3 gc = u_plain != 0 ? vec3(0.1, 0.12, 0.2) : vec3(1.0);
    if (u_plain != 0) {
        vec2 g5 = g * 5.0;
        vec2 f5 = fw * 5.0;
        col = mix(col, gc, 0.13 * max(isoline(g5.x, f5.x, 1.0), isoline(g5.y, f5.y, 1.0)));
    }
    col = mix(col, gc, (u_plain != 0 ? 0.35 : 0.16) * l);
    vec2 fa = fwidth(s);
    float ax = max(1.0 - smoothstep(0.0, 1.5 * fa.x, abs(s.x)), 1.0 - smoothstep(0.0, 1.5 * fa.y, abs(s.y - axis_t)));
    return mix(col, gc, (u_plain != 0 ? 0.7 : 0.45) * ax);
}
`;

const PLANE_FS = GLSL_HEADER + SHADE_GLSL + `
uniform sampler2D u_tex;
uniform vec2 u_crel, u_half, u_res, u_step;
uniform float u_time, u_flow, u_axis_t;
uniform int u_contours, u_grid;
out vec4 o;
void main() {
    vec2 uv = gl_FragCoord.xy / u_res;
    vec4 d = texture(u_tex, uv);
    vec2 s = u_crel + (uv * 2.0 - 1.0) * u_half;
    float logm = d.x;
    vec2 ph = d.yz / max(length(d.yz), 1e-9);
    vec3 col = domain_color(logm, ph, u_flow, u_contours);
    if (isnan(logm)) col = vec3(1.0);
    if (u_grid != 0) col = grid_overlay(col, s, u_step, u_axis_t);
    float zm = zero_mask(logm);
    float px = 2.0 * u_half.x / u_res.x;
    float dc = abs(s.x - 0.5) / px;
    float pulse = 0.8 + 0.2 * sin(u_time * 2.5);
    if (u_plain != 0)
        col = mix(col, vec3(0.1, 0.4, 0.85), 0.9 * (1.0 - smoothstep(0.0, 1.5, dc)));
    else
        col += vec3(0.25, 0.85, 1.0) * (exp(-dc * dc / 2.0) * 0.8 + exp(-dc / 10.0) * 0.12) * pulse * (1.0 - zm);
    float d1 = abs(s.x - 1.0) / px;
    float py = 2.0 * u_half.y / u_res.y;
    vec3 lc = u_plain != 0 ? vec3(0.2) : vec3(1.0);
    col = mix(col, lc, 0.25 * (1.0 - smoothstep(0.0, 1.2, d1)) * step(0.5, fract(s.y / (12.0 * py))));
    col = apply_zero(col, logm);
    if (u_plain == 0) col *= 1.0 - 0.35 * pow(length(uv - 0.5) * 1.3, 2.5);
    o = vec4(col, 1.0);
}
`;

const BG_FS = GLSL_HEADER + `
in vec2 v_uv;
out vec4 o;
void main() {
    vec3 col = mix(vec3(0.012, 0.012, 0.03), vec3(0.07, 0.05, 0.14), smoothstep(0.0, 1.0, v_uv.y));
    col += vec3(0.05, 0.08, 0.16) * exp(-6.0 * dot(v_uv - vec2(0.5, 0.35), v_uv - vec2(0.5, 0.35)));
    o = vec4(col, 1.0);
}
`;

const LAND_VS = GLSL_HEADER + `
uniform sampler2D u_tex;
uniform int u_res, u_hmode;
uniform vec2 u_crel, u_half;
uniform mat4 u_mvp;
uniform float u_hscale;
out vec3 v_pos; out vec3 v_nrm; out float v_logm; out vec2 v_ph; out vec2 v_s;
float height(float lm) {
    if (u_hmode == 0) { float m = exp(min(lm, 30.0)); return 6.0 * (1.0 - exp(-m / 6.0)); }
    return 4.0 * tanh(clamp(lm, -20.0, 20.0) / 4.0);
}
float hat(int i, int j) {
    return height(texelFetch(u_tex, ivec2(clamp(i, 0, u_res - 1), clamp(j, 0, u_res - 1)), 0).x) * u_hscale;
}
void main() {
    int i = gl_VertexID % u_res, j = gl_VertexID / u_res;
    vec4 d = texelFetch(u_tex, ivec2(i, j), 0);
    float ar = u_half.x / u_half.y;
    vec2 uv = vec2(i, j) / float(u_res - 1);
    v_pos = vec3((uv.x * 2.0 - 1.0) * ar, height(d.x) * u_hscale, -(uv.y * 2.0 - 1.0));
    float dx = 2.0 * ar / float(u_res - 1), dz = 2.0 / float(u_res - 1);
    float gx = (hat(i + 1, j) - hat(i - 1, j)) / (2.0 * dx);
    float gz = (hat(i, j + 1) - hat(i, j - 1)) / (-2.0 * dz);
    v_nrm = normalize(vec3(-gx, 1.0, -gz));
    v_logm = d.x; v_ph = d.yz;
    v_s = u_crel + (uv * 2.0 - 1.0) * u_half;
    gl_Position = u_mvp * vec4(v_pos, 1.0);
}
`;

const LAND_FS = GLSL_HEADER + SHADE_GLSL + `
in vec3 v_pos; in vec3 v_nrm; in float v_logm; in vec2 v_ph; in vec2 v_s;
uniform vec3 u_eye;
uniform vec2 u_step;
uniform float u_flow, u_time, u_axis_t;
uniform int u_contours, u_grid;
out vec4 o;
void main() {
    vec2 ph = v_ph / max(length(v_ph), 1e-9);
    vec3 base = domain_color(v_logm, ph, u_flow, u_contours);
    if (u_grid != 0) base = grid_overlay(base, v_s, u_step, u_axis_t);
    float zm = zero_mask(v_logm);
    base = apply_zero(base, v_logm);
    vec3 N = normalize(v_nrm), V = normalize(u_eye - v_pos);
    if (dot(N, V) < 0.0) N = -N;
    vec3 L1 = normalize(vec3(-0.4, 0.9, 0.5)), L2 = normalize(vec3(0.6, 0.3, -0.7));
    float diff = max(dot(N, L1), 0.0), diff2 = max(dot(N, L2), 0.0);
    float spec = pow(max(dot(N, normalize(L1 + V)), 0.0), 48.0);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    vec3 col = base * (0.16 + 0.85 * diff + 0.25 * diff2) + vec3(0.45) * spec + rim * vec3(0.25, 0.45, 1.0) * 0.45;
    float fw = max(fwidth(v_s.x), 1e-6);
    float dc = abs(v_s.x - 0.5) / fw;
    float cl = exp(-dc * dc / 3.0) * (1.0 - zm);
    if (u_plain != 0) col = mix(col, vec3(0.1, 0.4, 0.85), cl);
    else col += vec3(0.3, 0.9, 1.0) * cl * (1.0 + 0.25 * sin(u_time * 2.5));
    float dist = length(u_eye - v_pos);
    col = mix(col, vec3(0.03, 0.03, 0.07), 1.0 - exp(-max(dist - 2.0, 0.0) * 0.12));
    o = vec4(col, 1.0);
}
`;

const SPIRAL_VS = GLSL_HEADER + PALETTE_GLSL + `
uniform sampler2D u_line;           // (Re zeta, Im zeta, theta, log|zeta|)
uniform float u_head, u_len, u_k, u_psize, u_gain, u_unwind;
uniform int u_n, u_done, u_anim, u_plain;
uniform mat4 u_mvp;
out vec3 v_col;
void main() {
    int idx = gl_VertexID;
    if (idx >= u_done) { gl_Position = vec4(3.0, 3.0, 3.0, 1.0); v_col = vec3(0.0); gl_PointSize = 1.0; return; }
    vec4 d = texelFetch(u_line, ivec2(idx & 4095, idx >> 12), 0);
    float f = float(idx) / float(u_n - 1);
    float c = cos(u_unwind * d.z), s = sin(u_unwind * d.z);
    vec2 z = vec2(d.x * c - d.y * s, d.x * s + d.y * c);   // zeta * e^{i u theta}; u=1 -> Hardy Z(t) (real)
    gl_Position = u_mvp * vec4((f - 0.5) * u_len, z.y * u_k, z.x * u_k, 1.0);
    float a = 0.6;
    if (u_anim != 0) a = f > u_head ? 0.05 : 0.1 + 0.9 * exp(-(u_head - f) / 0.12);
    float zg = exp(-length(z) * 9.0);
    vec3 col = u_plain != 0 ? vec3(0.8) : mix(palette(f * 3.0), vec3(1.0), 0.2);
    v_col = (col + vec3(1.0, 0.95, 0.8) * zg * 4.0) * a * u_gain;
    gl_PointSize = u_psize * (1.0 + 3.0 * zg);
}
`;

const SPIRAL_FS = GLSL_HEADER + `
in vec3 v_col;
uniform int u_points;
out vec4 o;
void main() {
    float a = 1.0;
    if (u_points != 0) {
        vec2 q = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(q, q);
        if (r2 > 1.0) discard;
        a = exp(-r2 * 3.5);
    }
    o = vec4(v_col * a, 1.0);
}
`;

const LINES_VS = GLSL_HEADER + `
in vec3 in_pos; in vec4 in_col;
uniform mat4 u_mvp;
out vec4 v_col;
void main() { v_col = in_col; gl_Position = u_mvp * vec4(in_pos, 1.0); }
`;
const LINES_FS = GLSL_HEADER + `
in vec4 v_col; out vec4 o;
void main() { o = v_col; }
`;
