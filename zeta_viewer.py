#!/usr/bin/env python3
"""
Riemann zeta explorer -- GPU accelerated (OpenGL 4.1 via moderngl + glfw).

zeta(s) is evaluated on the GPU (see zeta_core.py):
  Euler-Maclaurin + functional equation for |t| < 1000, Riemann-Siegel above
  (works to t ~ 1e10 and beyond; heavier views refine progressively).

Views
  1  Complex plane   domain colouring: hue = arg zeta, bands = log2|zeta|, white glow = zeros
  2  Landscape       3D surface of |zeta| (or log|zeta|) over the same region, coloured by phase
  3  Critical line   3D trace of zeta(sigma + it) along the t axis; hits the axis at each zero

Press ? in the window for all controls.
"""
import argparse
import math
import os
import struct
import sys
import time
import zlib

import numpy as np
import glfw
import moderngl

from zeta_core import (FULLSCREEN_VS, COMPUTE_FS, LINE_FS, PhaseTable, terms_needed, cost_per_sample,
                       RS_THRESHOLD, TAU)

try:
    import mpmath
except ImportError:
    mpmath = None

HEADER = "#version 410 core\nconst float PI = 3.14159265358979;\nconst float TAU = 6.28318530717959;\n"

# --------------------------------------------------------------------------------------
# colour schemes: 6 cyclic key colours each, Catmull-Rom interpolated (arg 0 = first colour)
# --------------------------------------------------------------------------------------
SCHEMES = {
    "Rainbow":    [(1, .15, .15), (1, .85, .1), (.15, .85, .25), (.1, .8, 1), (.25, .3, 1), (.95, .2, .9)],
    "Neon":       [(1, .1, .55), (1, .55, .1), (.95, 1, .3), (.1, 1, .75), (.1, .5, 1), (.55, .15, 1)],
    "Twilight":   [(.89, .85, .88), (.55, .66, .80), (.34, .30, .62), (.19, .07, .23), (.55, .17, .25), (.80, .55, .45)],
    "Fire & Ice": [(1, .97, .75), (1, .55, .08), (.55, .05, .08), (.04, .04, .12), (.08, .35, .8), (.6, .92, 1)],
    "Pastel":     [(1, .55, .55), (1, .92, .5), (.55, .92, .6), (.5, .9, 1), (.6, .62, 1), (.97, .58, .95)],
    "Monochrome": [(.95, .95, .95), (.75, .75, .75), (.4, .4, .4), (.1, .1, .1), (.4, .4, .4), (.75, .75, .75)],
    "HSV":        [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)],
    "Phase":      [(.66, .47, .04), (.86, .24, .34), (.75, .17, .75), (.38, .35, .95), (.09, .56, .66), (.36, .62, .12)],
    "Ocean":      [(.02, .10, .30), (.05, .35, .60), (.20, .70, .80), (.85, .95, .95), (.40, .75, .60), (.10, .35, .35)],
    "Sunset":     [(.15, .05, .30), (.55, .10, .45), (.90, .30, .35), (1, .65, .30), (1, .90, .60), (.50, .35, .60)],
    "Aurora":     [(.05, .10, .20), (.10, .60, .50), (.50, 1, .60), (.30, .50, 1), (.70, .30, .90), (.30, .10, .40)],
    "Copper":     [(.95, .75, .50), (.80, .45, .20), (.40, .18, .08), (.08, .20, .22), (.20, .55, .55), (.70, .85, .75)],
    "Earth":      [(.20, .35, .15), (.55, .60, .25), (.90, .85, .60), (.70, .50, .30), (.35, .20, .15), (.15, .25, .35)],
    "Blue-Orange": [(.10, .30, .80), (.50, .70, 1), (.95, .95, .95), (1, .70, .40), (.85, .35, .05), (.35, .15, .40)],
}
SCHEME_NAMES = list(SCHEMES)
DEFAULT_SCHEME = "Aurora"

PALETTE_GLSL = """
uniform int u_scheme;
const vec3 PAL[%d] = vec3[%d](%s);
vec3 palette(float x) {
    float y = fract(x) * 6.0;
    int i = int(floor(y));
    float f = y - float(i);
    int b = u_scheme * 6;
    vec3 p0 = PAL[b + (i + 5) %% 6], p1 = PAL[b + i], p2 = PAL[b + (i + 1) %% 6], p3 = PAL[b + (i + 2) %% 6];
    vec3 c = 0.5 * (2.0 * p1 + (-p0 + p2) * f + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f * f
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f * f * f);
    return clamp(c, 0.0, 1.0);
}
vec3 hue(float a) { return palette(a / TAU); }
""" % (6 * len(SCHEMES), 6 * len(SCHEMES),
       ", ".join("vec3(%g, %g, %g)" % c for cols in SCHEMES.values() for c in cols))

SHADE_GLSL = PALETTE_GLSL + """
uniform int u_plain;        // 1: colours off -> neutral background with contour lines only
uniform int u_style;        // 0 hue + brightness   1 hue only (arg)   2 brightness only (|zeta|)   3 lines only
uniform int u_markers;      // 1: small dark marker at each zero (|zeta| itself is shown by the brightness bands)
float isoline(float x, float fw, float w) {
    float d = abs(fract(x + 0.5) - 0.5);
    float l = 1.0 - smoothstep(0.0, max(fw * w, 1e-6), d);
    return l * (1.0 - smoothstep(0.08, 0.3, fw));
}
vec3 domain_color(float logm, vec2 ph, float flow, int contours) {
    float a = atan(ph.y, ph.x);
    vec3 col = u_style == 2 ? vec3(0.95) : hue(a);
    float lx = clamp(logm, -60.0, 60.0) * 1.4426950408889634;
    // for large sigma zeta ~ 1 + 2^-s: log|zeta| and arg zeta sink into float noise (~1e-6) and would
    // smear the |zeta| = 1 contour / arg = 0 phase line over the whole region -> treat that as flat
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
        saw *= 1.0 - smoothstep(1.0 - min(fx, 0.5), 1.0, saw);     // anti-aliased drop at |zeta| = 2^k
        float fade = 1.0 - smoothstep(0.15, 0.6, fx);              // too dense to resolve -> average
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
    float dot_px = (1.0 - smoothstep(2.5, 4.5, r)) * (1.0 - smoothstep(-1.5, -1.0, logm));
    return dot_px;                                      // fixed ~3-4 px dot at any zoom
}
vec3 apply_zero(vec3 col, float logm) {
    return mix(col, vec3(0.0), zero_mask(logm));
}
// s relative to a grid origin that is a multiple of step; axis_t = position of t = 0 in that frame
vec3 grid_overlay(vec3 col, vec2 s, vec2 step, float axis_t) {
    vec2 g = s / step;
    vec2 fw = fwidth(g);
    float l = max(isoline(g.x, fw.x, 1.0), isoline(g.y, fw.y, 1.0));
    vec3 gc = u_plain != 0 ? vec3(0.1, 0.12, 0.2) : vec3(1.0);
    if (u_plain != 0) {                                   // finer grid: 5 minor lines per step
        vec2 g5 = g * 5.0;
        vec2 f5 = fw * 5.0;
        col = mix(col, gc, 0.13 * max(isoline(g5.x, f5.x, 1.0), isoline(g5.y, f5.y, 1.0)));
    }
    col = mix(col, gc, (u_plain != 0 ? 0.35 : 0.16) * l);
    vec2 fa = fwidth(s);
    float ax = max(1.0 - smoothstep(0.0, 1.5 * fa.x, abs(s.x)), 1.0 - smoothstep(0.0, 1.5 * fa.y, abs(s.y - axis_t)));
    return mix(col, gc, (u_plain != 0 ? 0.7 : 0.45) * ax);
}
"""

PLANE_FS = HEADER + SHADE_GLSL + """
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
"""

BG_FS = HEADER + """
in vec2 v_uv;
out vec4 o;
void main() {
    vec3 col = mix(vec3(0.012, 0.012, 0.03), vec3(0.07, 0.05, 0.14), smoothstep(0.0, 1.0, v_uv.y));
    col += vec3(0.05, 0.08, 0.16) * exp(-6.0 * dot(v_uv - vec2(0.5, 0.35), v_uv - vec2(0.5, 0.35)));
    o = vec4(col, 1.0);
}
"""

LAND_VS = HEADER + """
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
"""

LAND_FS = HEADER + SHADE_GLSL + """
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
"""

SPIRAL_VS = HEADER + PALETTE_GLSL + """
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
"""

SPIRAL_FS = HEADER + """
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
"""

CBAR_VS = HEADER + """
uniform vec4 u_rect;          // x0, y0, x1, y1 in pixels (y down)
uniform vec2 u_res;
out vec2 v_f;
void main() {
    vec2 c = vec2(gl_VertexID & 1, gl_VertexID >> 1);
    v_f = c;
    vec2 p = mix(u_rect.xy, u_rect.zw, c);
    gl_Position = vec4(p.x / u_res.x * 2.0 - 1.0, 1.0 - p.y / u_res.y * 2.0, 0.0, 1.0);
}
"""
CBAR_FS = HEADER + PALETTE_GLSL + """
uniform int u_kind;           // 0: arg  1: log2|zeta| bands  2: position in t (3 palette cycles)
uniform float u_lo, u_hi;
in vec2 v_f;
out vec4 o;
void main() {
    vec3 col;
    if (u_kind == 0) col = hue(mix(-PI, PI, v_f.x));
    else if (u_kind == 1) col = vec3(0.95) * (0.45 + 0.55 * fract(mix(u_lo, u_hi, v_f.x)));
    else col = mix(palette(v_f.x * 3.0), vec3(1.0), 0.2);
    o = vec4(col, 1.0);
}
"""

LINES_VS = HEADER + """
in vec3 in_pos; in vec4 in_col;
uniform mat4 u_mvp;
out vec4 v_col;
void main() { v_col = in_col; gl_Position = u_mvp * vec4(in_pos, 1.0); }
"""
LINES_FS = HEADER + """
in vec4 v_col; out vec4 o;
void main() { o = v_col; }
"""

TEXT_VS = HEADER + """
in vec2 in_pos; in vec2 in_uv; in vec4 in_col;
uniform vec2 u_res;
out vec2 v_uv; out vec4 v_col;
void main() {
    v_uv = in_uv; v_col = in_col;
    gl_Position = vec4(in_pos.x / u_res.x * 2.0 - 1.0, 1.0 - in_pos.y / u_res.y * 2.0, 0.0, 1.0);
}
"""
TEXT_FS = HEADER + """
uniform sampler2D u_atlas;
in vec2 v_uv; in vec4 v_col;
out vec4 o;
void main() {
    vec2 m = texture(u_atlas, v_uv).rg;     // r: glyph, g: dilated outline
    o = vec4(v_col.rgb * m.r * v_col.a, max(m.r, 0.8 * m.g) * v_col.a);   // premultiplied
}
"""

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(f, (0, 0, -1))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class OrbitCam:
    def __init__(self, yaw, pitch, dist, target):
        self.init = (yaw, pitch, dist, tuple(target))
        self.goal = None
        self.reset()

    def reset(self):
        self.yaw, self.pitch, self.dist, t = self.init
        self.target = np.array(t, dtype=np.float64)
        self.goal = None

    def fly_to(self, yaw, pitch, dist, target=(0, 0, 0)):
        self.goal = (yaw, pitch, dist, np.array(target, dtype=np.float64))

    def update(self, dt):
        if self.goal is None:
            return
        yaw, pitch, dist, tgt = self.goal
        k = 1.0 - math.exp(-dt * 5.0)
        dy = (yaw - self.yaw + math.pi) % TAU - math.pi
        self.yaw += dy * k
        self.pitch += (pitch - self.pitch) * k
        self.dist += (dist - self.dist) * k
        self.target += (tgt - self.target) * k
        if abs(dy) < 1e-3 and abs(pitch - self.pitch) < 1e-3 and abs(dist - self.dist) < 1e-3:
            self.goal = None

    def eye(self):
        cp = math.cos(self.pitch)
        return self.target + self.dist * np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])

    def mvp(self, aspect):
        return perspective(math.radians(45), aspect, 0.02, 200.0) @ look_at(self.eye(), self.target)

    def pan(self, dx, dy):
        f = self.target - self.eye()
        f /= np.linalg.norm(f)
        r = np.cross(f, (0, 1, 0))
        r /= max(np.linalg.norm(r), 1e-9)
        u = np.cross(r, f)
        self.target += (-dx * r + dy * u) * self.dist * 0.0015
        self.goal = None


def write_png(path, w, h, rgb):
    stride = w * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in reversed(range(h)))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def set_uniforms(prog, **kw):
    for k, v in kw.items():
        try:
            u = prog[k]
        except KeyError:
            continue
        if isinstance(v, np.ndarray):
            u.write(v.T.astype("f4").tobytes())
        else:
            u.value = v


def nice_step(span, target=6):
    """1-2-5 step giving about `target` intervals over span"""
    raw = max(span, 1e-12) / max(target, 1)
    e = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * e:
            return m * e
    return 10 * e


def fmt_num(v, step=None):
    if step is not None:
        dec = max(0, -int(math.floor(math.log10(step) + 1e-9)))
        s = f"{v:.{dec}f}"
    else:
        s = f"{v:.6g}" if abs(v) < 1e7 else f"{v:.10g}"
    if s.strip("-0.") == "":
        s = "0"
    return s.replace("-", "−")


def fmt_complex(z):
    a, b = z.real, z.imag
    f = (lambda v: f"{v:.6g}") if max(abs(a), abs(b)) < 1e6 else (lambda v: f"{v:.4e}")
    return f"{f(a)} {'+' if b >= 0 else '−'} {f(abs(b))}i".replace("-", "−")


def find_zero(t, direction):
    """next zero of Hardy's Z (= zero of zeta on the critical line) from t in the given direction"""
    if mpmath is None:
        return None
    mpmath.mp.dps = 20
    step = 0.1 * TAU / math.log(max(abs(t), 30.0) / TAU)
    a = t + direction * step * 0.02
    za = mpmath.siegelz(a)
    for _ in range(5000):
        b = a + direction * step
        zb = mpmath.siegelz(b)
        if za * zb < 0:
            return float(mpmath.findroot(mpmath.siegelz, (min(a, b), max(a, b)), solver="anderson"))
        a, za = b, zb
    return None


# --------------------------------------------------------------------------------------
# text (Pillow glyph atlas, outlined glyphs)
# --------------------------------------------------------------------------------------

class Text:
    CHARS = [chr(c) for c in range(32, 127)] + list("σζγθχπ½−–—·∈≤≥…→←↑↓×√∞°ᵏ")

    def __init__(self, ctx, px):
        from PIL import Image, ImageDraw, ImageFont
        font = None
        for path in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf",
                     "DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
            try:
                font = ImageFont.truetype(path, px)
                break
            except OSError:
                pass
        if font is None:
            font = ImageFont.load_default(size=px)
        asc, desc = font.getmetrics()
        pad = max(2, px // 6)
        self.h = asc + desc
        self.pad = pad
        cell_h = self.h + 2 * pad
        W = 1024
        x, y = 12, 0  # 8x8 solid block at (0,0) for rectangles
        layout = {}
        for c in self.CHARS:
            adv = font.getlength(c)
            w = int(math.ceil(adv)) + 2 * pad
            if x + w > W:
                x, y = 0, y + cell_h + 3
            layout[c] = (x, y, w, adv)
            x += w + 3
        H = y + cell_h
        fill, stroke = Image.new("L", (W, H)), Image.new("L", (W, H))
        df, ds = ImageDraw.Draw(fill), ImageDraw.Draw(stroke)
        df.rectangle((0, 0, 7, 7), fill=255)
        ds.rectangle((0, 0, 7, 7), fill=255)
        sw = max(1, px // 7)
        for c, (gx, gy, w, adv) in layout.items():
            df.text((gx + pad, gy + pad), c, font=font, fill=255)
            ds.text((gx + pad, gy + pad), c, font=font, fill=255, stroke_width=sw, stroke_fill=255)
        data = np.dstack([np.asarray(fill), np.asarray(stroke)]).astype("u1")
        self.tex = ctx.texture((W, H), 2, data.tobytes(), alignment=1)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.cell_h = cell_h
        self.glyphs = {c: (gx / W, gy / H, (gx + w) / W, (gy + cell_h) / H, w, adv) for c, (gx, gy, w, adv) in layout.items()}
        self.solid = (4 / W, 4 / H)
        self.prog = ctx.program(vertex_shader=TEXT_VS, fragment_shader=TEXT_FS)
        self.buf = ctx.buffer(reserve=1 << 22)
        self.vao = ctx.vertex_array(self.prog, [(self.buf, "2f 2f 4f", "in_pos", "in_uv", "in_col")])
        self.ctx = ctx
        self.verts = []

    def release(self):
        for o in (self.vao, self.buf, self.prog, self.tex):
            o.release()

    def width(self, s):
        return sum(self.glyphs.get(c, self.glyphs["?"])[5] for c in s)

    def add(self, x, y, s, col=(1, 1, 1, 1), anchor="lt"):
        """anchor: [l|c|r][t|m|b]"""
        w = self.width(s)
        x -= {"l": 0, "c": w / 2, "r": w}[anchor[0]]
        y -= {"t": 0, "m": self.h / 2, "b": self.h}[anchor[1]]
        x, y = round(x), round(y)
        for c in s:
            u0, v0, u1, v1, gw, adv = self.glyphs.get(c, self.glyphs["?"])
            x0, y0 = x - self.pad, y - self.pad
            x1, y1 = x0 + gw, y0 + self.cell_h
            self.verts += [(x0, y0, u0, v0, *col), (x1, y0, u1, v0, *col), (x0, y1, u0, v1, *col),
                           (x1, y0, u1, v0, *col), (x1, y1, u1, v1, *col), (x0, y1, u0, v1, *col)]
            x += adv
        return w

    def rect(self, x0, y0, x1, y1, col):
        u, v = self.solid
        self.verts += [(x0, y0, u, v, *col), (x1, y0, u, v, *col), (x0, y1, u, v, *col),
                       (x1, y0, u, v, *col), (x1, y1, u, v, *col), (x0, y1, u, v, *col)]

    def flush(self, res):
        if not self.verts:
            return
        data = np.asarray(self.verts, dtype="f4")[: (1 << 22) // 32]
        self.buf.write(data.tobytes())
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.tex.use(0)
        set_uniforms(self.prog, u_atlas=0, u_res=(float(res[0]), float(res[1])))
        self.vao.render(moderngl.TRIANGLES, vertices=len(data))
        self.ctx.disable(moderngl.BLEND)
        self.verts = []


# --------------------------------------------------------------------------------------
# progressive GPU evaluation jobs (a few rows per frame -> the UI never freezes)
# --------------------------------------------------------------------------------------

class Job:
    def __init__(self, ctx, prog, vao, size):
        self.ctx, self.prog, self.vao = ctx, prog, vao
        self.tex = ctx.texture(size, 4, dtype="f4")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = self.tex.repeat_y = False
        self.fbo = ctx.framebuffer(color_attachments=[self.tex])
        self.size = size
        self.row = 0
        self.started = False
        self.uniforms = {}
        self.cost = 1.0
        self.table = None

    def release(self):
        self.fbo.release()
        self.tex.release()

    def start(self, table, cost, **uniforms):
        self.table, self.cost, self.uniforms = table, cost, uniforms
        self.row = 0
        self.started = True

    @property
    def done(self):
        return self.started and self.row >= self.size[1]

    @property
    def progress(self):
        return self.row / self.size[1]

    def step(self, budget, rs_t):
        if not self.started or self.done:
            return 0.0
        w, h = self.size
        rows = int(min(h - self.row, max(1, budget // (w * self.cost))))
        self.fbo.use()
        self.ctx.viewport = (0, self.row, w, rows)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.table.bind(self.prog, rs_t)
        set_uniforms(self.prog, **self.uniforms)
        self.vao.render(moderngl.TRIANGLES, vertices=3)
        self.row += rows
        return rows * w * self.cost


# --------------------------------------------------------------------------------------
# application
# --------------------------------------------------------------------------------------

STYLES = ["colour + brightness", "colour only (arg ζ)", "brightness only (|ζ|)", "lines only"]
METHODS = [("auto", RS_THRESHOLD), ("Euler–Maclaurin", 1e30), ("Riemann–Siegel", 30.0)]
CAM_PRESETS = [("side", 0.45, 0.28, 8.5), ("end-on (complex plane)", -math.pi / 2, 0.0, 6.5),
               ("top", 0.0, 1.45, 8.5), ("front", 0.0, 0.02, 9.0)]
SHOT_DIR = "screenshots"  # where the S key saves PNGs
BUDGET = 3e8           # series terms evaluated per frame


class App:
    MODES = ["complex plane", "landscape", "critical line"]
    RES3D = 512

    def __init__(self, width=1400, height=900, mode=0, shot=None):
        if not glfw.init():
            sys.exit("could not initialise GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 4)
        self.win = glfw.create_window(width, height, "ζ explorer", None, None)
        if not self.win:
            glfw.terminate()
            sys.exit("could not create an OpenGL 4.1 window")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)
        self.ctx = ctx = moderngl.create_context()

        self.p_compute = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=COMPUTE_FS)
        self.p_line = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=LINE_FS)
        self.p_plane = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=PLANE_FS)
        self.p_bg = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=BG_FS)
        self.p_land = ctx.program(vertex_shader=LAND_VS, fragment_shader=LAND_FS)
        self.p_spiral = ctx.program(vertex_shader=SPIRAL_VS, fragment_shader=SPIRAL_FS)
        self.p_lines = ctx.program(vertex_shader=LINES_VS, fragment_shader=LINES_FS)
        self.p_cbar = ctx.program(vertex_shader=CBAR_VS, fragment_shader=CBAR_FS)
        self.vao_cbar = ctx.vertex_array(self.p_cbar, [])
        self.bars = []
        self.vao_compute = ctx.vertex_array(self.p_compute, [])
        self.vao_line = ctx.vertex_array(self.p_line, [])
        self.vao_plane = ctx.vertex_array(self.p_plane, [])
        self.vao_bg = ctx.vertex_array(self.p_bg, [])
        self.vao_spiral = ctx.vertex_array(self.p_spiral, [])

        R = self.RES3D
        i, j = np.meshgrid(np.arange(R - 1), np.arange(R - 1))
        a = (j * R + i).ravel()
        idx = np.stack([a, a + 1, a + R, a + 1, a + R + 1, a + R], axis=1).astype("u4")
        self.n_land_idx = idx.size
        self.vao_land = ctx.vertex_array(self.p_land, [], index_buffer=ctx.buffer(idx.tobytes()), index_element_size=4)
        self.lines_buf = ctx.buffer(reserve=1 << 20)
        self.vao_lines = ctx.vertex_array(self.p_lines, [(self.lines_buf, "3f 4f", "in_pos", "in_col")])

        self.table = PhaseTable(ctx, 1)      # plane + landscape
        self.sp_table = PhaseTable(ctx, 2)   # critical-line view
        self.jobs = {}
        self.land_job = self.make_job("land", self.p_compute, self.vao_compute, (R, R))
        self.land_job.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.text = None
        self.text_px = 0

        self.mode = mode
        self.shot = shot
        self.animate = False
        self.contours = 1
        self.grid = 1
        self.labels = 0
        self.info = 0
        self.style = 1          # 0 hue+brightness, 1 hue only, 2 brightness only, 3 lines only
        self.colorbar = 0
        self.markers = 1        # dark marker at each zero
        self.help = False
        self.scheme = SCHEME_NAMES.index(DEFAULT_SCHEME)
        self.method = 0
        self.hmode = 0
        self.hscale = 0.06
        self.input = None
        self.message, self.message_t = "", 0.0
        self.cam_land = OrbitCam(0.0, 0.62, 3.6, (0, 0.25, 0))
        self.cam_spiral = OrbitCam(0.45, 0.28, 8.5, (0, 0, 0))
        self.cam_preset = 0
        self.sp_n, self.sp_t0, self.sp_auto = 1, 0.0, 1.0
        self.reset_region()
        self.reset_spiral()
        self.fit_land_camera()
        self.drag = None
        self.last_mouse = (0, 0)
        self.t_start = time.time()
        self.flow = 0.0
        self.anim_t = 0.0
        self.want_shot = False
        self.frames, self.fps, self.fps_t = 0, 0.0, time.time()
        self.readout, self.readout_t = "", 0.0

        if not shot:     # screenshot runs ignore input, so a stray scroll or key press can't change the picture
            glfw.set_key_callback(self.win, self.on_key)
            glfw.set_char_callback(self.win, self.on_char)
            glfw.set_mouse_button_callback(self.win, self.on_button)
            glfw.set_cursor_pos_callback(self.win, self.on_cursor)
            glfw.set_scroll_callback(self.win, self.on_scroll)
        glfw.set_framebuffer_size_callback(self.win, lambda *a: self.region_changed())

    # ---------------------------------------------------------------- state
    @property
    def plain(self):
        return int(self.style == 3)

    @property
    def rs_t(self):
        return METHODS[self.method][1]

    def make_job(self, name, prog, vao, size):
        old = self.jobs.get(name)
        if old is not None and old.size == size:
            return old
        if old is not None:
            old.release()
        job = Job(self.ctx, prog, vao, size)
        self.jobs[name] = job
        return job

    def say(self, msg):
        self.message, self.message_t = msg, time.time()
        print(msg)

    def reset_region(self):
        self.center = np.array([0.5, 0.0])     # t symmetric around 0 (zeros come in conjugate pairs)
        self.hy = 26.0                          # shows the first three zero pairs, t = ±14.13, ±21.02, ±25.01
        self.stretch = 0.4      # default: sigma axis squeezed 2.5x (1.0 would be 1:1)
        self.aspect_mode = 1    # A cycles: 0 1:1, 1 sigma squeezed, 2 t stretched
        self.region_changed()

    def reset_spiral(self):
        self.sp_c, self.sp_w = 30.0, 60.0
        self.sp_sigma = 0.5
        self.sp_amp = 0.45
        self.sp_speed = 1.0
        self.unwind, self.unwind_cur = 0.0, 0.0
        self.sp_zeros = []
        self.spiral_changed()

    def fit_land_camera(self):
        """aim the landscape camera at the centre of the surface's bounding box, far enough to see all of it"""
        hf = self.half()
        ar = hf[0] / hf[1]                                  # surface spans x in [-ar, ar], z in [-1, 1]
        top = 6.0 * self.hscale                             # height cap of the |zeta| surface
        w, h = glfw.get_framebuffer_size(self.win)
        aspect = w / max(h, 1)
        half_v = math.radians(45) / 2
        half_h = math.atan(math.tan(half_v) * aspect)
        r = math.sqrt(ar * ar + 1.0 + (top / 2) ** 2)       # bounding-sphere radius
        dist = 0.44 * r / math.sin(min(half_v, half_h))       # half the distance that fits it all in view
        # seen from the sigma > 0 side, slightly from below the t range: t runs left -> right across the
        # screen, the left half-plane rises behind the critical line
        yaw, pitch, zoom = getattr(self, "land_view", None) or (1.0, 0.6, 1.0)
        focus_x = (-2.0 - self.center[0]) / self.hy           # look at sigma = -2 (world x = (sigma - centre) / hy)
        self.cam_land.init = (yaw, pitch, dist / zoom, (focus_x, top / 2, 0.0))
        self.cam_land.reset()

    def region_changed(self):
        self.region_dirty = True
        self.changed_at = time.time()

    def spiral_changed(self):
        self.spiral_dirty = True

    def half(self):
        w, h = glfw.get_framebuffer_size(self.win)
        return np.array([self.hy * self.stretch * w / max(h, 1), self.hy])

    def zoom_region(self, f, anchor=None):
        if anchor is None:
            anchor = self.center.copy()
        self.center = anchor + (self.center - anchor) * f
        self.hy = float(np.clip(self.hy * f, 1e-9, 1e9))
        self.region_changed()

    def cursor_ndc(self):
        x, y = glfw.get_cursor_pos(self.win)
        w, h = glfw.get_window_size(self.win)
        return np.array([2 * x / max(w, 1) - 1, 1 - 2 * y / max(h, 1)])

    def cursor_s(self):
        return self.center + self.cursor_ndc() * self.half()

    def grid_info(self):
        hf = self.half()
        w, h = glfw.get_framebuffer_size(self.win)
        step = np.array([nice_step(2 * hf[0], max(2, 7 * w / max(h, 1) / 1.6)), nice_step(2 * hf[1], 7)])
        origin_t = round(self.center[1] / step[1]) * step[1]
        return step, origin_t

    # ---------------------------------------------------------------- job scheduling
    def plan_region(self, fb):
        """(re)start the plane / landscape jobs after the region changed"""
        hf = self.half()
        sig_max = abs(self.center[0]) + hf[0]
        t_max = abs(self.center[1]) + hf[1]
        self.table.update(self.center[1], hf[1], terms_needed(sig_max, t_max, self.rs_t))
        cost = cost_per_sample(sig_max, t_max, self.rs_t)
        w, h = fb
        sc = float(np.clip(math.sqrt(BUDGET * 1.5 / (w * h * cost)), 0.12, 0.45))
        lo = self.make_job("lo", self.p_compute, self.vao_compute, (max(2, int(w * sc)), max(2, int(h * sc))))
        hi = self.make_job("hi", self.p_compute, self.vao_compute, (w, h))
        common = dict(u_sig=(float(self.center[0]), float(hf[0])),
                      u_tt=(float(self.table.rel(self.center[1])), float(hf[1])), u_flip=self.table.flip)
        lo.start(self.table, cost, u_off=(0.0, 0.0), u_den=tuple(map(float, lo.size)), **common)
        hi.start(self.table, cost, u_off=(0.0, 0.0), u_den=(float(w), float(h)), **common)
        R = self.RES3D
        self.land_job.start(self.table, cost, u_off=(0.5, 0.5), u_den=(R - 1.0, R - 1.0), **common)
        self.region_dirty = False

    def plan_spiral(self):
        t0, w = self.sp_c - self.sp_w / 2, self.sp_w
        t_max = max(abs(t0), abs(t0 + w))
        self.sp_table.update(self.sp_c, w / 2, terms_needed(abs(self.sp_sigma), t_max, self.rs_t))
        n = int(np.clip(w * 150 * math.log(max(t_max / TAU, 3.0)), 4000, 400000))
        rows = (n + 4095) // 4096
        job = self.make_job("line", self.p_line, self.vao_line, (4096, rows))
        job.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        job.start(self.sp_table, cost_per_sample(abs(self.sp_sigma), t_max, self.rs_t),
                  u_dlo=float(self.sp_table.rel(t0)), u_w=float(w), u_sigma=float(self.sp_sigma), u_n=n,
                  u_flip=self.sp_table.flip)
        self.sp_n, self.sp_t0 = n, t0
        self.sp_zeros = []
        self.spiral_dirty = False

    def find_spiral_zeros(self):
        """sign changes of Hardy's Z(t) = Re(e^{i theta} zeta(1/2+it)) on the computed samples;
        also picks an automatic amplitude scale so large-t traces stay on screen"""
        job = self.jobs["line"]
        d = np.frombuffer(job.tex.read(), dtype="f4").reshape(-1, 4)[: self.sp_n].astype(np.float64)
        mag = np.hypot(d[:, 0], d[:, 1])
        mag = mag[np.isfinite(mag)]
        self.sp_auto = 2.2 / max(np.percentile(mag, 99.5), 2.2) if mag.size else 1.0
        if abs(self.sp_sigma - 0.5) > 1e-9:
            self.sp_zeros = []
            return
        Z = d[:, 0] * np.cos(d[:, 2]) - d[:, 1] * np.sin(d[:, 2])
        i = np.nonzero(np.sign(Z[:-1]) * np.sign(Z[1:]) < 0)[0]
        dt = self.sp_w / (self.sp_n - 1)
        self.sp_zeros = list(self.sp_t0 + (i + Z[i] / (Z[i] - Z[i + 1])) * dt)

    def run_jobs(self, now):
        if self.mode in (0, 1):
            if self.region_dirty:
                self.plan_region(glfw.get_framebuffer_size(self.win))
            if self.mode == 0:
                lo, hi = self.jobs["lo"], self.jobs["hi"]
                used = lo.step(BUDGET * 2, self.rs_t)
                if lo.done and (now - self.changed_at > 0.15 or self.shot) and used < BUDGET:
                    hi.step(BUDGET - used, self.rs_t)
            else:
                self.land_job.step(BUDGET * 1.5, self.rs_t)
        else:
            if self.spiral_dirty:
                self.plan_spiral()
            job = self.jobs["line"]
            if not job.done:
                job.step(BUDGET * 1.5, self.rs_t)
                if job.done:
                    self.find_spiral_zeros()

    def busy(self):
        if self.mode == 0:
            job = self.jobs.get("hi")
        else:
            job = self.land_job if self.mode == 1 else self.jobs.get("line")
        return 0 if job is None or job.done else max(job.progress, 1e-3)

    # ---------------------------------------------------------------- input
    def on_char(self, win, cp):
        c = chr(cp)
        if self.input is not None:
            if c in "0123456789.eE+-":
                self.input += c
        elif c == "?":
            self.help = not self.help

    def goto(self, txt):
        try:
            v = float(txt)
        except ValueError:
            self.say(f"not a number: {txt!r}")
            return
        if self.mode == 2:
            self.sp_c = v + self.sp_w / 2
            self.spiral_changed()
        else:
            self.center[1] = v
            self.hy = min(self.hy, 20.0)
            self.region_changed()
        self.say(f"t = {fmt_num(v)}")

    def on_key(self, win, key, sc, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        K = glfw
        shift = bool(mods & glfw.MOD_SHIFT)
        # '?' arrives through on_char; on many layouts it is shift + the '-', '/' or apostrophe key,
        # so don't also run that key's own command
        if shift and self.input is None and key in (K.KEY_MINUS, K.KEY_SLASH, K.KEY_APOSTROPHE):
            return
        if self.input is not None:
            if key in (K.KEY_ENTER, K.KEY_KP_ENTER):
                txt, self.input = self.input, None
                if txt:
                    self.goto(txt)
            elif key == K.KEY_ESCAPE:
                self.input = None
            elif key == K.KEY_BACKSPACE:
                self.input = self.input[:-1]
            return
        spiral = self.mode == 2
        if key in (K.KEY_ESCAPE, K.KEY_Q):
            if self.help:
                self.help = False
            else:
                glfw.set_window_should_close(win, True)
        elif key in (K.KEY_1, K.KEY_2, K.KEY_3):
            self.mode = key - K.KEY_1
        elif key == K.KEY_TAB:
            self.mode = (self.mode + (2 if shift else 1)) % 3
        elif key == K.KEY_F:
            self.toggle_fullscreen()
        elif key == K.KEY_F1:
            self.help = not self.help
        elif key == K.KEY_SPACE:
            self.animate = not self.animate
        elif key == K.KEY_K:
            self.scheme = (self.scheme + (-1 if shift else 1)) % len(SCHEME_NAMES)
            if self.style >= 2:
                self.style = 1
            self.say(f"colour map {self.scheme + 1}/{len(SCHEME_NAMES)}: {SCHEME_NAMES[self.scheme]}")
        elif key == K.KEY_X:
            self.style = (self.style + (-1 if shift else 1)) % len(STYLES)
            self.say(f"display: {STYLES[self.style]}")
        elif key == K.KEY_B:
            self.colorbar ^= 1
        elif key == K.KEY_Z:
            self.markers ^= 1
            self.say("zero markers on" if self.markers else "zero markers off: zeros are where all colours meet")
        elif key == K.KEY_L:
            self.labels ^= 1
        elif key == K.KEY_I:
            self.info ^= 1
        elif key == K.KEY_C:
            self.contours ^= 1
        elif key == K.KEY_G:
            self.grid ^= 1
        elif key == K.KEY_T:
            self.input = ""
        elif key == K.KEY_M:
            self.method = (self.method + 1) % len(METHODS)
            self.say(f"method: {METHODS[self.method][0]}")
            self.region_changed()
            self.spiral_changed()
        elif key == K.KEY_S:
            self.want_shot = True
        elif key == K.KEY_R:
            if spiral:
                self.reset_spiral()
                self.cam_spiral.reset()
                self.cam_preset = 0
            else:
                self.reset_region()
                self.fit_land_camera()
        elif spiral:
            self.on_key_spiral(key, shift)
        else:
            self.on_key_region(key, shift)

    def toggle_fullscreen(self):
        if sys.platform == "darwin":                    # macOS: native fullscreen, same as the green button
            try:
                import ctypes
                import ctypes.util
                objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
                objc.sel_registerName.restype = ctypes.c_void_p
                objc.sel_registerName.argtypes = [ctypes.c_char_p]
                objc.objc_msgSend.restype = ctypes.c_void_p
                objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                objc.objc_msgSend(glfw.get_cocoa_window(self.win), objc.sel_registerName(b"toggleFullScreen:"), None)
                return
            except Exception as e:                      # fall back to GLFW's monitor fullscreen
                print("native fullscreen unavailable:", e)
        if glfw.get_window_monitor(self.win):          # back to the window we had before
            x, y, w, h = self.windowed
            glfw.set_window_monitor(self.win, None, x, y, w, h, 0)
        else:
            self.windowed = (*glfw.get_window_pos(self.win), *glfw.get_window_size(self.win))
            mon = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(mon)
            glfw.set_window_monitor(self.win, mon, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
        self.region_changed()

    def on_key_region(self, key, shift):
        K = glfw
        if key == K.KEY_A:
            # cycle: 1:1  ->  sigma squeezed 2.5x (same t range)  ->  t stretched 2.5x (same sigma range)  ->  1:1
            self.aspect_mode = (self.aspect_mode + 1) % 3
            if self.aspect_mode == 1:
                self.stretch = 0.4
            elif self.aspect_mode == 2:
                self.hy *= 2.5
            else:
                self.hy /= 2.5
                self.stretch = 1.0
            if self.mode == 0:
                self.say(["aspect 1:1", "σ axis squeezed 2.5× (same t range)",
                          "t range stretched 2.5× (same σ range)"][self.aspect_mode])
            else:       # the landscape always draws sigma and t to the same scale: only the ranges change
                hf = self.half()
                self.say(f"σ ∈ [{fmt_num(self.center[0] - hf[0])}, {fmt_num(self.center[0] + hf[0])}]   "
                         f"t ∈ [{fmt_num(self.center[1] - hf[1])}, {fmt_num(self.center[1] + hf[1])}]")
            self.region_changed()                   # the camera stays where it is
        elif key == K.KEY_H:
            self.hmode ^= 1
        elif key == K.KEY_EQUAL:
            self.hscale *= 1.2
        elif key == K.KEY_MINUS:
            self.hscale /= 1.2
        elif key in (K.KEY_N, K.KEY_P):
            t = find_zero(self.center[1], 1 if key == K.KEY_N else -1)
            if t is None:
                self.say("install mpmath to jump between zeros")
                return
            self.center = np.array([0.5, t])        # keep the current zoom
            self.region_changed()
            self.say(f"zero at s = 1/2 + {t:.10g} i")
        elif key in (K.KEY_LEFT_BRACKET, K.KEY_RIGHT_BRACKET):
            self.zoom_region(1.25 if key == K.KEY_LEFT_BRACKET else 0.8)
        elif key in (K.KEY_LEFT, K.KEY_RIGHT, K.KEY_UP, K.KEY_DOWN):
            d = {K.KEY_LEFT: (-1, 0), K.KEY_RIGHT: (1, 0), K.KEY_UP: (0, 1), K.KEY_DOWN: (0, -1)}[key]
            self.center += np.array(d) * self.half() * (0.5 if shift else 0.1)
            self.region_changed()

    def on_key_spiral(self, key, shift):
        K = glfw
        if key in (K.KEY_LEFT, K.KEY_RIGHT):
            self.sp_c += (1 if key == K.KEY_RIGHT else -1) * self.sp_w * (0.5 if shift else 0.1)
            self.spiral_changed()
        elif key in (K.KEY_LEFT_BRACKET, K.KEY_RIGHT_BRACKET):
            f = 1.25 if key == K.KEY_LEFT_BRACKET else 0.8
            self.sp_w = float(np.clip(self.sp_w * f, 0.5, 5000))
            self.spiral_changed()
        elif key in (K.KEY_UP, K.KEY_DOWN):
            self.sp_sigma = round(self.sp_sigma + (1 if key == K.KEY_UP else -1) * (0.1 if shift else 0.02), 4)
            self.spiral_changed()
        elif key == K.KEY_0:
            self.sp_sigma = 0.5
            self.spiral_changed()
        elif key == K.KEY_EQUAL:
            self.sp_amp *= 1.2
        elif key == K.KEY_MINUS:
            self.sp_amp /= 1.2
        elif key == K.KEY_PERIOD:
            self.sp_speed = min(self.sp_speed * 1.5, 20.0)
        elif key == K.KEY_COMMA:
            self.sp_speed = max(self.sp_speed / 1.5, 0.05)
        elif key == K.KEY_U:
            self.unwind = 1.0 - self.unwind
            self.say("unwinding: e^{iθ(t)}·ζ(1/2+it) = Z(t), Hardy's real function" if self.unwind else "ζ(σ+it)")
        elif key == K.KEY_V:
            self.cam_preset = (self.cam_preset + 1) % len(CAM_PRESETS)
            name, yaw, pitch, dist = CAM_PRESETS[self.cam_preset]
            self.cam_spiral.fly_to(yaw, pitch, dist)
            self.say(f"camera: {name}")

    def on_button(self, win, button, action, mods):
        if action == glfw.PRESS:
            pan = button == glfw.MOUSE_BUTTON_RIGHT or (mods & glfw.MOD_SHIFT)
            self.drag = "pan" if pan else "rotate"
            self.last_mouse = glfw.get_cursor_pos(win)
        else:
            self.drag = None

    def on_cursor(self, win, x, y):
        if self.drag is None:
            return
        dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
        self.last_mouse = (x, y)
        if self.mode == 0:
            w, h = glfw.get_window_size(win)
            hf = self.half()
            self.center -= np.array([dx * 2 * hf[0] / w, -dy * 2 * hf[1] / h])
            self.region_changed()
            return
        cam = self.cam_land if self.mode == 1 else self.cam_spiral
        if self.drag == "rotate":
            cam.goal = None
            cam.yaw -= dx * 0.008
            cam.pitch = float(np.clip(cam.pitch + dy * 0.008, -1.55, 1.55))
        else:
            cam.pan(dx, dy)

    def on_scroll(self, win, sx, sy):
        f = math.exp(-0.12 * sy)
        shift = glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        if self.mode == 0:
            self.zoom_region(f, self.cursor_s())
        elif self.mode == 1 and shift:
            self.zoom_region(f)
        else:
            cam = self.cam_land if self.mode == 1 else self.cam_spiral
            cam.goal = None
            cam.dist = float(np.clip(cam.dist * f, 0.2, 80))

    # ---------------------------------------------------------------- drawing helpers
    def draw_lines(self, verts, mvp, mode=moderngl.LINES):
        if not verts:
            return
        data = np.asarray(verts, dtype="f4")
        self.lines_buf.write(data.tobytes())
        set_uniforms(self.p_lines, u_mvp=mvp)
        self.vao_lines.render(mode, vertices=len(data))

    @staticmethod
    def project(mvp, p, fb):
        c = mvp @ np.array([p[0], p[1], p[2], 1.0])
        if c[3] <= 1e-6:
            return None
        n = c[:3] / c[3]
        if abs(n[0]) > 1.1 or abs(n[1]) > 1.1 or abs(n[2]) > 1:
            return None
        return ((n[0] + 1) / 2 * fb[0], (1 - n[1]) / 2 * fb[1])

    def label3d(self, mvp, p, s, fb, col=(0.85, 0.9, 1.0, 0.95), anchor="cm", off=(0, 0)):
        q = self.project(mvp, p, fb)
        if q is not None:
            self.text.add(q[0] + off[0], q[1] + off[1], s, col, anchor)

    def common_shade(self, now):
        return dict(u_time=now - self.t_start, u_flow=self.flow, u_contours=self.contours,
                    u_grid=self.grid, u_scheme=self.scheme, u_plain=self.plain, u_style=self.style, u_markers=self.markers)

    # ---------------------------------------------------------------- views
    def render_plane(self, fb, now):
        w, h = fb
        lo, hi = self.jobs["lo"], self.jobs["hi"]
        tex = (hi if hi.done else lo).tex
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, w, h)
        tex.use(0)
        step, origin_t = self.grid_info()
        set_uniforms(self.p_plane, u_tex=0, u_crel=(float(self.center[0]), float(self.center[1] - origin_t)),
                     u_half=tuple(map(float, self.half())), u_res=(float(w), float(h)), u_step=tuple(map(float, step)),
                     u_axis_t=float(-origin_t), **self.common_shade(now))
        self.vao_plane.render(moderngl.TRIANGLES, vertices=3)
        if self.labels:
            self.plane_labels(fb, step)

    def plane_labels(self, fb, step):
        w, h = fb
        hf = self.half()
        T, m = self.text, self.text.h * 0.5
        col = (0.95, 0.97, 1.0, 0.9)
        lo_t = self.center[1] - hf[1]
        for k in range(math.ceil(lo_t / step[1]), math.floor((self.center[1] + hf[1]) / step[1]) + 1):
            t = k * step[1]
            y = (1 - (t - lo_t) / (2 * hf[1])) * h
            if 2.5 * T.h < y < h - 2.5 * T.h:
                T.add(w - m, y, fmt_num(t, step[1]), col, "rm")
        lo_s = self.center[0] - hf[0]
        for k in range(math.ceil(lo_s / step[0]), math.floor((self.center[0] + hf[0]) / step[0]) + 1):
            sg = k * step[0]
            x = (sg - lo_s) / (2 * hf[0]) * w
            if 3 * m < x < w - 8 * m:
                T.add(x, h - m, fmt_num(sg, step[0]), col, "cb")
        T.add(w - m, h - m, "σ →", (1, 1, 0.8, 1), "rb")
        T.add(w - m, m, "↑ t", (1, 1, 0.8, 1), "rt")
        x = (0.5 - lo_s) / (2 * hf[0]) * w
        if 0 < x < w:
            T.add(x + m, h - 3.2 * T.h, "σ = ½", (0.5, 0.9, 1.0, 1), "lb")

    def render_landscape(self, fb, now):
        R = self.RES3D
        w, h = fb
        ctx = self.ctx
        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)
        ctx.disable(moderngl.DEPTH_TEST)
        self.vao_bg.render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)
        mvp = self.cam_land.mvp(w / h)
        hf = self.half()
        step, origin_t = self.grid_info()
        self.land_job.tex.use(0)
        set_uniforms(self.p_land, u_tex=0, u_res=R, u_hmode=self.hmode,
                     u_crel=(float(self.center[0]), float(self.center[1] - origin_t)),
                     u_half=tuple(map(float, hf)), u_mvp=mvp, u_hscale=self.hscale, u_eye=tuple(self.cam_land.eye()),
                     u_step=tuple(map(float, step)), u_axis_t=float(-origin_t), **self.common_shade(now))
        self.vao_land.render(moderngl.TRIANGLES, vertices=self.n_land_idx)
        # frame, ticks and a height axis
        ar = hf[0] / hf[1]
        c = (0.5, 0.7, 1.0, 0.55)
        corners = [(-ar, 0, -1), (ar, 0, -1), (ar, 0, 1), (-ar, 0, 1)]
        verts = []
        for a, b in zip(corners, corners[1:] + corners[:1]):
            verts += [(*a, *c), (*b, *c)]
        sx = lambda sg: (sg - self.center[0]) / hf[1]
        sz = lambda t: -(t - self.center[1]) / hf[1]
        sig_ticks = [k * step[0] for k in range(math.ceil((self.center[0] - hf[0]) / step[0]),
                                                 math.floor((self.center[0] + hf[0]) / step[0]) + 1)]
        t_ticks = [k * step[1] for k in range(math.ceil((self.center[1] - hf[1]) / step[1]),
                                               math.floor((self.center[1] + hf[1]) / step[1]) + 1)]
        tk = 0.04
        for sg in sig_ticks:
            verts += [(sx(sg), 0, 1, *c), (sx(sg), 0, 1 + tk, *c)]
        for t in t_ticks:
            verts += [(-ar, 0, sz(t), *c), (-ar - tk, 0, sz(t), *c)]
        if self.hmode == 0:
            hval, h_ticks, top_v = (lambda m: 6 * (1 - math.exp(-m / 6))), [0, 1, 2, 3, 5, 10], 1e9
        else:
            hval, h_ticks, top_v = (lambda lm: 4 * math.tanh(lm / 4)), [-6, -3, -1, 0, 1, 3, 6], 20
        top = hval(top_v) * self.hscale
        bot = 0 if self.hmode == 0 else -top
        verts += [(-ar, bot, 1, *c), (-ar, top, 1, *c)]
        for v in h_ticks:
            y = hval(v) * self.hscale
            verts += [(-ar, y, 1, *c), (-ar - tk, y, 1, *c)]
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.draw_lines(verts, mvp)
        ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
        if self.labels:
            for sg in sig_ticks:
                self.label3d(mvp, (sx(sg), 0, 1 + 3 * tk), fmt_num(sg, step[0]), fb)
            for t in t_ticks:
                self.label3d(mvp, (-ar - 2.5 * tk, 0, sz(t)), fmt_num(t, step[1]), fb, anchor="rm")
            for v in h_ticks:
                self.label3d(mvp, (-ar - 2 * tk, hval(v) * self.hscale, 1), fmt_num(v), fb, anchor="rm")
            hc = (1, 1, 0.8, 1)
            self.label3d(mvp, (0, 0, 1.3), "σ", fb, hc)
            self.label3d(mvp, (-ar - 0.12, 0, -1.15), "t ↑", fb, hc)
            self.label3d(mvp, (-ar, top, 1), "|ζ|" if self.hmode == 0 else "log|ζ|", fb, hc, "cb", (0, -4))
            if abs(sx(0.5)) <= ar:
                self.label3d(mvp, (sx(0.5), 0, -1.08), "σ = ½", fb, (0.5, 0.9, 1.0, 1))

    def render_spiral(self, fb, now, dt):
        w, h = fb
        ctx = self.ctx
        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)
        ctx.disable(moderngl.DEPTH_TEST)
        self.vao_bg.render(moderngl.TRIANGLES, vertices=3)
        if self.animate:
            self.anim_t += dt * self.sp_speed
            if self.cam_spiral.goal is None and self.drag is None:
                self.cam_spiral.yaw += 0.0015 * self.sp_speed
        self.unwind_cur += (self.unwind - self.unwind_cur) * (1 - math.exp(-dt * 2.5))
        L, k = 9.0, self.sp_amp * self.sp_auto
        mvp = self.cam_spiral.mvp(w / h)
        t0, W = self.sp_t0, self.sp_w
        X = lambda t: ((t - t0) / W - 0.5) * L
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ac = (0.6, 0.7, 1.0, 0.6)
        verts = [(-L / 2, 0, 0, *ac), (L / 2, 0, 0, *ac)]
        step = nice_step(W, 8)
        ticks = [j * step for j in range(math.ceil(t0 / step), math.floor((t0 + W) / step) + 1)]
        for t in ticks:
            verts += [(X(t), -0.06, 0, *ac), (X(t), 0.06, 0, *ac)]
        x0 = -L / 2 - 0.3
        verts += [(x0, 0, -k * 2, *ac), (x0, 0, k * 2, *ac), (x0, -k * 2, 0, *ac), (x0, k * 2, 0, *ac)]
        if self.unwind_cur < 0.5:   # unit circle |zeta| = 1, for scale
            cc = (0.6, 0.7, 1.0, 0.3 * (1 - 2 * self.unwind_cur))
            for i in range(64):
                a0, a1 = i / 64 * TAU, (i + 1) / 64 * TAU
                verts += [(x0, k * math.sin(a0), k * math.cos(a0), *cc), (x0, k * math.sin(a1), k * math.cos(a1), *cc)]
        self.draw_lines(verts, mvp)
        job = self.jobs.get("line")
        if job is not None and job.started:
            n = self.sp_n
            head = 1.2 * ((self.anim_t / (8.0 + W / 20.0)) % 1.0)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            job.tex.use(0)
            common = dict(u_line=0, u_head=head, u_len=L, u_k=k, u_n=n, u_done=min(n, job.row * 4096),
                          u_anim=int(self.animate), u_mvp=mvp, u_psize=3.0 * h / 900.0, u_unwind=self.unwind_cur,
                          u_scheme=self.scheme, u_plain=int(self.style >= 2))
            set_uniforms(self.p_spiral, u_gain=0.9, u_points=0, **common)
            self.vao_spiral.render(moderngl.LINE_STRIP, vertices=n)
            set_uniforms(self.p_spiral, u_gain=0.3 * (20000 / n) ** 0.5, u_points=1, **common)
            self.vao_spiral.render(moderngl.POINTS, vertices=n)
        ctx.disable(moderngl.BLEND)
        if self.labels:
            # skip t labels when the t axis is seen nearly end-on (they would all pile up)
            qa, qb = self.project(mvp, (-L / 2, 0, 0), fb), self.project(mvp, (L / 2, 0, 0), fb)
            t_visible = qa is None or qb is None or math.dist(qa, qb) > 0.25 * fb[0]
            for t in (ticks if t_visible else []):
                self.label3d(mvp, (X(t), -0.14, 0), fmt_num(t, step), fb, anchor="ct")
            hc = (1, 1, 0.8, 1)
            self.label3d(mvp, (x0, 0, k * 2.3), "Z(t)" if self.unwind_cur > 0.5 else "Re ζ", fb, hc)
            self.label3d(mvp, (x0, k * 2.3, 0), "Im ζ", fb, hc)
            self.label3d(mvp, (L / 2 + 0.3, 0, 0), "t", fb, hc)
            if t_visible and 0 < len(self.sp_zeros) <= 400:
                dec = int(np.clip(7 - math.log10(max(abs(t0) + W, 10)), 2, 5))
                pts = []
                for z in self.sp_zeros:
                    q = self.project(mvp, (X(z), 0.2, 0), fb)
                    if q is not None:
                        pts.append((q, f"{z:.{dec}f}"))
                pts.sort(key=lambda e: e[0][0])
                gap, lh = self.text.h * 0.6, self.text.h * 1.15
                last = [-1e9] * 3            # up to 3 staggered rows; skip labels that would overlap
                for (qx, qy), s in pts:
                    wd = self.text.width(s)
                    for r in range(3):
                        if qx - wd / 2 > last[r] + gap:
                            last[r] = qx + wd / 2
                            self.text.add(qx, qy - r * lh, s, (1.0, 0.95, 0.75, 0.95), "cb")
                            break

    # ---------------------------------------------------------------- HUD
    def hud(self, fb, now):
        T = self.text
        lh = T.h * 1.35
        m = T.h * 0.6
        lines = []
        if self.info:
            meth = METHODS[self.method][0]
            if self.method == 0:
                meth += f" (Riemann–Siegel for t ≥ {RS_THRESHOLD:g})"
            sch = STYLES[self.style] if self.style >= 2 else f"{SCHEME_NAMES[self.scheme]}, {STYLES[self.style]}"
            lines.append((f"ζ explorer · {self.MODES[self.mode]} · {sch} · {meth} · {self.fps:.0f} fps",
                          (0.75, 0.85, 1.0, 1)))
            if self.mode == 2:
                t0 = self.sp_t0
                s = f"ζ({fmt_num(self.sp_sigma)} + it)   t ∈ [{fmt_num(t0)}, {fmt_num(t0 + self.sp_w)}]"
                if abs(self.sp_sigma - 0.5) < 1e-9 and self.jobs["line"].done:
                    s += f"   {len(self.sp_zeros)} zeros"
                s += f"   speed {self.sp_speed:.2g}×   amplitude {self.sp_amp / 0.45:.2g}× (auto {self.sp_auto:.2g})"
                if self.unwind_cur > 0.02:
                    s += f"   unwound {self.unwind_cur:.0%}"
                lines.append((s, (1, 1, 1, 1)))
            else:
                hf = self.half()
                lines.append((f"σ ∈ [{fmt_num(self.center[0] - hf[0])}, {fmt_num(self.center[0] + hf[0])}]   "
                              f"t ∈ [{fmt_num(self.center[1] - hf[1])}, {fmt_num(self.center[1] + hf[1])}]", (1, 1, 1, 1)))
                if self.mode == 0:
                    if now - self.readout_t > 0.1:
                        self.readout_t = now
                        self.readout = self.cursor_readout()
                    lines.append((self.readout, (1, 1, 0.85, 1)))
            p = self.busy()
            if p:
                lines.append((f"computing … {p:.0%}", (1.0, 0.75, 0.4, 1)))
            lines.append(("? or F1 : help", (0.6, 0.65, 0.8, 1)))
        if self.input is not None:
            lines.append((f"go to t = {self.input}_      (Enter / Esc)", (0.4, 1.0, 0.6, 1)))
        if self.message and now - self.message_t < 3.0:
            lines.append((self.message, (0.4, 1.0, 0.6, min(1.0, 3.0 - (now - self.message_t)))))
        if lines:
            wmax = max(T.width(s) for s, _ in lines)
            T.rect(m * 0.5, m * 0.5, m * 1.5 + wmax, m * 1.2 + lh * len(lines), (0.02, 0.02, 0.06, 0.55))
            for i, (s, c) in enumerate(lines):
                T.add(m, m + i * lh, s, c)
        if self.colorbar:
            self.colorbar_panel(fb)
        if self.help:
            self.help_panel(fb)

    def colorbar_panel(self, fb):
        """legend explaining what colour / brightness / lines encode (top-right)"""
        T = self.text
        W, H = fb
        lh, m = T.h * 1.3, T.h * 0.6
        bw, bh = T.h * 20, T.h * 0.9
        items = []          # ("bar", kind, lo, hi, title, [(f, label)]) | ("swatch", rgba, label, shape)
        if self.mode == 2:
            if self.style >= 2:
                items.append(("swatch", (0.8, 0.8, 0.8, 1), "trace of ζ(σ + it)", "line"))
            else:
                t0, t1 = self.sp_t0, self.sp_t0 + self.sp_w
                st = (t1 - t0) / 3
                items.append(("bar", 2, 0, 1, "colour = position along t (3 cycles)",
                              [(k / 3, fmt_num(t0 + k * st, nice_step(st, 10))) for k in range(4)]))
            items.append(("swatch", (1, 0.95, 0.8, 1), "zero of ζ: curve meets the t axis", "dot"))
        else:
            if self.plain:
                if self.contours:
                    items.append(("swatch", (0.05, 0.05, 0.05, 1), "phase lines: arg ζ = k·30°", "line"))
                    items.append(("swatch", (0.5, 0.5, 0.5, 1), "|ζ| contours: |ζ| = 2ᵏ", "line"))
            else:
                if self.style in (0, 1):
                    items.append(("bar", 0, 0, 1, f"hue = arg ζ(s)   ·   {SCHEME_NAMES[self.scheme]}  (K)",
                                  [(0, "−π"), (0.25, "−π/2"), (0.5, "0"), (0.75, "π/2"), (1, "π")]))
                if self.style in (0, 2):
                    labels = {-3: "1/8", -2: "1/4", -1: "1/2", 0: "1", 1: "2", 2: "4", 3: "8"}
                    items.append(("bar", 1, -3, 3, "brightness = |ζ(s)|  (one band per doubling)",
                                  [((k + 3) / 6, v) for k, v in labels.items()]))
            items.append(("swatch", (0.1, 0.4, 0.85, 1) if self.plain else (0.3, 0.9, 1.0, 1),
                          "critical line  σ = ½", "line"))
            if self.markers:
                items.append(("swatch", (0, 0, 0, 1), "zero of ζ(s)   (Z: hide markers)", "dot"))
            else:
                items.append(("swatch", None, "zeros: where all colours meet   (Z: markers)", "none"))
        heights = [(lh + bh + lh) if it[0] == "bar" else lh for it in items]
        ph = sum(heights) + m
        pw = max(bw, max(T.width(it[4] if it[0] == "bar" else it[2]) + 2.5 * T.h for it in items)) + 2 * m
        right, top = W - m * 0.5, m * 0.5
        if self.labels and self.mode == 0:          # keep clear of the t labels on the right edge
            step, _ = self.grid_info()
            hf = self.half()
            right -= max(T.width(fmt_num(self.center[1] + sgn * hf[1], step[1])) for sgn in (-1, 1)) + 1.5 * T.h
        x0 = right - pw
        y0 = top
        T.rect(x0, y0, x0 + pw, y0 + ph, (0.02, 0.02, 0.06, 0.72))
        y = y0 + m * 0.8
        for it, hgt in zip(items, heights):
            if it[0] == "bar":
                _, kind, lo, hi, title, ticks = it
                T.add(x0 + m, y, title, (0.75, 0.85, 1.0, 1))
                by = y + lh
                self.bars.append(((x0 + m, by, x0 + m + bw, by + bh), kind, lo, hi))
                for f, lab in ticks:
                    tx = x0 + m + f * bw
                    T.rect(tx - 0.5, by + bh, tx + 0.5, by + bh + T.h * 0.25, (0.9, 0.9, 1, 0.9))
                    T.add(tx, by + bh + T.h * 0.25, lab, (0.95, 0.95, 1, 1), "ct")
            else:
                _, col, label, shape = it
                cy = y + T.h / 2
                if shape == "line":
                    T.rect(x0 + m, cy - 1.5, x0 + m + 1.6 * T.h, cy + 1.5, col)
                elif shape == "dot":
                    r = T.h * 0.32
                    T.rect(x0 + m + 0.8 * T.h - r - 1, cy - r - 1, x0 + m + 0.8 * T.h + r + 1, cy + r + 1, (0.6, 0.6, 0.65, 1))
                    T.rect(x0 + m + 0.8 * T.h - r, cy - r, x0 + m + 0.8 * T.h + r, cy + r, col)
                T.add(x0 + m + 2.2 * T.h, y, label, (0.95, 0.95, 1, 1))
            y += hgt

    def draw_bars(self, fb):
        if not self.bars:
            return
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        for rect, kind, lo, hi in self.bars:
            set_uniforms(self.p_cbar, u_rect=tuple(map(float, rect)), u_res=(float(fb[0]), float(fb[1])),
                         u_kind=kind, u_lo=float(lo), u_hi=float(hi), u_scheme=self.scheme)
            self.vao_cbar.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.bars = []

    def cursor_readout(self):
        s = self.cursor_s()
        out = f"s = {fmt_complex(complex(*s))}"
        job = self.jobs["hi"] if self.jobs["hi"].done else self.jobs["lo"]
        if not job.done:
            return out
        ndc = self.cursor_ndc()
        tw, th = job.size
        px, py = int((ndc[0] + 1) / 2 * tw), int((ndc[1] + 1) / 2 * th)
        if 0 <= px < tw and 0 <= py < th:
            # bind the float framebuffer for the read: with the (8-bit) window framebuffer bound,
            # the driver may clamp the result to [0, 1] -- which turned log|zeta| < 0 into |zeta| = 1
            job.fbo.use()
            d = np.frombuffer(job.fbo.read(viewport=(px, py, 1, 1), components=4, dtype="f4"), dtype="f4")
            fw, fh = glfw.get_framebuffer_size(self.win)
            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, fw, fh)
            if np.all(np.isfinite(d)):
                lm, ang = float(d[0]), math.atan2(d[2], d[1])
                if lm < 700:
                    z = math.exp(lm) * complex(math.cos(ang), math.sin(ang))
                    out += f"    ζ(s) = {fmt_complex(z)}    |ζ| = {abs(z):.5g}"
                else:
                    out += f"    |ζ(s)| = e^{lm:.6g}    arg = {ang:.5f}"
        return out

    def help_panel(self, fb):
        common = [
            ("1 2 3 / Tab", "switch view"),
            ("mouse", "drag: pan / rotate    right- or shift-drag: pan camera    scroll: zoom"),
            ("space", "animate"),
            ("K / shift-K", f"next / previous colour map ({len(SCHEME_NAMES)}: {SCHEME_NAMES[self.scheme]} now)"),
            ("X / shift-X", "display: colour + brightness / colour only / brightness only / lines only"),
            ("B", "colour bar / legend: what colour and brightness mean"),
            ("Z", "zero markers on / off"),
            ("L / I", "axis labels / info panel"),
            ("T", "go to a value of t (type e.g. 1e6, then Enter)"),
            ("M", "method: auto / Euler–Maclaurin only / Riemann–Siegel only"),
            ("S / R", "screenshot / reset view"),
            ("F", "fullscreen on / off"),
            ("?  (or F1)", "this help            Esc / Q   quit"),
        ]
        if self.mode == 2:
            spec = [
                ("← →", "scroll the t-window (shift: faster)"),
                ("[ ]", "widen / narrow the t-window"),
                ("↑ ↓ / 0", "change σ (shift: coarse) / back to σ = ½"),
                ("- =", "amplitude"),
                (", .", "animation speed"),
                ("U", "unwind: multiply by e^{iθ(t)} -> Hardy's real Z(t)"),
                ("V", "camera: side / end-on / top / front"),
            ]
        else:
            spec = [
                ("arrows [ ]", "move / zoom the region (shift: faster)"),
                ("N / P", "next / previous zero on the critical line"),
                ("A", "aspect: 1:1 → σ squeezed 2.5× → t stretched 2.5× → 1:1" if self.mode == 0
                      else "ranges: wider σ → wider σ and longer t → narrower σ"),
                ("C / G", "contour lines / grid"),
            ]
            if self.mode == 1:
                spec += [("H", "height: |ζ|  <->  log|ζ|"), ("- =", "height scale"), ("shift-scroll", "zoom the region")]
        rows = [("", "Controls")] + common + [("", ""), ("", self.MODES[self.mode])] + spec
        T = self.text
        lh = T.h * 1.45
        kw = max(T.width(k) for k, _ in rows) + T.h * 1.5
        tw = kw + max(T.width(v) for _, v in rows)
        W, H = fb
        x0, y0 = (W - tw) / 2, (H - lh * len(rows)) / 2
        pad = T.h * 1.2
        T.rect(x0 - pad, y0 - pad, x0 + tw + pad, y0 + lh * len(rows) + pad * 0.6, (0.02, 0.02, 0.07, 0.88))
        for i, (k, v) in enumerate(rows):
            y = y0 + i * lh
            if not k:
                T.add(x0, y, v, (0.5, 0.9, 1.0, 1))
            else:
                T.add(x0, y, k, (1, 0.9, 0.5, 1))
                T.add(x0 + kw, y, v, (0.92, 0.94, 1, 1))

    # ---------------------------------------------------------------- main loop
    def save_shot(self, fb, path=None):
        w, h = fb
        data = self.ctx.screen.read(viewport=(0, 0, w, h), components=3, alignment=1)
        if path is None:      # S key: timestamped file in ./screenshots (created on first use)
            os.makedirs(SHOT_DIR, exist_ok=True)
            path = os.path.join(SHOT_DIR, time.strftime("zeta_%Y%m%d_%H%M%S.png"))
        write_png(path, w, h, data)
        self.say(f"saved {path}")

    def run(self):
        last = time.time()
        frames_total = 0
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            now = time.time()
            dt, last = now - last, now
            if self.animate:
                self.flow += dt * 0.6
                if self.mode == 1 and self.drag is None:
                    self.cam_land.yaw += 0.003
            fb = glfw.get_framebuffer_size(self.win)
            if fb[0] == 0 or fb[1] == 0:
                time.sleep(0.05)
                continue
            ww = glfw.get_window_size(self.win)[0]
            px = int(round(13 * fb[0] / max(ww, 1)))
            if self.text is None or px != self.text_px:
                if self.text is not None:
                    self.text.release()
                self.text, self.text_px = Text(self.ctx, px), px
            self.cam_land.update(dt)
            self.cam_spiral.update(dt)
            self.run_jobs(now)
            self.ctx.screen.use()
            self.ctx.screen.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
            if self.mode == 0:
                self.render_plane(fb, now)
            elif self.mode == 1:
                self.render_landscape(fb, now)
            else:
                self.render_spiral(fb, now, dt)
            self.hud(fb, now)
            self.text.flush(fb)
            self.draw_bars(fb)
            if self.want_shot:
                self.save_shot(fb)
                self.want_shot = False
            frames_total += 1
            if self.shot and frames_total > 5 and (not self.busy() or now - self.t_start > 90):
                self.save_shot(fb, self.shot)
                break
            glfw.swap_buffers(self.win)
            self.frames += 1
            if now - self.fps_t > 0.5:
                self.fps = self.frames / (now - self.fps_t)
                self.frames, self.fps_t = 0, now
        glfw.terminate()


def main():
    ap = argparse.ArgumentParser(description="GPU Riemann zeta explorer", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", type=int, default=2, choices=[1, 2, 3], help="initial view (1 plane, 2 landscape, 3 critical line)")
    ap.add_argument("--size", default="1400x900", help="window size WxH")
    ap.add_argument("--center", type=float, nargs=2, metavar=("SIGMA", "T"), help="centre of the s-region (views 1, 2)")
    ap.add_argument("--height", type=float, help="half-height (in t) of the s-region")
    ap.add_argument("--t", type=float, nargs=2, metavar=("T0", "T1"), help="t-window for view 3")
    ap.add_argument("--sigma", type=float, help="sigma for view 3 (default 0.5)")
    ap.add_argument("--scheme", default=DEFAULT_SCHEME, help="colour scheme: " + ", ".join(SCHEME_NAMES))
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--unwind", action="store_true", help="view 3: start unwound (Hardy Z)")
    ap.add_argument("--camera", type=int, default=0, help="view 3 camera preset 0-3")
    ap.add_argument("--labels", action="store_true", help="start with axis labels on (toggle: L)")
    ap.add_argument("--info", action="store_true", help="start with the info panel on (toggle: I)")
    ap.add_argument("--style", choices=["full", "phase", "modulus", "lines"], default="phase",
                    help="full = hue (arg) + brightness (|zeta|), phase = hue only, modulus = brightness only, lines = contour lines only (cycle: X)")
    ap.add_argument("--no-color", action="store_true", help="same as --style lines")
    ap.add_argument("--colorbar", action="store_true", help="start with the colour bar shown (toggle: B)")
    ap.add_argument("--no-markers", action="store_true", help="hide the zero markers (toggle: Z)")
    ap.add_argument("--help-panel", action="store_true", help="start with the help panel open")
    ap.add_argument("--view", type=float, nargs=3, metavar=("YAW", "PITCH", "ZOOM"),
                    help="landscape camera: yaw and pitch in radians (default 1.0 0.6), zoom factor (default 1)")
    ap.add_argument("--shot", metavar="PNG", help="render, save a screenshot and exit")
    a = ap.parse_args()
    w, h = map(int, a.size.lower().split("x"))
    app = App(w, h, mode=a.mode - 1, shot=a.shot)
    if a.center:
        app.center = np.array(a.center)
    if a.height:
        app.hy = a.height
    if a.t:
        app.sp_c, app.sp_w = (a.t[0] + a.t[1]) / 2, a.t[1] - a.t[0]
    if a.sigma is not None:
        app.sp_sigma = a.sigma
    names = [n.lower() for n in SCHEME_NAMES]
    app.scheme = names.index(a.scheme.lower()) if a.scheme.lower() in names else SCHEME_NAMES.index(DEFAULT_SCHEME)
    app.animate = a.animate
    if a.unwind:
        app.unwind = app.unwind_cur = 1.0
    if a.camera:
        app.cam_preset = a.camera % len(CAM_PRESETS)
        _, app.cam_spiral.yaw, app.cam_spiral.pitch, app.cam_spiral.dist = CAM_PRESETS[app.cam_preset]
    app.help = a.help_panel
    app.labels, app.info = int(a.labels), int(a.info)
    app.colorbar, app.markers = int(a.colorbar), int(not a.no_markers)
    app.style = 3 if a.no_color else ["full", "phase", "modulus", "lines"].index(a.style)
    if a.view:
        app.land_view = tuple(a.view)
        app.fit_land_camera()
    if not app.info and not a.shot:
        app.say("press ? or F1 for help")
    app.region_changed()
    app.spiral_changed()
    app.run()


if __name__ == "__main__":
    main()
