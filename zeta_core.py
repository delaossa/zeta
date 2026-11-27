"""
GPU evaluation of the Riemann zeta function (GLSL, float32) + the CPU-side helpers it needs.

Float32 cannot represent t*ln(n) for large t, so all GPU code works with t = t_ref + d, where
d is small.  The CPU precomputes, in double precision (or mpmath for t > 1e8):

    phase table   phi_n = (t_ref * ln n) mod 2pi          n = 0 .. ntab-1   (texture)
    theta0(t_ref) mod 2pi,   a0 = sqrt(t_ref / 2pi)

and the GPU only adds d*ln n etc.  With that, two methods are available per pixel:

  * Euler-Maclaurin (|t| < u_rs_t)  -- exact to float precision, cost ~0.4|s| terms
       Re s < 1/2 handled with the functional equation (log space, no overflow)
  * Riemann-Siegel  (t >= u_rs_t)    -- approximate functional equation
       zeta(s) = R(s) + chi(s) conj(R(1 - conj s)),
       R(s)    = sum_{n<=N} n^-s + (-1)^(N-1) U a^-sigma F(p)     (Arias de Reyna, first term)
       a = sqrt(t/2pi), N = floor(a), p = 1 - 2(a - N), U = exp(-i theta0(t)),
       F(p) = (exp(i pi (p^2/2 + 3/8)) - i sqrt2 cos(pi p / 2)) / (2 cos(pi p)),
     valid for any sigma, cost ~2 sqrt(t/2pi) terms.
"""
import math

import numpy as np
import moderngl

try:
    import mpmath
except ImportError:
    mpmath = None

TAU = 2.0 * math.pi
TABLE_W = 4096
RS_THRESHOLD = 1000.0  # auto switch EM -> RS (see test_zeta_gpu.py)

# B_{2k} / (2k)!  for k = 1..8
_BERN = [1 / 12, -1 / 720, 1 / 30240, -1 / 1209600, 1 / 47900160,
         -691 / 1307674368000, 1 / 74724249600, -3617 / 10670622842880000]

HEADER = "#version 410 core\n"

ZETA_GLSL = """
const float PI  = 3.14159265358979;
const float TAU = 6.28318530717959;
const float BERN[8] = float[8](%s);

uniform sampler2D u_phase;     // r: (t_ref * ln n) mod 2pi   at texel (n %% 4096, n / 4096)
uniform float u_t0;            // |t_ref|
uniform float u_theta0;        // theta0(t_ref) mod 2pi
uniform float u_a0;            // sqrt(t_ref / 2pi)
uniform float u_a0frac;        // a0 - floor(a0)
uniform int   u_N0;            // floor(a0)
uniform int   u_ntab;          // entries in the phase table
uniform float u_rs_t;          // use Riemann-Siegel for t >= this

float wrap(float x) {
    float k = floor(x * 0.15915494309189535 + 0.5);
    return (x - k * 6.28125) - k * 0.0019353071795864769;
}
vec2 cis(float a)          { a = wrap(a); return vec2(cos(a), sin(a)); }
vec2 cmul(vec2 a, vec2 b)  { return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 cdiv(vec2 a, vec2 b)  { return vec2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / dot(b, b); }
vec2 cexp(vec2 z)          { return exp(z.x) * cis(z.y); }
vec2 clog(vec2 z)          { return vec2(log(length(z)), atan(z.y, z.x)); }
float log1p_(float x) {
    if (abs(x) < 1e-2) return x * (1.0 - x * (0.5 - x * (1.0/3.0 - x * (0.25 - x * 0.2))));
    return log(1.0 + x);
}
// (t * ln n) mod 2pi  for t = t_ref + d
float phase_n(int n, float d) {
    return texelFetch(u_phase, ivec2(n & 4095, n >> 12), 0).r + d * log(float(n));
}

// Euler-Maclaurin: zeta(sg + i*sgn*T), T = t_ref + d
vec2 zeta_em(float sg, float d, float sgn) {
    vec2 s = vec2(sg, sgn * (u_t0 + d));
    float N = min(floor(8.0 + 0.4 * length(s)), float(u_ntab - 1));
    int Ni = int(N);
    vec2 sum = vec2(0.0);
    for (int n = 1; n < 1048576; n++) {
        if (n >= Ni) break;
        sum += exp(-sg * log(float(n))) * cis(-sgn * phase_n(n, d));
    }
    vec2 Nms = exp(-sg * log(N)) * cis(-sgn * phase_n(Ni, d));    // N^-s
    sum += cdiv(N * Nms, s - vec2(1.0, 0.0));                      // N^(1-s)/(s-1)
    sum += 0.5 * Nms;
    vec2 w = cmul(Nms, s) / N;
    sum += BERN[0] * w;
    for (int k = 1; k < 8; k++) {
        float a = float(2 * k - 1);
        w = cmul(w, cmul(s + vec2(a, 0.0), s + vec2(a + 1.0, 0.0))) / (N * N);
        sum += BERN[k] * w;
    }
    return sum;
}

vec2 clogsin(vec2 z) {
    const vec2 LOG2I = vec2(0.6931471805599453, 1.5707963267948966);
    if (z.y >= 0.0) {
        vec2 e = cexp(vec2(-2.0 * z.y, 2.0 * z.x));
        return vec2(z.y, -z.x) + clog(e - vec2(1.0, 0.0)) - LOG2I;
    } else {
        vec2 e = cexp(vec2(2.0 * z.y, -2.0 * z.x));
        return vec2(-z.y, z.x) + clog(vec2(1.0, 0.0) - e) - LOG2I;
    }
}
vec2 clgamma(vec2 z) {   // Re z > 0
    vec2 acc = vec2(0.0);
    for (int k = 0; k < 8; k++) { acc += clog(z); z.x += 1.0; }
    vec2 zi = cdiv(vec2(1.0, 0.0), z);
    vec2 zi2 = cmul(zi, zi);
    vec2 ser = cmul(zi, vec2(1.0/12.0, 0.0) + cmul(zi2, vec2(-1.0/360.0, 0.0)
               + cmul(zi2, vec2(1.0/1260.0, 0.0) + zi2 * (-1.0/1680.0))));
    return cmul(z - vec2(0.5, 0.0), clog(z)) - z + vec2(0.91893853320467, 0.0) + ser - acc;
}

// theta0(t) - theta0(t_ref),  theta0(t) = t/2 log(t/2pi) - t/2 - pi/8
float dtheta0(float d) {
    float T = u_t0 + d;
    return 0.5 * u_t0 * log1p_(d / u_t0) + 0.5 * d * log(T / TAU) - 0.5 * d;
}
// Riemann-Siegel theta(t) mod 2pi
float theta_rel(float d) {
    float T = u_t0 + d;
    if (abs(T) < 200.0 || u_t0 < 1.0) {
        float at = abs(T);
        float th = clgamma(vec2(0.25, 0.5 * at)).y - 0.5 * at * 1.1447298858494002;
        return wrap(T < 0.0 ? -th : th);
    }
    return wrap(u_theta0 + dtheta0(d) + 1.0 / (48.0 * T) + 7.0 / (5760.0 * T * T * T));
}

vec2 rsF0(float p) {
    vec2 num = cis(PI * (0.5 * p * p + 0.375)) - vec2(0.0, 1.4142135623730951 * cos(0.5 * PI * p));
    return num / (2.0 * cos(PI * p));
}
vec2 rsF(float p) {      // removable singularities at p = +-1/2
    const float h = 0.01;
    float pc = p > 0.0 ? 0.5 : -0.5;
    if (abs(p - pc) < h) return mix(rsF0(pc - h), rsF0(pc + h), (p - (pc - h)) / (2.0 * h));
    return rsF0(p);
}

// Riemann-Siegel, t = t_ref + d > 0 large; returns (log|zeta|, arg zeta)
vec2 zeta_rs_log(float sg, float d) {
    float T = u_t0 + d;
    float a = sqrt(T / TAU);
    float fr = u_a0frac + (d / TAU) / (u_a0 + a);
    float fl = floor(fr);
    int N = min(u_N0 + int(fl), u_ntab - 1);
    fr -= fl;
    float p = 1.0 - 2.0 * fr;
    vec2 A = vec2(0.0), B = vec2(0.0);
    for (int n = 1; n < 1048576; n++) {
        if (n > N) break;
        float ln = log(float(n));
        vec2 c = cis(-phase_n(n, d));
        A += exp(-sg * ln) * c;
        B += exp((sg - 1.0) * ln) * c;
    }
    float th = u_theta0 + dtheta0(d);
    vec2 UF = cmul(cis(-th), rsF(p)) * ((N & 1) == 1 ? 1.0 : -1.0);
    A += pow(a, -sg) * UF;
    B += pow(a, sg - 1.0) * UF;
    B.y = -B.y;
    vec2 s = vec2(sg, T);
    vec2 Lg = clog(vec2(1.0, -sg / T));                       // log(1 - i sigma / t)
    vec2 lchi = vec2((0.5 - sg) * log(T / TAU) + sg, -2.0 * th)
              - cmul(s - vec2(0.5, 0.0), Lg)
              - cdiv(vec2(1.0 / 12.0, 0.0), s) + cdiv(vec2(1.0 / 360.0, 0.0), cmul(s, cmul(s, s)));
    vec2 X = lchi + clog(B), Y = clog(A);
    float M = max(X.x, Y.x);
    vec2 lz = clog(exp(X.x - M) * cis(X.y) + exp(Y.x - M) * cis(Y.y));
    return vec2(M + lz.x, lz.y);
}

// (log|zeta(s)|, arg zeta(s)) for s = sg + i (t_ref + d)
vec2 zeta_log_rel(float sg, float d) {
    float T = u_t0 + d;
    if (T >= u_rs_t) return zeta_rs_log(sg, d);
    if (sg >= 0.5) return clog(zeta_em(sg, d, 1.0));
    vec2 s = vec2(sg, T);
    vec2 r = vec2(1.0 - sg, -T);
    vec2 lchi = s * 0.6931471805599453 + (s - vec2(1.0, 0.0)) * 1.1447298858494002
              + clogsin(s * (0.5 * PI)) + clgamma(r);
    vec2 L = lchi + clog(zeta_em(1.0 - sg, d, -1.0));
    return vec2(L.x, wrap(L.y));
}
""" % ", ".join(repr(b) for b in _BERN)

FULLSCREEN_VS = HEADER + """
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# region of the s-plane -> texture (log|zeta|, cos arg, sin arg)
COMPUTE_FS = HEADER + ZETA_GLSL + """
uniform vec2 u_sig;            // sigma: (center, half)
uniform vec2 u_tt;             // t - t_ref: (center, half)
uniform int  u_flip;           // t_ref < 0: evaluate at conj(s) and conjugate
uniform vec2 u_off, u_den;     // uv = (fragcoord - off) / den
out vec4 o;
void main() {
    vec2 uv = (gl_FragCoord.xy - u_off) / u_den;
    float sg = u_sig.x + (uv.x * 2.0 - 1.0) * u_sig.y;
    float d  = u_tt.x  + (uv.y * 2.0 - 1.0) * u_tt.y;
    if (u_flip != 0) d = -d;
    vec2 L = zeta_log_rel(sg, d);
    if (u_flip != 0) L.y = -L.y;
    o = vec4(L.x, cos(L.y), sin(L.y), 1.0);
}
"""

# samples along a vertical line sigma + i t, t in [lo, lo + w] -> (Re zeta, Im zeta, theta(t), log|zeta|)
LINE_FS = HEADER + ZETA_GLSL + """
uniform float u_dlo, u_w, u_sigma;
uniform int u_n, u_flip;
out vec4 o;
void main() {
    int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * 4096;
    float f = float(min(idx, u_n - 1)) / float(u_n - 1);
    float d = u_dlo + f * u_w;
    if (u_flip != 0) d = -d;
    vec2 L = zeta_log_rel(u_sigma, d);
    float th = theta_rel(d);
    if (u_flip != 0) { L.y = -L.y; th = -th; }
    o = vec4(exp(min(L.x, 80.0)) * cis(L.y), th, L.x);
}
"""


def theta0_mod(t):
    """(t/2 log(t/2pi) - t/2 - pi/8) mod 2pi, accurately."""
    if t <= 0:
        return 0.0
    if t < 1e8 or mpmath is None:
        return math.fmod(t / 2 * math.log(t / TAU) - t / 2 - math.pi / 8, TAU)
    with mpmath.workdps(40):
        T = mpmath.mpf(t)
        return float(mpmath.fmod(T / 2 * mpmath.log(T / (2 * mpmath.pi)) - T / 2 - mpmath.pi / 8, 2 * mpmath.pi))


def terms_needed(sigma_absmax, t_absmax, rs_t):
    em_t = min(t_absmax, rs_t)
    n = 10 + 0.4 * math.hypot(sigma_absmax + 1.0, em_t)
    if t_absmax >= rs_t:
        n = max(n, math.sqrt(t_absmax / TAU) + 3)
    return int(n) + 2


def cost_per_sample(sigma_absmax, t_absmax, rs_t):
    """rough number of series terms per evaluated point"""
    if t_absmax >= rs_t:
        return 2.0 * math.sqrt(t_absmax / TAU) + 30
    return 8 + 0.4 * math.hypot(sigma_absmax + 1.0, t_absmax) + 30


class PhaseTable:
    """GPU texture of (t_ref * ln n) mod 2pi plus the scalar uniforms that go with it."""

    def __init__(self, ctx, unit):
        self.ctx, self.unit = ctx, unit
        self.tex = None
        self.tref = None
        self.ntab = 0
        self.max_n = TABLE_W * 1024

    def update(self, t_center, t_half, n_need):
        """rebuild when the view moved too far from t_ref or needs more terms. returns True if rebuilt"""
        n_need = min(n_need, self.max_n)
        if (self.tref is not None and n_need <= self.ntab
                and abs(t_center - self.tref) <= max(16.0, t_half)
                and (t_center < 0) == (self.tref < 0)):
            return False
        self.tref = float(t_center)
        t0 = abs(self.tref)
        ntab = max(1024, 1 << (n_need - 1).bit_length())
        n = np.arange(ntab, dtype=np.float64)
        n[0] = 1.0
        if t0 < 1e8 or mpmath is None:
            ph = np.mod(t0 * np.log(n), TAU)
        else:
            with mpmath.workdps(30):
                T, tau = mpmath.mpf(t0), 2 * mpmath.pi
                ph = np.array([float(mpmath.fmod(T * mpmath.log(k), tau)) for k in range(1, ntab + 1)])
                ph = np.concatenate([[0.0], ph[:-1]])
        rows = (ntab + TABLE_W - 1) // TABLE_W
        data = np.zeros(rows * TABLE_W, dtype="f4")
        data[:ntab] = ph
        if self.tex is None or self.tex.size != (TABLE_W, rows):
            if self.tex is not None:
                self.tex.release()
            self.tex = self.ctx.texture((TABLE_W, rows), 1, dtype="f4")
            self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex.write(data.tobytes())
        self.ntab = ntab
        a0 = math.sqrt(t0 / TAU)
        self.t0, self.flip = t0, int(self.tref < 0)
        self.theta0 = theta0_mod(t0)
        self.a0, self.N0 = a0, int(math.floor(a0))
        return True

    def bind(self, prog, rs_t):
        """use texture + set uniforms on prog"""
        self.tex.use(self.unit)
        vals = dict(u_phase=self.unit, u_t0=self.t0, u_theta0=self.theta0, u_a0=self.a0,
                    u_a0frac=self.a0 - self.N0, u_N0=self.N0, u_ntab=self.ntab, u_rs_t=float(rs_t))
        for k, v in vals.items():
            try:
                prog[k].value = v
            except KeyError:
                pass

    def rel(self, t):
        """t -> d in the (possibly flipped) shader frame"""
        return t - self.tref
