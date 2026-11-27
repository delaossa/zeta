// GPU evaluation of the Riemann zeta function (WebGL2 / GLSL ES 3.00, float32), plus the CPU-side
// helpers it needs. A port of zeta_core.py: see that file for the maths.
//
// Float32 cannot represent t*ln(n) for large t, so all GPU code works with t = t_ref + d, where d is
// small. The CPU precomputes (in double / double-double precision)
//     phase table   phi_n = (t_ref * ln n) mod 2pi        (texture)
//     theta0(t_ref) mod 2pi,   a0 = sqrt(t_ref / 2pi)
// and the GPU only adds d*ln n etc.
"use strict";

const TAU = 2 * Math.PI;
const TABLE_W = 4096;
const RS_THRESHOLD = 1000.0;    // auto switch Euler-Maclaurin -> Riemann-Siegel

// B_{2k} / (2k)!  for k = 1..8
const BERN = [1 / 12, -1 / 720, 1 / 30240, -1 / 1209600, 1 / 47900160,
              -691 / 1307674368000, 1 / 74724249600, -3617 / 10670622842880000];

const glslFloat = (x) => {
    let s = String(x);
    if (!/[.eE]/.test(s)) s += ".0";
    return s;
};

const GLSL_HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const float PI  = 3.14159265358979;
const float TAU = 6.28318530717959;
`;

const ZETA_GLSL = `
const float BERN[8] = float[8](${BERN.map(glslFloat).join(", ")});

uniform sampler2D u_phase;     // r: (t_ref * ln n) mod 2pi   at texel (n % 4096, n / 4096)
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

// (log|zeta|, arg zeta) for s = sg + i (t_ref + d)
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
`;

const FULLSCREEN_VS = GLSL_HEADER + `
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

// region of the s-plane -> texture (log|zeta|, cos arg, sin arg)
const COMPUTE_FS = GLSL_HEADER + ZETA_GLSL + `
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
`;

// samples along a vertical line sigma + i t, t in [lo, lo + w] -> (Re zeta, Im zeta, theta(t), log|zeta|)
const LINE_FS = GLSL_HEADER + ZETA_GLSL + `
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
`;

// ---------------------------------------------------------------------------------------------
// double-double helpers: (t * x) mod 2pi without losing the fractional part for t up to ~1e10
// ---------------------------------------------------------------------------------------------
const TAU_HI = 6.283185307179586, TAU_LO = 2.4492935982947064e-16;

function twoProd(a, b) {
    const p = a * b;
    let c = 134217729 * a;
    const ah = c - (c - a), al = a - ah;
    c = 134217729 * b;
    const bh = c - (c - b), bl = b - bh;
    return [p, ((ah * bh - p) + ah * bl + al * bh) + al * bl];
}

// (p + e) mod 2pi in [0, 2pi), where p + e is an unevaluated double-double sum
function modTauDD(p, e) {
    const k = Math.round(p / TAU_HI);
    const [q, qe] = twoProd(k, TAU_HI);
    let r = (p - q) + (e - qe) - k * TAU_LO;
    r -= TAU_HI * Math.floor(r / TAU_HI);
    return r;
}

function phaseMod(t, l) {          // (t * l) mod 2pi
    const [p, e] = twoProd(t, l);
    return modTauDD(p, e);
}

function theta0Mod(t) {            // (t/2 log(t/2pi) - t/2 - pi/8) mod 2pi
    if (t <= 0) return 0;
    const [p, e] = twoProd(0.5 * t, Math.log(t / TAU) - 1);
    return modTauDD(p, e - Math.PI / 8);
}

function termsNeeded(sigmaAbsMax, tAbsMax, rsT) {
    const emT = Math.min(tAbsMax, rsT);
    let n = 10 + 0.4 * Math.hypot(sigmaAbsMax + 1, emT);
    if (tAbsMax >= rsT) n = Math.max(n, Math.sqrt(tAbsMax / TAU) + 3);
    return Math.floor(n) + 2;
}

function costPerSample(sigmaAbsMax, tAbsMax, rsT) {   // rough number of series terms per point
    if (tAbsMax >= rsT) return 2 * Math.sqrt(tAbsMax / TAU) + 30;
    return 8 + 0.4 * Math.hypot(sigmaAbsMax + 1, tAbsMax) + 30;
}

// GPU texture of (t_ref * ln n) mod 2pi plus the scalar uniforms that go with it
class PhaseTable {
    constructor(gl, unit) {
        this.gl = gl;
        this.unit = unit;
        this.tex = null;
        this.rows = 0;
        this.tref = null;
        this.ntab = 0;
        this.maxN = TABLE_W * 1024;
    }

    // rebuild when the view moved too far from t_ref or needs more terms. returns true if rebuilt
    update(tCenter, tHalf, nNeed) {
        nNeed = Math.min(nNeed, this.maxN);
        if (this.tref !== null && nNeed <= this.ntab && Math.abs(tCenter - this.tref) <= Math.max(16, tHalf)
            && (tCenter < 0) === (this.tref < 0)) return false;
        this.tref = tCenter;
        const t0 = Math.abs(tCenter);
        let ntab = 1024;
        while (ntab < nNeed) ntab *= 2;
        const rows = Math.ceil(ntab / TABLE_W);
        const data = new Float32Array(rows * TABLE_W);
        for (let n = 2; n < ntab; n++) data[n] = phaseMod(t0, Math.log(n));
        const gl = this.gl;
        if (this.tex === null || this.rows !== rows) {
            if (this.tex) gl.deleteTexture(this.tex);
            this.tex = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this.tex);
            gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, TABLE_W, rows);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            this.rows = rows;
        }
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TABLE_W, rows, gl.RED, gl.FLOAT, data);
        this.ntab = ntab;
        const a0 = Math.sqrt(t0 / TAU);
        this.t0 = t0;
        this.flip = tCenter < 0 ? 1 : 0;
        this.theta0 = theta0Mod(t0);
        this.a0 = a0;
        this.N0 = Math.floor(a0);
        return true;
    }

    // bind the texture and set the uniforms on a program (see Program.set)
    bind(prog, rsT) {
        const gl = this.gl;
        gl.activeTexture(gl.TEXTURE0 + this.unit);
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        prog.set({u_phase: this.unit, u_t0: this.t0, u_theta0: this.theta0, u_a0: this.a0,
                  u_a0frac: this.a0 - this.N0, u_N0: this.N0, u_ntab: this.ntab, u_rs_t: rsT});
    }

    rel(t) { return t - this.tref; }    // t -> d in the (possibly flipped) shader frame
}

// ---------------------------------------------------------------------------------------------
// CPU (double precision) Hardy Z(t) -- used to jump to the next / previous zero (N / P keys)
// ---------------------------------------------------------------------------------------------
const C = {
    mul: (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]],
    div: (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; },
    log: (z) => [Math.log(Math.hypot(z[0], z[1])), Math.atan2(z[1], z[0])],
};

function clgammaCPU(z) {         // log Gamma(z), Re z > 0
    let acc = [0, 0];
    z = [z[0], z[1]];
    for (let k = 0; k < 8; k++) { const l = C.log(z); acc = [acc[0] + l[0], acc[1] + l[1]]; z[0] += 1; }
    const zi = C.div([1, 0], z), zi2 = C.mul(zi, zi);
    let ser = [-1 / 1680 * zi2[0], -1 / 1680 * zi2[1]];
    ser = C.mul(zi2, [ser[0] + 1 / 1260, ser[1]]);
    ser = C.mul(zi2, [ser[0] - 1 / 360, ser[1]]);
    ser = C.mul(zi, [ser[0] + 1 / 12, ser[1]]);
    const lz = C.log(z), a = C.mul([z[0] - 0.5, z[1]], lz);
    return [a[0] - z[0] + 0.91893853320467274 + ser[0] - acc[0], a[1] - z[1] + ser[1] - acc[1]];
}

function thetaCPU(t) {           // Riemann-Siegel theta(t) mod 2pi, t > 0
    if (t < 50) return modTauDD(clgammaCPU([0.25, 0.5 * t])[1] - 0.5 * t * Math.log(Math.PI), 0);
    return modTauDD(theta0Mod(t), 1 / (48 * t) + 7 / (5760 * t * t * t) + 31 / (80640 * t ** 5));
}

function hardyZ(t) {
    t = Math.abs(t);
    const th = thetaCPU(t);
    if (t < 1e4) {               // Euler-Maclaurin for zeta(1/2 + it)
        const s = [0.5, t];
        const N = Math.floor(10 + 0.4 * Math.hypot(0.5, t));
        let re = 0, im = 0;
        for (let n = 1; n < N; n++) {
            const ph = th - t * Math.log(n), m = 1 / Math.sqrt(n);
            re += m * Math.cos(ph);
            im += m * Math.sin(ph);
        }
        // tail terms, all multiplied by e^{i theta}
        const lnN = Math.log(N);
        const Nms = [Math.cos(th - t * lnN) / Math.sqrt(N), Math.sin(th - t * lnN) / Math.sqrt(N)];
        const tail1 = C.div([N * Nms[0], N * Nms[1]], [s[0] - 1, s[1]]);
        re += tail1[0] + 0.5 * Nms[0];
        im += tail1[1] + 0.5 * Nms[1];
        let w = C.mul(Nms, s);
        w = [w[0] / N, w[1] / N];
        re += BERN[0] * w[0];
        for (let k = 1; k < 8; k++) {
            const a = 2 * k - 1;
            w = C.mul(w, C.mul([s[0] + a, s[1]], [s[0] + a + 1, s[1]]));
            w = [w[0] / (N * N), w[1] / (N * N)];
            re += BERN[k] * w[0];
        }
        return re;
    }
    // Riemann-Siegel with the first two correction terms
    const a = Math.sqrt(t / TAU), N = Math.floor(a), p = a - N;
    let sum = 0;
    for (let n = 1; n <= N; n++) sum += Math.cos(th - phaseMod(t, Math.log(n))) / Math.sqrt(n);
    const C0 = (q) => {
        const c = Math.cos(TAU * q);
        if (Math.abs(c) < 1e-6) q += 1e-5;
        return Math.cos(TAU * (q * q - q - 1 / 16)) / Math.cos(TAU * q);
    };
    const h = 2e-3;
    const d3 = (C0(p + 2 * h) - 2 * C0(p + h) + 2 * C0(p - h) - C0(p - 2 * h)) / (2 * h * h * h);
    const C1 = -d3 / (96 * Math.PI * Math.PI);
    const sign = (N - 1) % 2 === 0 ? 1 : -1;
    return 2 * sum + sign * Math.pow(a, -0.5) * (C0(p) + C1 / a);
}

// next zero of Hardy's Z (= zero of zeta on the critical line) from t in the given direction
function findZero(t, direction) {
    const step = 0.1 * TAU / Math.log(Math.max(Math.abs(t), 30) / TAU);
    let a = t + direction * step * 0.02, za = hardyZ(a);
    for (let i = 0; i < 5000; i++) {
        const b = a + direction * step, zb = hardyZ(b);
        if (za * zb < 0) {          // Illinois regula falsi on the bracket
            let x0 = a, f0 = za, x1 = b, f1 = zb, side = 0;
            for (let it = 0; it < 100 && Math.abs(x1 - x0) > 1e-13 * Math.max(1, Math.abs(x0)); it++) {
                const xm = (x0 * f1 - x1 * f0) / (f1 - f0), fm = hardyZ(xm);
                if (fm * f1 > 0) { x1 = xm; f1 = fm; if (side === -1) f0 /= 2; side = -1; }
                else if (fm * f0 > 0) { x0 = xm; f0 = fm; if (side === 1) f1 /= 2; side = 1; }
                else return xm;
            }
            return (x0 * f1 - x1 * f0) / (f1 - f0);
        }
        a = b; za = zb;
    }
    return null;
}
