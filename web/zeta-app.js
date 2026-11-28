// ζ explorer, web version: WebGL2 port of zeta_viewer.py.
"use strict";

const STYLES = ["colour + brightness", "colour only (arg ζ)", "brightness only (|ζ|)", "lines only"];
const STYLE_KEYS = ["full", "phase", "modulus", "lines"];
const METHODS = [["auto", RS_THRESHOLD], ["Euler–Maclaurin", 1e30], ["Riemann–Siegel", 30.0]];
const CAM_PRESETS = [["side", 0.45, 0.28, 8.5], ["end-on (complex plane)", -Math.PI / 2, 0.0, 6.5],
                     ["top", 0.0, 1.45, 8.5], ["front", 0.0, 0.02, 9.0]];
const MODES = ["complex plane", "landscape", "critical line"];
const RES3D = 512;

// ------------------------------------------------------------------------------------------------
// small helpers
// ------------------------------------------------------------------------------------------------
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vnorm = (a) => { const l = Math.hypot(...a); return l > 0 ? a.map((x) => x / l) : a; };

// column-major 4x4 matrices (as WebGL expects)
function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0];
}
function lookAt(eye, target, up = [0, 1, 0]) {
    const f = vnorm(vsub(target, eye));
    let s = vcross(f, up);
    if (Math.hypot(...s) < 1e-9) s = vcross(f, [0, 0, -1]);
    s = vnorm(s);
    const u = vcross(s, f);
    return [s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0,
            -vdot(s, eye), -vdot(u, eye), vdot(f, eye), 1];
}
function matmul(a, b) {
    const o = new Array(16);
    for (let c = 0; c < 4; c++)
        for (let r = 0; r < 4; r++) {
            let v = 0;
            for (let k = 0; k < 4; k++) v += a[k * 4 + r] * b[c * 4 + k];
            o[c * 4 + r] = v;
        }
    return o;
}

function niceStep(span, target = 6) {      // 1-2-5 step giving about `target` intervals over span
    const raw = Math.max(span, 1e-12) / Math.max(target, 1);
    const e = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 5, 10]) if (raw <= m * e) return m * e;
    return 10 * e;
}

function fmtG(v, p) {                     // like Python's "%.{p}g"
    if (v === 0) return "0";
    if (!isFinite(v)) return String(v);
    const ex = Math.floor(Math.log10(Math.abs(v)));
    if (ex < -4 || ex >= p) {
        let [m, e] = v.toExponential(p - 1).split("e");
        if (m.includes(".")) m = m.replace(/\.?0+$/, "");
        const en = parseInt(e, 10);
        return `${m}e${en < 0 ? "-" : "+"}${String(Math.abs(en)).padStart(2, "0")}`;
    }
    let s = v.toFixed(Math.max(0, p - 1 - ex));
    if (s.includes(".")) s = s.replace(/\.?0+$/, "");
    return s;
}

function fmtNum(v, step) {
    let s;
    if (step !== undefined) {
        const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
        s = v.toFixed(dec);
    } else s = Math.abs(v) < 1e7 ? fmtG(v, 6) : fmtG(v, 10);
    if (s.replace(/[-0.]/g, "") === "") s = "0";
    return s.replace(/-/g, "−");
}

function fmtE(v, p) {                    // like Python's "%.{p}e"
    const [m, e] = v.toExponential(p).split("e");
    const en = parseInt(e, 10);
    return `${m}e${en < 0 ? "-" : "+"}${String(Math.abs(en)).padStart(2, "0")}`;
}

function fmtComplex(a, b) {
    const f = Math.max(Math.abs(a), Math.abs(b)) < 1e6 ? (v) => fmtG(v, 6) : (v) => fmtE(v, 4);
    return `${f(a)} ${b >= 0 ? "+" : "−"} ${f(Math.abs(b))}i`.replace(/-/g, "−");
}

// ------------------------------------------------------------------------------------------------
// WebGL plumbing
// ------------------------------------------------------------------------------------------------
class Program {
    constructor(gl, vs, fs) {
        this.gl = gl;
        const sh = (type, src) => {
            const s = gl.createShader(type);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
                const log = gl.getShaderInfoLog(s);
                console.error(src.split("\n").map((l, i) => `${i + 1}: ${l}`).join("\n"));
                throw new Error("shader compile failed: " + log);
            }
            return s;
        };
        const p = gl.createProgram();
        gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
        gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("link failed: " + gl.getProgramInfoLog(p));
        this.p = p;
        this.u = {};
        const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < n; i++) {
            const info = gl.getActiveUniform(p, i);
            const name = info.name.replace(/\[0\]$/, "");
            this.u[name] = {loc: gl.getUniformLocation(p, info.name), type: info.type};
        }
    }

    use() { this.gl.useProgram(this.p); }

    set(vals) {
        const gl = this.gl;
        gl.useProgram(this.p);
        for (const k in vals) {
            const u = this.u[k];
            if (!u) continue;
            let v = vals[k];
            if (typeof v === "boolean") v = v ? 1 : 0;
            switch (u.type) {
                case gl.FLOAT: gl.uniform1f(u.loc, v); break;
                case gl.FLOAT_VEC2: gl.uniform2fv(u.loc, v); break;
                case gl.FLOAT_VEC3: gl.uniform3fv(u.loc, v); break;
                case gl.FLOAT_VEC4: gl.uniform4fv(u.loc, v); break;
                case gl.FLOAT_MAT4: gl.uniformMatrix4fv(u.loc, false, v); break;
                default: gl.uniform1i(u.loc, v | 0);       // int, bool, samplers
            }
        }
    }
}

// progressive GPU evaluation: a few rows per frame so the page never freezes
class Job {
    constructor(app, prog, w, h, linear) {
        const gl = this.gl = app.gl;
        this.app = app;
        this.prog = prog;
        this.size = [w, h];
        this.tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, w, h);
        const filt = linear && app.floatLinear ? gl.LINEAR : gl.NEAREST;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this.fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        this.row = 0;
        this.started = false;
        this.uniforms = {};
        this.cost = 1;
        this.table = null;
    }

    release() {
        this.gl.deleteFramebuffer(this.fbo);
        this.gl.deleteTexture(this.tex);
    }

    start(table, cost, uniforms) {
        this.table = table;
        this.cost = cost;
        this.uniforms = uniforms;
        this.row = 0;
        this.started = true;
    }

    get done() { return this.started && this.row >= this.size[1]; }
    get progress() { return this.row / this.size[1]; }

    step(budget, rsT) {
        if (!this.started || this.done) return 0;
        const gl = this.gl, [w, h] = this.size;
        const rows = Math.floor(Math.min(h - this.row, Math.max(1, budget / (w * this.cost))));
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
        gl.viewport(0, this.row, w, rows);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        this.table.bind(this.prog, rsT);
        this.prog.set(this.uniforms);
        gl.bindVertexArray(this.app.vaoEmpty);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        this.row += rows;
        return rows * w * this.cost;
    }

    read(x, y, w, h) {
        const gl = this.gl, out = new Float32Array(w * h * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
        gl.readPixels(x, y, w, h, gl.RGBA, gl.FLOAT, out);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return out;
    }
}

class OrbitCam {
    constructor(yaw, pitch, dist, target) {
        this.init = [yaw, pitch, dist, target];
        this.goal = null;
        this.reset();
    }

    reset() {
        [this.yaw, this.pitch, this.dist] = this.init;
        this.target = this.init[3].slice();
        this.goal = null;
    }

    flyTo(yaw, pitch, dist, target = [0, 0, 0]) { this.goal = [yaw, pitch, dist, target]; }

    update(dt) {
        if (!this.goal) return;
        const [yaw, pitch, dist, tgt] = this.goal;
        const k = 1 - Math.exp(-dt * 5);
        const dy = ((yaw - this.yaw + Math.PI) % TAU + TAU) % TAU - Math.PI;
        this.yaw += dy * k;
        this.pitch += (pitch - this.pitch) * k;
        this.dist += (dist - this.dist) * k;
        this.target = this.target.map((v, i) => v + (tgt[i] - v) * k);
        if (Math.abs(dy) < 1e-3 && Math.abs(pitch - this.pitch) < 1e-3 && Math.abs(dist - this.dist) < 1e-3) this.goal = null;
    }

    eye() {
        const cp = Math.cos(this.pitch);
        return [this.target[0] + this.dist * cp * Math.sin(this.yaw), this.target[1] + this.dist * Math.sin(this.pitch),
                this.target[2] + this.dist * cp * Math.cos(this.yaw)];
    }

    mvp(aspect) { return matmul(perspective(Math.PI / 4, aspect, 0.02, 200), lookAt(this.eye(), this.target)); }

    pan(dx, dy) {
        const f = vnorm(vsub(this.target, this.eye()));
        let r = vcross(f, [0, 1, 0]);
        const rl = Math.max(Math.hypot(...r), 1e-9);
        r = r.map((x) => x / rl);
        const u = vcross(r, f);
        const k = this.dist * 0.0015;
        this.target = this.target.map((v, i) => v + (-dx * r[i] + dy * u[i]) * k);
    }
}

// text and panels, drawn on a 2D canvas on top of the WebGL one
class Overlay {
    constructor(canvas) {
        this.c = canvas;
        this.g = canvas.getContext("2d");
    }

    begin(px) {
        const g = this.g;
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.clearRect(0, 0, this.c.width, this.c.height);
        this.px = px;
        this.h = Math.round(px * 1.25);
        g.font = `${px}px Menlo, "SF Mono", Consolas, "DejaVu Sans Mono", monospace`;
        g.textBaseline = "top";
        g.lineJoin = "round";
    }

    static css(col, mulA = 1) {
        const a = col.length > 3 ? col[3] : 1;
        return `rgba(${Math.round(col[0] * 255)},${Math.round(col[1] * 255)},${Math.round(col[2] * 255)},${a * mulA})`;
    }

    width(s) { return this.g.measureText(s).width; }

    add(x, y, s, col = [1, 1, 1, 1], anchor = "lt") {     // anchor: [l|c|r][t|m|b]
        const g = this.g, w = this.width(s);
        x -= {l: 0, c: w / 2, r: w}[anchor[0]];
        y -= {t: 0, m: this.h / 2, b: this.h}[anchor[1]];
        x = Math.round(x);
        y = Math.round(y + (this.h - this.px) / 2);
        g.textAlign = "left";
        g.lineWidth = Math.max(2, this.px / 4);
        g.strokeStyle = Overlay.css([0, 0, 0, col.length > 3 ? col[3] : 1], 0.75);
        g.strokeText(s, x, y);
        g.fillStyle = Overlay.css(col);
        g.fillText(s, x, y);
        return w;
    }

    rect(x0, y0, x1, y1, col) {
        this.g.fillStyle = Overlay.css(col);
        this.g.fillRect(x0, y0, x1 - x0, y1 - y0);
    }
}

// ------------------------------------------------------------------------------------------------
// application
// ------------------------------------------------------------------------------------------------
class App {
    constructor(glCanvas, ovCanvas) {
        this.canvas = glCanvas;
        const gl = this.gl = glCanvas.getContext("webgl2", {antialias: true, preserveDrawingBuffer: false});
        if (!gl) throw new Error("This browser does not support WebGL2.");
        if (!gl.getExtension("EXT_color_buffer_float"))
            throw new Error("This GPU / browser cannot render to float textures (EXT_color_buffer_float).");
        this.floatLinear = !!gl.getExtension("OES_texture_float_linear");
        this.ov = new Overlay(ovCanvas);
        this.ovCanvas = ovCanvas;

        this.pCompute = new Program(gl, FULLSCREEN_VS, COMPUTE_FS);
        this.pLine = new Program(gl, FULLSCREEN_VS, LINE_FS);
        this.pPlane = new Program(gl, FULLSCREEN_VS, PLANE_FS);
        this.pBg = new Program(gl, FULLSCREEN_VS, BG_FS);
        this.pLand = new Program(gl, LAND_VS, LAND_FS);
        this.pSpiral = new Program(gl, SPIRAL_VS, SPIRAL_FS);
        this.pLines = new Program(gl, LINES_VS, LINES_FS);
        this.vaoEmpty = gl.createVertexArray();

        const R = RES3D;
        const idx = new Uint32Array((R - 1) * (R - 1) * 6);
        let q = 0;
        for (let j = 0; j < R - 1; j++)
            for (let i = 0; i < R - 1; i++) {
                const a = j * R + i;
                idx[q++] = a; idx[q++] = a + 1; idx[q++] = a + R;
                idx[q++] = a + 1; idx[q++] = a + R + 1; idx[q++] = a + R;
            }
        this.nLandIdx = idx.length;
        this.vaoLand = gl.createVertexArray();
        gl.bindVertexArray(this.vaoLand);
        const ib = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
        gl.bindVertexArray(null);

        this.vaoLines = gl.createVertexArray();
        this.linesBuf = gl.createBuffer();
        gl.bindVertexArray(this.vaoLines);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.linesBuf);
        const lp = gl.getAttribLocation(this.pLines.p, "in_pos"), lc = gl.getAttribLocation(this.pLines.p, "in_col");
        gl.enableVertexAttribArray(lp);
        gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 28, 0);
        gl.enableVertexAttribArray(lc);
        gl.vertexAttribPointer(lc, 4, gl.FLOAT, false, 28, 12);
        gl.bindVertexArray(null);

        this.table = new PhaseTable(gl, 1);      // plane + landscape
        this.spTable = new PhaseTable(gl, 2);    // critical-line view
        this.jobs = {};
        this.landJob = this.makeJob("land", this.pCompute, R, R, false);

        this.budget = 1.5e8;     // series terms per frame (adapted to the frame rate)
        this.mode = 1;
        this.animate = false;
        this.contours = 1;
        this.grid = 1;
        this.labels = 0;
        this.info = 0;
        this.style = 1;               // colour only
        this.colorbar = 0;
        this.markers = 1;
        this.help = false;
        this.scheme = SCHEME_NAMES.indexOf(DEFAULT_SCHEME);
        this.method = 0;
        this.hmode = 0;
        this.hscale = 0.06;
        this.landView = null;
        this.message = "";
        this.messageT = 0;
        this.camLand = new OrbitCam(0.0, 0.62, 3.6, [0, 0.25, 0]);
        this.camSpiral = new OrbitCam(0.45, 0.28, 8.5, [0, 0, 0]);
        this.camPreset = 0;
        this.spN = 1; this.spT0 = 0; this.spAuto = 1;
        this.resize();
        this.resetRegion();
        this.resetSpiral();
        this.fitLandCamera();
        this.drag = null;
        this.mouse = [0, 0];         // CSS pixels
        this.pointers = new Map();
        this.tStart = performance.now() / 1000;
        this.flow = 0;
        this.animT = 0;
        this.wantShot = false;
        this.frames = 0; this.fps = 0; this.fpsT = this.tStart;
        this.readout = ""; this.readoutT = 0;
        this.onChange = null;        // callback for the toolbar
        this.installInput();
    }

    // -------------------------------------------------------------- state
    get plain() { return this.style === 3 ? 1 : 0; }
    get rsT() { return METHODS[this.method][1]; }
    get fb() { return [this.canvas.width, this.canvas.height]; }

    makeJob(name, prog, w, h, linear) {
        const old = this.jobs[name];
        if (old && old.size[0] === w && old.size[1] === h) return old;
        if (old) old.release();
        const job = new Job(this, prog, w, h, linear);
        this.jobs[name] = job;
        return job;
    }

    say(msg) {
        this.message = msg;
        this.messageT = performance.now() / 1000;
    }

    resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
        const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
        if (w !== this.canvas.width || h !== this.canvas.height) {
            this.canvas.width = this.ovCanvas.width = w;
            this.canvas.height = this.ovCanvas.height = h;
            this.regionChanged();
        }
        this.dpr = dpr;
    }

    resetRegion() {
        this.center = [0.5, 0.0];     // t symmetric around 0 (zeros come in conjugate pairs)
        this.hy = 26.0;               // the first three zero pairs, t = ±14.13, ±21.02, ±25.01
        this.stretch = 0.4;           // default: sigma axis squeezed 2.5x
        this.aspectMode = 1;          // A cycles: 0 1:1, 1 sigma squeezed, 2 t stretched
        this.regionChanged();
    }

    resetSpiral() {
        this.spC = 30.0; this.spW = 60.0;
        this.spSigma = 0.5;
        this.spAmp = 0.45;
        this.spSpeed = 1.0;
        this.unwind = 0; this.unwindCur = 0;
        this.spZeros = [];
        this.spiralChanged();
    }

    fitLandCamera() {           // aim at the centre of the surface's bounding box, far enough to see all of it
        const hf = this.half();
        const ar = hf[0] / hf[1];
        const top = 6.0 * this.hscale;
        const [w, h] = this.fb;
        const aspect = w / Math.max(h, 1);
        const halfV = Math.PI / 8, halfH = Math.atan(Math.tan(halfV) * aspect);
        const r = Math.sqrt(ar * ar + 1 + (top / 2) ** 2);
        const dist = 0.44 * r / Math.sin(Math.min(halfV, halfH));     // half the distance that fits it all in view
        const [yaw, pitch, zoom] = this.landView || [1.0, 0.6, 1.0];
        const focusX = (-2.0 - this.center[0]) / this.hy;             // look at sigma = -2 (world x = (sigma - centre) / hy)
        this.camLand.init = [yaw, pitch, dist / zoom, [focusX, top / 2, 0]];
        this.camLand.reset();
    }

    regionChanged() {
        this.regionDirty = true;
        this.changedAt = performance.now() / 1000;
    }

    spiralChanged() { this.spiralDirty = true; }

    half() {
        const [w, h] = this.fb;
        return [this.hy * this.stretch * w / Math.max(h, 1), this.hy];
    }

    zoomRegion(f, anchor) {
        anchor = anchor || this.center.slice();
        this.center = [anchor[0] + (this.center[0] - anchor[0]) * f, anchor[1] + (this.center[1] - anchor[1]) * f];
        this.hy = clamp(this.hy * f, 1e-9, 1e9);
        this.regionChanged();
    }

    cursorNdc() {
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        return [2 * this.mouse[0] / Math.max(w, 1) - 1, 1 - 2 * this.mouse[1] / Math.max(h, 1)];
    }

    cursorS() {
        const n = this.cursorNdc(), hf = this.half();
        return [this.center[0] + n[0] * hf[0], this.center[1] + n[1] * hf[1]];
    }

    gridInfo() {
        const hf = this.half(), [w, h] = this.fb;
        const step = [niceStep(2 * hf[0], Math.max(2, 7 * w / Math.max(h, 1) / 1.6)), niceStep(2 * hf[1], 7)];
        return [step, Math.round(this.center[1] / step[1]) * step[1]];
    }

    // -------------------------------------------------------------- job scheduling
    planRegion() {
        const hf = this.half();
        const sigMax = Math.abs(this.center[0]) + hf[0], tMax = Math.abs(this.center[1]) + hf[1];
        this.table.update(this.center[1], hf[1], termsNeeded(sigMax, tMax, this.rsT));
        const cost = costPerSample(sigMax, tMax, this.rsT);
        const [w, h] = this.fb;
        const sc = clamp(Math.sqrt(this.budget * 1.5 / (w * h * cost)), 0.12, 0.45);
        const lo = this.makeJob("lo", this.pCompute, Math.max(2, Math.floor(w * sc)), Math.max(2, Math.floor(h * sc)), true);
        const hi = this.makeJob("hi", this.pCompute, w, h, false);
        const common = {u_sig: [this.center[0], hf[0]], u_tt: [this.table.rel(this.center[1]), hf[1]], u_flip: this.table.flip};
        lo.start(this.table, cost, {...common, u_off: [0, 0], u_den: lo.size.slice()});
        hi.start(this.table, cost, {...common, u_off: [0, 0], u_den: [w, h]});
        this.landJob.start(this.table, cost, {...common, u_off: [0.5, 0.5], u_den: [RES3D - 1, RES3D - 1]});
        this.regionDirty = false;
    }

    planSpiral() {
        const t0 = this.spC - this.spW / 2, w = this.spW;
        const tMax = Math.max(Math.abs(t0), Math.abs(t0 + w));
        this.spTable.update(this.spC, w / 2, termsNeeded(Math.abs(this.spSigma), tMax, this.rsT));
        const n = Math.floor(clamp(w * 150 * Math.log(Math.max(tMax / TAU, 3)), 4000, 400000));
        const job = this.makeJob("line", this.pLine, 4096, Math.ceil(n / 4096), false);
        job.start(this.spTable, costPerSample(Math.abs(this.spSigma), tMax, this.rsT),
                  {u_dlo: this.spTable.rel(t0), u_w: w, u_sigma: this.spSigma, u_n: n, u_flip: this.spTable.flip});
        this.spN = n; this.spT0 = t0;
        this.spZeros = [];
        this.spiralDirty = false;
    }

    // sign changes of Hardy's Z(t) on the computed samples; also an automatic amplitude scale
    findSpiralZeros() {
        const job = this.jobs.line, n = this.spN;
        const d = job.read(0, 0, job.size[0], job.size[1]);
        const mags = [];
        for (let i = 0; i < n; i++) {
            const m = Math.hypot(d[4 * i], d[4 * i + 1]);
            if (isFinite(m)) mags.push(m);
        }
        mags.sort((a, b) => a - b);
        const p995 = mags.length ? mags[Math.min(mags.length - 1, Math.floor(0.995 * (mags.length - 1)))] : 0;
        this.spAuto = mags.length ? 2.2 / Math.max(p995, 2.2) : 1;
        this.spZeros = [];
        if (Math.abs(this.spSigma - 0.5) > 1e-9) return;
        const dt = this.spW / (n - 1);
        let zp = d[0] * Math.cos(d[2]) - d[1] * Math.sin(d[2]);
        for (let i = 1; i < n; i++) {
            const z = d[4 * i] * Math.cos(d[4 * i + 2]) - d[4 * i + 1] * Math.sin(d[4 * i + 2]);
            if (Math.sign(zp) * Math.sign(z) < 0) this.spZeros.push(this.spT0 + (i - 1 + zp / (zp - z)) * dt);
            zp = z;
        }
    }

    runJobs(now) {
        const B = this.budget;
        if (this.mode === 0 || this.mode === 1) {
            if (this.regionDirty) this.planRegion();
            if (this.mode === 0) {
                const lo = this.jobs.lo, hi = this.jobs.hi;
                const used = lo.step(B * 2, this.rsT);
                if (lo.done && (now - this.changedAt > 0.15) && used < B) hi.step(B - used, this.rsT);
            } else this.landJob.step(B * 1.5, this.rsT);
        } else {
            if (this.spiralDirty) this.planSpiral();
            const job = this.jobs.line;
            if (!job.done) {
                job.step(B * 1.5, this.rsT);
                if (job.done) this.findSpiralZeros();
            }
        }
    }

    busy() {
        let job;
        if (this.mode === 0) job = this.jobs.hi;
        else job = this.mode === 1 ? this.landJob : this.jobs.line;
        return !job || job.done ? 0 : Math.max(job.progress, 1e-3);
    }

    // -------------------------------------------------------------- input
    installInput() {
        const el = this.ovCanvas;
        el.addEventListener("contextmenu", (e) => e.preventDefault());
        el.addEventListener("pointerdown", (e) => {
            el.setPointerCapture(e.pointerId);
            this.pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
            this.mouse = [e.offsetX, e.offsetY];
            if (this.pointers.size === 1) {
                const pan = e.button === 2 || e.shiftKey;
                this.drag = pan ? "pan" : "rotate";
            } else {
                this.drag = "pinch";
                this.pinch = this.pinchState();
            }
        });
        const up = (e) => {
            this.pointers.delete(e.pointerId);
            if (this.pointers.size === 0) this.drag = null;
            else if (this.pointers.size === 1) {      // back to a one-finger drag
                this.drag = "rotate";
                this.mouse = [...this.pointers.values()][0];
            }
        };
        el.addEventListener("pointerup", up);
        el.addEventListener("pointercancel", up);
        el.addEventListener("pointermove", (e) => {
            const prev = this.mouse;
            if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
            if (this.drag === "pinch") {
                const st = this.pinchState();
                if (this.pinch && st.d > 0 && this.pinch.d > 0) this.zoomBy(this.pinch.d / st.d, st.mid);
                this.pinch = st;
                return;
            }
            this.mouse = [e.offsetX, e.offsetY];
            if (!this.drag) return;
            this.onDrag(this.mouse[0] - prev[0], this.mouse[1] - prev[1]);
        });
        el.addEventListener("wheel", (e) => {
            e.preventDefault();
            this.mouse = [e.offsetX, e.offsetY];
            const scale = e.deltaMode === 1 ? 0.12 : e.deltaMode === 2 ? 1.2 : 0.0012;
            this.zoomBy(Math.exp(clamp(e.deltaY * scale, -1, 1)), null, e.shiftKey);
        }, {passive: false});
        window.addEventListener("keydown", (e) => {
            if (e.ctrlKey || e.metaKey || e.altKey) return;
            if (this.onKey(e.key, e.shiftKey)) e.preventDefault();
        });
        window.addEventListener("resize", () => this.resize());
    }

    pinchState() {
        const p = [...this.pointers.values()];
        if (p.length < 2) return {d: 0, mid: this.mouse};
        return {d: Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]), mid: [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2]};
    }

    onDrag(dx, dy) {            // dx, dy in CSS pixels
        if (this.mode === 0) {
            const w = this.canvas.clientWidth, h = this.canvas.clientHeight, hf = this.half();
            this.center = [this.center[0] - dx * 2 * hf[0] / w, this.center[1] + dy * 2 * hf[1] / h];
            this.regionChanged();
            return;
        }
        const cam = this.mode === 1 ? this.camLand : this.camSpiral;
        if (this.drag === "rotate") {
            cam.goal = null;
            cam.yaw -= dx * 0.008;
            cam.pitch = clamp(cam.pitch + dy * 0.008, -1.55, 1.55);
        } else cam.pan(dx * this.dpr, dy * this.dpr);
    }

    zoomBy(f, at, shift = false) {     // f > 1 zooms out
        if (this.mode === 0) {
            if (at) this.mouse = at;
            this.zoomRegion(f, this.cursorS());
        } else if (this.mode === 1 && shift) this.zoomRegion(f);
        else {
            const cam = this.mode === 1 ? this.camLand : this.camSpiral;
            cam.goal = null;
            cam.dist = clamp(cam.dist * f, 0.2, 80);
        }
    }

    get fullscreen() { return !!(document.fullscreenElement || document.webkitFullscreenElement); }

    toggleFullscreen() {
        const d = document, el = d.documentElement;
        if (this.fullscreen) (d.exitFullscreen || d.webkitExitFullscreen).call(d);
        else if (el.requestFullscreen || el.webkitRequestFullscreen) {
            const r = (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
            if (r && r.catch) r.catch(() => this.say("the browser refused fullscreen"));
        }
        else this.say("this browser does not allow fullscreen pages (on iPhone: Share → Add to Home Screen)");
    }

    goto(txt) {
        const v = Number(txt);
        if (txt === null || txt.trim() === "" ) return;
        if (!isFinite(v)) { this.say(`not a number: ${txt}`); return; }
        if (this.mode === 2) {
            this.spC = v + this.spW / 2;
            this.spiralChanged();
        } else {
            this.center[1] = v;
            this.hy = Math.min(this.hy, 20);
            this.regionChanged();
        }
        this.say(`t = ${fmtNum(v)}`);
    }

    // key: KeyboardEvent.key (or a toolbar action name). returns true if handled
    onKey(key, shift = false) {
        const k = key.length === 1 ? key.toLowerCase() : key;
        const spiral = this.mode === 2;
        let handled = true;
        switch (k) {
            case "?": case "F1": this.help = !this.help; break;
            case "Escape": this.help = false; break;
            case "1": case "2": case "3": this.mode = Number(k) - 1; break;
            case "Tab": this.mode = (this.mode + (shift ? 2 : 1)) % 3; break;
            case " ": this.animate = !this.animate; break;
            case "k":
                this.scheme = (this.scheme + (shift ? -1 : 1) + SCHEME_NAMES.length) % SCHEME_NAMES.length;
                if (this.style >= 2) this.style = 1;
                this.say(`colour map ${this.scheme + 1}/${SCHEME_NAMES.length}: ${SCHEME_NAMES[this.scheme]}`);
                break;
            case "x":
                this.style = (this.style + (shift ? -1 : 1) + STYLES.length) % STYLES.length;
                this.say(`display: ${STYLES[this.style]}`);
                break;
            case "b": this.colorbar ^= 1; break;
            case "z":
                this.markers ^= 1;
                this.say(this.markers ? "zero markers on" : "zero markers off: zeros are where all colours meet");
                break;
            case "l": this.labels ^= 1; break;
            case "i": this.info ^= 1; break;
            case "c": this.contours ^= 1; break;
            case "g": this.grid ^= 1; break;
            case "t":
                setTimeout(() => this.goto(window.prompt("go to t =", "")), 0);
                break;
            case "m":
                this.method = (this.method + 1) % METHODS.length;
                this.say(`method: ${METHODS[this.method][0]}`);
                this.regionChanged();
                this.spiralChanged();
                break;
            case "s": this.wantShot = true; break;
            case "f": this.toggleFullscreen(); break;
            case "o":
                if (this.onToggleControls) {
                    const shown = this.onToggleControls();
                    this.say(shown ? "controls shown" : "controls hidden: press O to show them");
                }
                break;
            case "r":
                if (spiral) { this.resetSpiral(); this.camSpiral.reset(); this.camPreset = 0; }
                else { this.resetRegion(); this.fitLandCamera(); }
                break;
            default:
                handled = spiral ? this.onKeySpiral(k, shift) : this.onKeyRegion(k, shift);
        }
        if (handled && this.onChange) this.onChange();
        return handled;
    }

    onKeyRegion(k, shift) {
        switch (k) {
            case "a":
                this.aspectMode = (this.aspectMode + 1) % 3;
                if (this.aspectMode === 1) this.stretch = 0.4;
                else if (this.aspectMode === 2) this.hy *= 2.5;
                else { this.hy /= 2.5; this.stretch = 1.0; }
                if (this.mode === 0)
                    this.say(["aspect 1:1", "σ axis squeezed 2.5× (same t range)", "t range stretched 2.5× (same σ range)"][this.aspectMode]);
                else {      // the landscape always draws sigma and t to the same scale: only the ranges change
                    const hf = this.half();
                    this.say(`σ ∈ [${fmtNum(this.center[0] - hf[0])}, ${fmtNum(this.center[0] + hf[0])}]   ` +
                             `t ∈ [${fmtNum(this.center[1] - hf[1])}, ${fmtNum(this.center[1] + hf[1])}]`);
                }
                this.regionChanged();          // the camera stays where it is
                break;
            case "h": this.hmode ^= 1; break;
            case "=": case "+": this.hscale *= 1.2; break;
            case "-": case "_": this.hscale /= 1.2; break;
            case "n": case "p": {
                this.say("searching …");
                const t = findZero(this.center[1], k === "n" ? 1 : -1);
                if (t === null) { this.say("no zero found"); break; }
                this.center = [0.5, t];        // keep the current zoom
                this.regionChanged();
                this.say(`zero at s = 1/2 + ${fmtG(t, 10)} i`);
                break;
            }
            case "[": case "]": this.zoomRegion(k === "[" ? 1.25 : 0.8); break;
            case "ArrowLeft": case "ArrowRight": case "ArrowUp": case "ArrowDown": {
                const d = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1]}[k];
                const hf = this.half(), f = shift ? 0.5 : 0.1;
                this.center = [this.center[0] + d[0] * hf[0] * f, this.center[1] + d[1] * hf[1] * f];
                this.regionChanged();
                break;
            }
            default: return false;
        }
        return true;
    }

    onKeySpiral(k, shift) {
        switch (k) {
            case "ArrowLeft": case "ArrowRight":
                this.spC += (k === "ArrowRight" ? 1 : -1) * this.spW * (shift ? 0.5 : 0.1);
                this.spiralChanged();
                break;
            case "[": case "]":
                this.spW = clamp(this.spW * (k === "[" ? 1.25 : 0.8), 0.5, 5000);
                this.spiralChanged();
                break;
            case "ArrowUp": case "ArrowDown":
                this.spSigma = Math.round((this.spSigma + (k === "ArrowUp" ? 1 : -1) * (shift ? 0.1 : 0.02)) * 1e4) / 1e4;
                this.spiralChanged();
                break;
            case "0": this.spSigma = 0.5; this.spiralChanged(); break;
            case "=": case "+": this.spAmp *= 1.2; break;
            case "-": case "_": this.spAmp /= 1.2; break;
            case ".": case ">": this.spSpeed = Math.min(this.spSpeed * 1.5, 20); break;
            case ",": case "<": this.spSpeed = Math.max(this.spSpeed / 1.5, 0.05); break;
            case "u":
                this.unwind = 1 - this.unwind;
                this.say(this.unwind ? "unwinding: e^{iθ(t)}·ζ(1/2+it) = Z(t), Hardy's real function" : "ζ(σ+it)");
                break;
            case "v": {
                this.camPreset = (this.camPreset + 1) % CAM_PRESETS.length;
                const [name, yaw, pitch, dist] = CAM_PRESETS[this.camPreset];
                this.camSpiral.flyTo(yaw, pitch, dist);
                this.say(`camera: ${name}`);
                break;
            }
            default: return false;
        }
        return true;
    }

    // -------------------------------------------------------------- drawing helpers
    drawLines(verts, mvp, mode) {
        if (!verts.length) return;
        const gl = this.gl;
        const data = new Float32Array(verts.flat());
        gl.bindVertexArray(this.vaoLines);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.linesBuf);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        this.pLines.set({u_mvp: mvp});
        gl.drawArrays(mode === undefined ? gl.LINES : mode, 0, verts.length);
        gl.bindVertexArray(null);
    }

    static project(mvp, p, fb) {
        const v = [p[0], p[1], p[2], 1], c = [0, 0, 0, 0];
        for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) c[r] += mvp[k * 4 + r] * v[k];
        if (c[3] <= 1e-6) return null;
        const n = [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
        if (Math.abs(n[0]) > 1.1 || Math.abs(n[1]) > 1.1 || Math.abs(n[2]) > 1) return null;
        return [(n[0] + 1) / 2 * fb[0], (1 - n[1]) / 2 * fb[1]];
    }

    label3d(mvp, p, s, col = [0.85, 0.9, 1.0, 0.95], anchor = "cm", off = [0, 0]) {
        const q = App.project(mvp, p, this.fb);
        if (q) this.ov.add(q[0] + off[0], q[1] + off[1], s, col, anchor);
    }

    palUniform() { return SCHEMES[SCHEME_NAMES[this.scheme]].flat(); }

    commonShade(now) {
        return {u_time: now - this.tStart, u_flow: this.flow, u_contours: this.contours, u_grid: this.grid,
                u_pal: this.palUniform(), u_plain: this.plain, u_style: this.style, u_markers: this.markers};
    }

    // -------------------------------------------------------------- views
    renderPlane(now) {
        const gl = this.gl, [w, h] = this.fb;
        const lo = this.jobs.lo, hi = this.jobs.hi;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, w, h);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, (hi.done ? hi : lo).tex);
        const [step, originT] = this.gridInfo();
        this.pPlane.set({u_tex: 0, u_crel: [this.center[0], this.center[1] - originT], u_half: this.half(), u_res: [w, h],
                         u_step: step, u_axis_t: -originT, ...this.commonShade(now)});
        gl.bindVertexArray(this.vaoEmpty);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        if (this.labels) this.planeLabels(step);
    }

    planeLabels(step) {
        const [w, h] = this.fb, hf = this.half(), T = this.ov, m = T.h * 0.5;
        const col = [0.95, 0.97, 1.0, 0.9];
        const loT = this.center[1] - hf[1];
        for (let k = Math.ceil(loT / step[1]); k <= Math.floor((this.center[1] + hf[1]) / step[1]); k++) {
            const t = k * step[1], y = (1 - (t - loT) / (2 * hf[1])) * h;
            if (2.5 * T.h < y && y < h - 2.5 * T.h) T.add(w - m, y, fmtNum(t, step[1]), col, "rm");
        }
        const loS = this.center[0] - hf[0];
        for (let k = Math.ceil(loS / step[0]); k <= Math.floor((this.center[0] + hf[0]) / step[0]); k++) {
            const sg = k * step[0], x = (sg - loS) / (2 * hf[0]) * w;
            if (3 * m < x && x < w - 8 * m) T.add(x, h - m, fmtNum(sg, step[0]), col, "cb");
        }
        T.add(w - m, h - m, "σ →", [1, 1, 0.8, 1], "rb");
        T.add(w - m, m, "↑ t", [1, 1, 0.8, 1], "rt");
        const x = (0.5 - loS) / (2 * hf[0]) * w;
        if (x > 0 && x < w) T.add(x + m, h - 3.2 * T.h, "σ = ½", [0.5, 0.9, 1.0, 1], "lb");
    }

    renderLandscape(now) {
        const gl = this.gl, [w, h] = this.fb, R = RES3D;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, w, h);
        gl.disable(gl.DEPTH_TEST);
        this.pBg.use();
        gl.bindVertexArray(this.vaoEmpty);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.enable(gl.DEPTH_TEST);
        const mvp = this.camLand.mvp(w / h);
        const hf = this.half(), [step, originT] = this.gridInfo();
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.landJob.tex);
        this.pLand.set({u_tex: 0, u_res: R, u_hmode: this.hmode, u_crel: [this.center[0], this.center[1] - originT],
                        u_half: hf, u_mvp: mvp, u_hscale: this.hscale, u_eye: this.camLand.eye(),
                        u_step: step, u_axis_t: -originT, ...this.commonShade(now)});
        gl.bindVertexArray(this.vaoLand);
        gl.drawElements(gl.TRIANGLES, this.nLandIdx, gl.UNSIGNED_INT, 0);
        gl.bindVertexArray(null);
        // frame, ticks and a height axis
        const ar = hf[0] / hf[1], c = [0.5, 0.7, 1.0, 0.55];
        const corners = [[-ar, 0, -1], [ar, 0, -1], [ar, 0, 1], [-ar, 0, 1]];
        const verts = [];
        for (let i = 0; i < 4; i++) verts.push([...corners[i], ...c], [...corners[(i + 1) % 4], ...c]);
        const sx = (sg) => (sg - this.center[0]) / hf[1], sz = (t) => -(t - this.center[1]) / hf[1];
        const sigTicks = [], tTicks = [];
        for (let k = Math.ceil((this.center[0] - hf[0]) / step[0]); k <= Math.floor((this.center[0] + hf[0]) / step[0]); k++) sigTicks.push(k * step[0]);
        for (let k = Math.ceil((this.center[1] - hf[1]) / step[1]); k <= Math.floor((this.center[1] + hf[1]) / step[1]); k++) tTicks.push(k * step[1]);
        const tk = 0.04;
        for (const sg of sigTicks) verts.push([sx(sg), 0, 1, ...c], [sx(sg), 0, 1 + tk, ...c]);
        for (const t of tTicks) verts.push([-ar, 0, sz(t), ...c], [-ar - tk, 0, sz(t), ...c]);
        let hval, hTicks, topV;
        if (this.hmode === 0) { hval = (m) => 6 * (1 - Math.exp(-m / 6)); hTicks = [0, 1, 2, 3, 5, 10]; topV = 1e9; }
        else { hval = (lm) => 4 * Math.tanh(lm / 4); hTicks = [-6, -3, -1, 0, 1, 3, 6]; topV = 20; }
        const top = hval(topV) * this.hscale, bot = this.hmode === 0 ? 0 : -top;
        verts.push([-ar, bot, 1, ...c], [-ar, top, 1, ...c]);
        for (const v of hTicks) {
            const y = hval(v) * this.hscale;
            verts.push([-ar, y, 1, ...c], [-ar - tk, y, 1, ...c]);
        }
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        this.drawLines(verts, mvp);
        gl.disable(gl.BLEND);
        gl.disable(gl.DEPTH_TEST);
        if (this.labels) {
            for (const sg of sigTicks) this.label3d(mvp, [sx(sg), 0, 1 + 3 * tk], fmtNum(sg, step[0]));
            for (const t of tTicks) this.label3d(mvp, [-ar - 2.5 * tk, 0, sz(t)], fmtNum(t, step[1]), undefined, "rm");
            for (const v of hTicks) this.label3d(mvp, [-ar - 2 * tk, hval(v) * this.hscale, 1], fmtNum(v), undefined, "rm");
            const hc = [1, 1, 0.8, 1];
            this.label3d(mvp, [0, 0, 1.3], "σ", hc);
            this.label3d(mvp, [-ar - 0.12, 0, -1.15], "t ↑", hc);
            this.label3d(mvp, [-ar, top, 1], this.hmode === 0 ? "|ζ|" : "log|ζ|", hc, "cb", [0, -4]);
            if (Math.abs(sx(0.5)) <= ar) this.label3d(mvp, [sx(0.5), 0, -1.08], "σ = ½", [0.5, 0.9, 1.0, 1]);
        }
    }

    renderSpiral(now, dt) {
        const gl = this.gl, [w, h] = this.fb;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, w, h);
        gl.disable(gl.DEPTH_TEST);
        this.pBg.use();
        gl.bindVertexArray(this.vaoEmpty);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        if (this.animate) {
            this.animT += dt * this.spSpeed;
            if (!this.camSpiral.goal && !this.drag) this.camSpiral.yaw += 0.0015 * this.spSpeed;
        }
        this.unwindCur += (this.unwind - this.unwindCur) * (1 - Math.exp(-dt * 2.5));
        const L = 9.0, k = this.spAmp * this.spAuto;
        const mvp = this.camSpiral.mvp(w / h);
        const t0 = this.spT0, W = this.spW;
        const X = (t) => ((t - t0) / W - 0.5) * L;
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        const ac = [0.6, 0.7, 1.0, 0.6];
        const verts = [[-L / 2, 0, 0, ...ac], [L / 2, 0, 0, ...ac]];
        const step = niceStep(W, 8);
        const ticks = [];
        for (let j = Math.ceil(t0 / step); j <= Math.floor((t0 + W) / step); j++) ticks.push(j * step);
        for (const t of ticks) verts.push([X(t), -0.06, 0, ...ac], [X(t), 0.06, 0, ...ac]);
        const x0 = -L / 2 - 0.3;
        verts.push([x0, 0, -k * 2, ...ac], [x0, 0, k * 2, ...ac], [x0, -k * 2, 0, ...ac], [x0, k * 2, 0, ...ac]);
        if (this.unwindCur < 0.5) {      // unit circle |zeta| = 1, for scale
            const cc = [0.6, 0.7, 1.0, 0.3 * (1 - 2 * this.unwindCur)];
            for (let i = 0; i < 64; i++) {
                const a0 = i / 64 * TAU, a1 = (i + 1) / 64 * TAU;
                verts.push([x0, k * Math.sin(a0), k * Math.cos(a0), ...cc], [x0, k * Math.sin(a1), k * Math.cos(a1), ...cc]);
            }
        }
        this.drawLines(verts, mvp);
        const job = this.jobs.line;
        if (job && job.started) {
            const n = this.spN;
            const head = 1.2 * ((this.animT / (8.0 + W / 20.0)) % 1.0);
            gl.blendFunc(gl.ONE, gl.ONE);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, job.tex);
            const common = {u_line: 0, u_head: head, u_len: L, u_k: k, u_n: n, u_done: Math.min(n, job.row * 4096),
                            u_anim: this.animate ? 1 : 0, u_mvp: mvp, u_psize: 3.0 * h / 900.0, u_unwind: this.unwindCur,
                            u_pal: this.palUniform(), u_plain: this.style >= 2 ? 1 : 0};
            gl.bindVertexArray(this.vaoEmpty);
            this.pSpiral.set({...common, u_gain: 0.9, u_points: 0});
            gl.drawArrays(gl.LINE_STRIP, 0, n);
            this.pSpiral.set({...common, u_gain: 0.3 * Math.sqrt(20000 / n), u_points: 1});
            gl.drawArrays(gl.POINTS, 0, n);
        }
        gl.disable(gl.BLEND);
        if (this.labels) {
            // skip t labels when the t axis is seen nearly end-on (they would all pile up)
            const fb = this.fb;
            const qa = App.project(mvp, [-L / 2, 0, 0], fb), qb = App.project(mvp, [L / 2, 0, 0], fb);
            const tVisible = !qa || !qb || Math.hypot(qa[0] - qb[0], qa[1] - qb[1]) > 0.25 * fb[0];
            if (tVisible) for (const t of ticks) this.label3d(mvp, [X(t), -0.14, 0], fmtNum(t, step), undefined, "ct");
            const hc = [1, 1, 0.8, 1];
            this.label3d(mvp, [x0, 0, k * 2.3], this.unwindCur > 0.5 ? "Z(t)" : "Re ζ", hc);
            this.label3d(mvp, [x0, k * 2.3, 0], "Im ζ", hc);
            this.label3d(mvp, [L / 2 + 0.3, 0, 0], "t", hc);
            if (tVisible && this.spZeros.length > 0 && this.spZeros.length <= 400) {
                const dec = clamp(Math.floor(7 - Math.log10(Math.max(Math.abs(t0) + W, 10))), 2, 5);
                const pts = [];
                for (const z of this.spZeros) {
                    const q = App.project(mvp, [X(z), 0.2, 0], fb);
                    if (q) pts.push([q, z.toFixed(dec)]);
                }
                pts.sort((a, b) => a[0][0] - b[0][0]);
                const T = this.ov, gap = T.h * 0.6, lh = T.h * 1.15;
                const last = [-1e9, -1e9, -1e9];     // up to 3 staggered rows; skip labels that would overlap
                for (const [[qx, qy], s] of pts) {
                    const wd = T.width(s);
                    for (let r = 0; r < 3; r++) {
                        if (qx - wd / 2 > last[r] + gap) {
                            last[r] = qx + wd / 2;
                            T.add(qx, qy - r * lh, s, [1.0, 0.95, 0.75, 0.95], "cb");
                            break;
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------- HUD
    hud(now) {
        const T = this.ov, lh = T.h * 1.35, m = T.h * 0.6;
        const lines = [];
        if (this.info) {
            let meth = METHODS[this.method][0];
            if (this.method === 0) meth += ` (Riemann–Siegel for t ≥ ${RS_THRESHOLD})`;
            const sch = this.style >= 2 ? STYLES[this.style] : `${SCHEME_NAMES[this.scheme]}, ${STYLES[this.style]}`;
            lines.push([`ζ explorer · ${MODES[this.mode]} · ${sch} · ${meth} · ${this.fps.toFixed(0)} fps`, [0.75, 0.85, 1.0, 1]]);
            if (this.mode === 2) {
                const t0 = this.spT0;
                let s = `ζ(${fmtNum(this.spSigma)} + it)   t ∈ [${fmtNum(t0)}, ${fmtNum(t0 + this.spW)}]`;
                if (Math.abs(this.spSigma - 0.5) < 1e-9 && this.jobs.line.done) s += `   ${this.spZeros.length} zeros`;
                s += `   speed ${fmtG(this.spSpeed, 2)}×   amplitude ${fmtG(this.spAmp / 0.45, 2)}× (auto ${fmtG(this.spAuto, 2)})`;
                if (this.unwindCur > 0.02) s += `   unwound ${(this.unwindCur * 100).toFixed(0)}%`;
                lines.push([s, [1, 1, 1, 1]]);
            } else {
                const hf = this.half();
                lines.push([`σ ∈ [${fmtNum(this.center[0] - hf[0])}, ${fmtNum(this.center[0] + hf[0])}]   ` +
                            `t ∈ [${fmtNum(this.center[1] - hf[1])}, ${fmtNum(this.center[1] + hf[1])}]`, [1, 1, 1, 1]]);
                if (this.mode === 0) {
                    if (now - this.readoutT > 0.1) { this.readoutT = now; this.readout = this.cursorReadout(); }
                    lines.push([this.readout, [1, 1, 0.85, 1]]);
                }
            }
            const p = this.busy();
            if (p) lines.push([`computing … ${(p * 100).toFixed(0)}%`, [1.0, 0.75, 0.4, 1]]);
            lines.push(["? or F1 : help", [0.6, 0.65, 0.8, 1]]);
        }
        if (this.message && now - this.messageT < 3) lines.push([this.message, [0.4, 1.0, 0.6, Math.min(1, 3 - (now - this.messageT))]]);
        if (lines.length) {
            const wmax = Math.max(...lines.map(([s]) => T.width(s)));
            T.rect(m * 0.5, m * 0.5, m * 1.5 + wmax, m * 1.2 + lh * lines.length, [0.02, 0.02, 0.06, 0.55]);
            lines.forEach(([s, c], i) => T.add(m, m + i * lh, s, c));
        }
        if (this.colorbar) this.colorbarPanel();
        if (this.help) this.helpPanel();
    }

    colorbarPanel() {          // legend explaining what colour / brightness / lines encode (top-right)
        const T = this.ov, [W] = this.fb;
        const lh = T.h * 1.3, m = T.h * 0.6, bw = T.h * 20, bh = T.h * 0.9;
        const items = [];      // ["bar", kind, lo, hi, title, [[f, label]]] | ["swatch", rgba, label, shape]
        if (this.mode === 2) {
            if (this.style >= 2) items.push(["swatch", [0.8, 0.8, 0.8, 1], "trace of ζ(σ + it)", "line"]);
            else {
                const t0 = this.spT0, st = this.spW / 3;
                items.push(["bar", 2, 0, 1, "colour = position along t (3 cycles)",
                            [0, 1, 2, 3].map((k) => [k / 3, fmtNum(t0 + k * st, niceStep(st, 10))])]);
            }
            items.push(["swatch", [1, 0.95, 0.8, 1], "zero of ζ: curve meets the t axis", "dot"]);
        } else {
            if (this.plain) {
                if (this.contours) {
                    items.push(["swatch", [0.05, 0.05, 0.05, 1], "phase lines: arg ζ = k·30°", "line"]);
                    items.push(["swatch", [0.5, 0.5, 0.5, 1], "|ζ| contours: |ζ| = 2ᵏ", "line"]);
                }
            } else {
                if (this.style === 0 || this.style === 1)
                    items.push(["bar", 0, 0, 1, `hue = arg ζ(s)   ·   ${SCHEME_NAMES[this.scheme]}  (K)`,
                                [[0, "−π"], [0.25, "−π/2"], [0.5, "0"], [0.75, "π/2"], [1, "π"]]]);
                if (this.style === 0 || this.style === 2)
                    items.push(["bar", 1, -3, 3, "brightness = |ζ(s)|  (one band per doubling)",
                                ["1/8", "1/4", "1/2", "1", "2", "4", "8"].map((v, i) => [i / 6, v])]);
            }
            items.push(["swatch", this.plain ? [0.1, 0.4, 0.85, 1] : [0.3, 0.9, 1.0, 1], "critical line  σ = ½", "line"]);
            if (this.markers) items.push(["swatch", [0, 0, 0, 1], "zero of ζ(s)   (Z: hide markers)", "dot"]);
            else items.push(["swatch", null, "zeros: where all colours meet   (Z: markers)", "none"]);
        }
        const heights = items.map((it) => it[0] === "bar" ? lh + bh + lh : lh);
        const ph = heights.reduce((a, b) => a + b, 0) + m;
        const pw = Math.max(bw, ...items.map((it) => T.width(it[0] === "bar" ? it[4] : it[2]) + 2.5 * T.h)) + 2 * m;
        let right = W - m * 0.5;
        if (this.labels && this.mode === 0) {     // keep clear of the t labels on the right edge
            const [step] = this.gridInfo(), hf = this.half();
            right -= Math.max(...[-1, 1].map((sg) => T.width(fmtNum(this.center[1] + sg * hf[1], step[1])))) + 1.5 * T.h;
        }
        const x0 = right - pw, y0 = m * 0.5;
        T.rect(x0, y0, x0 + pw, y0 + ph, [0.02, 0.02, 0.06, 0.72]);
        let y = y0 + m * 0.8;
        items.forEach((it, idx) => {
            if (it[0] === "bar") {
                const [, kind, lo, hi, title, ticks] = it;
                T.add(x0 + m, y, title, [0.75, 0.85, 1.0, 1]);
                const by = y + lh, bx = x0 + m;
                for (let i = 0; i < Math.ceil(bw); i++) {
                    const f = (i + 0.5) / bw;
                    let col;
                    if (kind === 0) col = paletteJS(this.scheme, (-Math.PI + f * TAU) / TAU);
                    else if (kind === 1) { const x = lo + f * (hi - lo); col = Array(3).fill(0.95 * (0.45 + 0.55 * (x - Math.floor(x)))); }
                    else col = paletteJS(this.scheme, f * 3).map((v) => v * 0.8 + 0.2);
                    T.rect(bx + i, by, bx + i + 1.2, by + bh, [...col, 1]);
                }
                for (const [f, lab] of ticks) {
                    const tx = bx + f * bw;
                    T.rect(tx - 0.5, by + bh, tx + 0.5, by + bh + T.h * 0.25, [0.9, 0.9, 1, 0.9]);
                    T.add(tx, by + bh + T.h * 0.25, lab, [0.95, 0.95, 1, 1], "ct");
                }
            } else {
                const [, col, label, shape] = it;
                const cy = y + T.h / 2;
                if (shape === "line") T.rect(x0 + m, cy - 1.5, x0 + m + 1.6 * T.h, cy + 1.5, col);
                else if (shape === "dot") {
                    const r = T.h * 0.32, cx = x0 + m + 0.8 * T.h;
                    T.rect(cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1, [0.6, 0.6, 0.65, 1]);
                    T.rect(cx - r, cy - r, cx + r, cy + r, col);
                }
                T.add(x0 + m + 2.2 * T.h, y, label, [0.95, 0.95, 1, 1]);
            }
            y += heights[idx];
        });
    }

    cursorReadout() {
        let s = this.cursorS();
        const job = this.jobs.hi.done ? this.jobs.hi : this.jobs.lo;
        if (!job.done) return `s = ${fmtComplex(s[0], s[1])}`;
        const n = this.cursorNdc(), [tw, th] = job.size;
        const px = Math.floor((n[0] + 1) / 2 * tw), py = Math.floor((n[1] + 1) / 2 * th);
        const hf = this.half();       // report s at the centre of the pixel that is read, so s and ζ(s) match
        s = [this.center[0] + ((px + 0.5) / tw * 2 - 1) * hf[0], this.center[1] + ((py + 0.5) / th * 2 - 1) * hf[1]];
        let out = `s = ${fmtComplex(s[0], s[1])}`;
        if (px >= 0 && px < tw && py >= 0 && py < th) {
            const d = job.read(px, py, 1, 1);
            if (d.every(isFinite)) {
                const lm = d[0], ang = Math.atan2(d[2], d[1]);
                if (lm < 700) {
                    const r = Math.exp(lm);
                    out += `    ζ(s) = ${fmtComplex(r * Math.cos(ang), r * Math.sin(ang))}    |ζ| = ${fmtG(r, 5)}`;
                } else out += `    |ζ(s)| = e^${fmtG(lm, 6)}    arg = ${ang.toFixed(5)}`;
            }
        }
        return out;
    }

    helpPanel() {
        const common = [
            ["1 2 3 / Tab", "switch view"],
            ["mouse / touch", "drag: pan / rotate    right- or shift-drag: pan camera    wheel / pinch: zoom"],
            ["space", "animate"],
            ["K / shift-K", `next / previous colour map (${SCHEME_NAMES.length}: ${SCHEME_NAMES[this.scheme]} now)`],
            ["X / shift-X", "display: colour + brightness / colour only / brightness only / lines only"],
            ["B", "colour bar / legend: what colour and brightness mean"],
            ["Z", "zero markers on / off"],
            ["L / I", "axis labels / info panel"],
            ["T", "go to a value of t (e.g. 1e6)"],
            ["M", "method: auto / Euler–Maclaurin only / Riemann–Siegel only"],
            ["S / R", "save a screenshot / reset view"],
            ["F / O", "fullscreen / hide or show the on-screen controls"],
            ["?  (or F1)", "this help            Esc   close"],
        ];
        let spec;
        if (this.mode === 2) {
            spec = [
                ["← →", "scroll the t-window (shift: faster)"],
                ["[ ]", "widen / narrow the t-window"],
                ["↑ ↓ / 0", "change σ (shift: coarse) / back to σ = ½"],
                ["- =", "amplitude"],
                [", .", "animation speed"],
                ["U", "unwind: multiply by e^{iθ(t)} -> Hardy's real Z(t)"],
                ["V", "camera: side / end-on / top / front"],
            ];
        } else {
            spec = [
                ["arrows [ ]", "move / zoom the region (shift: faster)"],
                ["N / P", "next / previous zero on the critical line"],
                ["A", this.mode === 0 ? "aspect: 1:1 → σ squeezed 2.5× → t stretched 2.5× → 1:1"
                                      : "ranges: wider σ → wider σ and longer t → narrower σ"],
                ["C / G", "contour lines / grid"],
            ];
            if (this.mode === 1) spec.push(["H", "height: |ζ|  <->  log|ζ|"], ["- =", "height scale"], ["shift-wheel", "zoom the region"]);
        }
        const rows = [["", "Controls"], ...common, ["", ""], ["", MODES[this.mode]], ...spec];
        const T = this.ov, lh = T.h * 1.45;
        const kw = Math.max(...rows.map(([k]) => T.width(k))) + T.h * 1.5;
        const tw = kw + Math.max(...rows.map(([, v]) => T.width(v)));
        const [W, H] = this.fb;
        const x0 = (W - tw) / 2, y0 = (H - lh * rows.length) / 2, pad = T.h * 1.2;
        T.rect(x0 - pad, y0 - pad, x0 + tw + pad, y0 + lh * rows.length + pad * 0.6, [0.02, 0.02, 0.07, 0.88]);
        rows.forEach(([k, v], i) => {
            const y = y0 + i * lh;
            if (!k) T.add(x0, y, v, [0.5, 0.9, 1.0, 1]);
            else {
                T.add(x0, y, k, [1, 0.9, 0.5, 1]);
                T.add(x0 + kw, y, v, [0.92, 0.94, 1, 1]);
            }
        });
    }

    // -------------------------------------------------------------- main loop
    saveShot() {
        const [w, h] = this.fb;
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        const g = c.getContext("2d");
        g.drawImage(this.canvas, 0, 0);
        g.drawImage(this.ovCanvas, 0, 0);
        const p = (x) => String(x).padStart(2, "0"), d = new Date();
        const name = `zeta_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.png`;
        c.toBlob((blob) => {
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = name;
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 5000);
        });
        this.say(`saved ${name}`);
    }

    frame() {
        const now = performance.now() / 1000;
        const dt = this.last === undefined ? 1 / 60 : Math.min(now - this.last, 0.25);
        this.last = now;
        this.resize();
        if (this.animate) {
            this.flow += dt * 0.6;
            if (this.mode === 1 && !this.drag) this.camLand.yaw += 0.003;
        }
        // adapt the per-frame GPU budget to keep the page responsive
        if (this.busy()) {
            if (dt > 0.05) this.budget = Math.max(2e7, this.budget * 0.8);
            else if (dt < 0.03) this.budget = Math.min(6e8, this.budget * 1.1);
        }
        this.camLand.update(dt);
        this.camSpiral.update(dt);
        this.runJobs(now);
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, ...this.fb);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        this.ov.begin(Math.round(13 * this.dpr));
        if (this.mode === 0) this.renderPlane(now);
        else if (this.mode === 1) this.renderLandscape(now);
        else this.renderSpiral(now, dt);
        this.hud(now);
        if (this.wantShot) { this.wantShot = false; this.saveShot(); }
        document.body.dataset.busy = this.busy() ? "1" : "0";
        this.frames++;
        if (now - this.fpsT > 0.5) { this.fps = this.frames / (now - this.fpsT); this.frames = 0; this.fpsT = now; }
    }

    run(useTimer = false) {       // useTimer: drive frames from a timer (headless browsers may not fire rAF)
        const next = useTimer ? (f) => setTimeout(f, 16) : requestAnimationFrame;
        const loop = () => { this.frame(); next(loop); };
        next(loop);
    }

    // URL parameters, e.g. ?mode=1&center=0.5,1e6&height=3&style=phase&scheme=twilight&colorbar
    applyParams(q) {
        const nums = (s) => s.split(",").map(Number);
        if (q.has("mode")) this.mode = clamp(Number(q.get("mode")) - 1, 0, 2) | 0;
        if (q.has("center")) this.center = nums(q.get("center"));
        if (q.has("height")) this.hy = Number(q.get("height"));
        if (q.has("t")) { const [a, b] = nums(q.get("t")); this.spC = (a + b) / 2; this.spW = b - a; }
        if (q.has("sigma")) this.spSigma = Number(q.get("sigma"));
        if (q.has("scheme")) {
            const i = SCHEME_NAMES.map((n) => n.toLowerCase()).indexOf(q.get("scheme").toLowerCase());
            if (i >= 0) this.scheme = i;
        }
        if (q.has("style")) this.style = Math.max(0, STYLE_KEYS.indexOf(q.get("style")));
        for (const f of ["colorbar", "labels", "info"]) if (q.has(f)) this[f] = q.get(f) === "0" ? 0 : 1;
        if (q.has("markers")) this.markers = q.get("markers") === "0" ? 0 : 1;
        if (q.has("animate")) this.animate = true;
        if (q.has("unwind")) { this.unwind = this.unwindCur = 1; }
        if (q.has("help")) this.help = true;
        if (q.has("view")) { this.landView = nums(q.get("view")); }
        this.regionChanged();
        this.spiralChanged();
        this.fitLandCamera();
    }
}
