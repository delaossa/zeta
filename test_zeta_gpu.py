"""Check the GPU (GLSL float32) zeta against mpmath.

Runs the exact shader used by the viewer, offscreen, on several regions and for each
method (Euler-Maclaurin / Riemann-Siegel).   usage:  python test_zeta_gpu.py [--quick]
"""
import sys

import numpy as np
import moderngl
import mpmath

from zeta_core import FULLSCREEN_VS, COMPUTE_FS, LINE_FS, PhaseTable, terms_needed, RS_THRESHOLD

# (sigma center, sigma half, t center, t half, method)   method: "em" | "rs" | "auto"
CASES = [
    (-4.0, 8.0, 20.0, 22.0, "em"),
    (0.5, 1.5, 120.0, 20.0, "em"),
    (-25.0, 5.0, 5.0, 5.0, "em"),
    (3.0, 2.0, -30.0, 10.0, "em"),
    (0.5, 1.5, 600.0, 5.0, "em"),
    (0.5, 1.5, 600.0, 5.0, "rs"),
    (0.5, 1.5, 1000.0, 5.0, "em"),
    (0.5, 1.5, 1000.0, 5.0, "rs"),
    (0.5, 1.0, 3000.0, 3.0, "em"),
    (0.5, 1.0, 3000.0, 3.0, "rs"),
    (0.5, 1.0, 1e4, 3.0, "auto"),
    (0.5, 3.0, 1e5, 2.0, "auto"),
    (0.5, 1.0, 1e6, 1.0, "auto"),
    (0.5, 1.0, -1e6, 1.0, "auto"),
    (0.5, 1.0, 1e8, 0.5, "auto"),
    (0.5, 0.5, 1e10, 0.5, "auto"),
]
N = 12 if "--quick" in sys.argv else 20

ctx = moderngl.create_standalone_context(require=410)
prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=COMPUTE_FS)
vao = ctx.vertex_array(prog, [])
tex = ctx.texture((N, N), 4, dtype="f4")
fbo = ctx.framebuffer(color_attachments=[tex])
table = PhaseTable(ctx, 1)
mpmath.mp.dps = 20


def err_stats(gpu_lm, gpu_ph, z):
    lm, ph = float(mpmath.log(abs(z))), float(mpmath.arg(z))
    return abs(gpu_lm - lm), abs((gpu_ph - ph + np.pi) % (2 * np.pi) - np.pi)


ok = True
for sc, sh, tc, th, method in CASES:
    rs_t = {"em": 1e30, "rs": 50.0, "auto": RS_THRESHOLD}[method]
    table.tref = None
    table.update(tc, th, terms_needed(abs(sc) + sh, abs(tc) + th, rs_t))
    table.bind(prog, rs_t)
    fbo.use()
    prog["u_sig"].value = (sc, sh)
    prog["u_tt"].value = (table.rel(tc), th)
    prog["u_flip"].value = table.flip
    prog["u_off"].value, prog["u_den"].value = (0.0, 0.0), (float(N), float(N))
    vao.render(moderngl.TRIANGLES, vertices=3)
    d = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype="f4").reshape(N, N, 4)
    em, ep = [], []
    for j in range(N):
        for i in range(N):
            s = mpmath.mpc(sc + ((i + 0.5) / N * 2 - 1) * sh, mpmath.mpf(tc) + ((j + 0.5) / N * 2 - 1) * th)
            a, b = err_stats(d[j, i, 0], np.arctan2(d[j, i, 2], d[j, i, 1]), mpmath.zeta(s))
            em.append(a)
            ep.append(b)
    em, ep = np.array(em), np.array(ep)
    bad = np.mean((em > 1e-2) | (ep > 1e-2))
    print(f"{method:4s} σ={sc:+6.1f}±{sh:<4g} t={tc:<8g}±{th:<4g}  log|ζ| err med {np.median(em):.1e} max {em.max():.1e}"
          f"   arg err med {np.median(ep):.1e} max {ep.max():.1e}   >1e-2: {bad:.0%}")
    if method != "rs" or tc >= RS_THRESHOLD:
        ok &= bad < 0.02

# critical-line sampler: Hardy Z(t) = Re(e^{i theta} zeta) must match mpmath.siegelz
lprog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=LINE_FS)
lvao = ctx.vertex_array(lprog, [])
ltex = ctx.texture((4096, 1), 4, dtype="f4")
lfbo = ctx.framebuffer(color_attachments=[ltex])
for t0, w in [(0.0, 60.0), (-700.0, 1000.0), (5000.0, 10.0), (1e6, 5.0)]:   # 2nd: window crossing t = 0
    n = 200
    table.tref = None
    table.update(t0 + w / 2, w / 2, terms_needed(1.0, abs(t0) + w, RS_THRESHOLD))
    table.bind(lprog, RS_THRESHOLD)
    lfbo.use()
    for k, v in dict(u_dlo=table.rel(t0), u_w=w, u_sigma=0.5, u_n=n, u_flip=table.flip).items():
        lprog[k].value = v
    lvao.render(moderngl.TRIANGLES, vertices=3)
    d = np.frombuffer(lfbo.read(components=4, dtype="f4"), dtype="f4").reshape(-1, 4)[:n]
    Zg = d[:, 0] * np.cos(d[:, 2]) - d[:, 1] * np.sin(d[:, 2])
    ts = t0 + np.arange(n) / (n - 1) * w
    Zm = np.array([float(mpmath.siegelz(t)) for t in ts])
    e = np.abs(Zg - Zm)
    print(f"Z(t) on [{t0:g}, {t0 + w:g}]:  abs err med {np.median(e):.1e} max {e.max():.1e}  (max|Z| {np.abs(Zm).max():.2f})")
    ok &= np.median(e) < 1e-2

print("PASS" if ok else "FAIL")
