# ζ explorer in the browser

A WebGL2 port of the desktop viewer. ζ is computed on the GPU with the same shaders (Euler–Maclaurin,
Riemann–Siegel, the CPU phase table), so the pictures and the accuracy match the Python version. It has
the same three views, display styles, colour maps, legend, labels and keyboard controls, plus mouse,
touch (drag, pinch) and a small toolbar.

## Running it

Live version: **https://delaossa.github.io/zeta/** (published from `main` by `.github/workflows/pages.yml`).

To run a local copy, open `web/index.html` in a recent browser (Chrome, Edge, Firefox,
Safari 15+); no build step or server is needed. If your browser blocks local files, serve the folder instead:

```bash
python3 -m http.server -d web 8000      # then open http://localhost:8000
```

It needs WebGL2 with float render targets (`EXT_color_buffer_float`), which almost every desktop browser
has. Phones and tablets may work but are slower.

Press `?` (or `F1`) for the controls. `S` downloads a PNG of the current view.

## URL parameters

The starting state can be set in the address, for example
`index.html?mode=1&center=0.5,1e6&height=3&colorbar`.

| Parameter | Meaning |
|---|---|
| `mode=1\|2\|3` | view: complex plane, landscape (default), critical line |
| `center=σ,t` / `height=h` | centre and half-height (in t) of the region, views 1 and 2 |
| `t=t0,t1` / `sigma=σ` | t-window and σ for view 3 |
| `style=full\|phase\|modulus\|lines` | display style |
| `scheme=name` | colour map, e.g. `twilight` |
| `colorbar`, `labels`, `info` | start with these panels on |
| `markers=0` | hide the zero markers |
| `view=yaw,pitch,zoom` | landscape camera |
| `animate`, `unwind`, `help` | start animated / unwound (view 3) / with the help open |
| `nobar` | hide the toolbar |

## Accuracy test

`test.html` evaluates ζ on the GPU at the same points as `../test_zeta_gpu.py` and compares with mpmath
values stored in `test-ref.js` (regenerate with `python web/make_test_ref.py`). Open it in a browser; it
should end with PASS. Relative error is about 10⁻⁶ up to t = 10⁸ and about 10⁻⁵ at t = 10¹⁰.

## Files

| File | Contents |
|---|---|
| `index.html` | the page: canvases, toolbar, start-up |
| `zeta-app.js` | the application: views, camera, progressive computation, panels, input |
| `zeta-shaders.js` | colour maps and the drawing shaders |
| `zeta-core.js` | ζ in GLSL, the phase table, and a CPU Hardy Z(t) for jumping between zeros |
| `test.html`, `test-ref.js`, `make_test_ref.py` | the accuracy test |
