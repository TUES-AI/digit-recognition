# server_jax.py  — fixed & hardened
from __future__ import annotations
from flask import Flask, jsonify, request, Response
import json
import os
import time

import jax            # <-- IMPORTANT: this was missing
import jax.numpy as jnp

import network_jax as net          # your network_jax.py
import param_io_jax as pio_jax     # for loading once

app = Flask(__name__)

# ---------- Load params once, build a fast inference fn ----------
try:
    PARAMS = pio_jax.load_params()  # {'Dense_0': {'kernel':..., 'bias':...}, ...}
except FileNotFoundError:
    PARAMS = None

def _sorted_dense_names(params_dict):
    # e.g. ["Dense_0", "Dense_1", ...] sorted by numeric suffix
    names = [k for k in params_dict.keys() if k.startswith("Dense_")]
    names.sort(key=lambda s: int(s.split("_", 1)[1]))
    return names

if PARAMS is not None:
    LAYER_NAMES = _sorted_dense_names(PARAMS)

    def _logits_fn(x: jnp.ndarray) -> jnp.ndarray:
        # x: (784,)
        z = x
        for name in LAYER_NAMES[:-1]:
            W = PARAMS[name]["kernel"]   # (in, out)
            b = PARAMS[name]["bias"]     # (out,)
            z = jnp.dot(z, W) + b
            z = jax.nn.sigmoid(z)
        last = LAYER_NAMES[-1]
        z = jnp.dot(z, PARAMS[last]["kernel"]) + PARAMS[last]["bias"]  # logits (10,)
        return z

    # JIT a function that returns probabilities directly
    PREDICT_PROBS = jax.jit(lambda x: jax.nn.softmax(_logits_fn(x)))
else:
    LAYER_NAMES = []
    PREDICT_PROBS = None

# ---------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>28×28 Digit Test UI</title>
<style>
  :root {
    --bg: #ffffff;
    --panel: #ffffff;
    --ink: #111111;
    --muted: #666666;
    --border: #e0e0e0;
    --shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    font-family: system-ui, sans-serif;
    color: var(--ink);
    display: grid;
    place-items: center;
  }
  .wrap {
    display: grid;
    grid-template-columns: auto 300px;
    gap: 24px;
    align-items: center;
    width: min(900px, 96vw);
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: var(--shadow);
  }

  /* Left: canvas + toolbar */
  .draw-card {
    padding: 16px;
    display: grid;
    gap: 12px;
    justify-items: center;
  }
  .toolbar {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .btn {
    appearance: none;
    border: 1px solid var(--border);
    background: #fff;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
  }

  /* The 28x28 canvas (logical) scaled up cleanly */
  #canvas {
    width: 560px;   /* display size */
    height: 560px;
    image-rendering: pixelated;
    border-radius: 6px;
    border: 1px solid var(--border);
    display: block;
    background: #fff;
  }

  /* Right: probabilities panel */
  .prob-card {
    padding: 16px;
    display: grid;
    gap: 12px;
  }
  .prob-header {
    display: flex; align-items: baseline; justify-content: space-between;
  }
  .prob-header h2 {
    font-size: 16px; margin: 0;
  }
  .prob-list {
    display: grid; gap: 8px;
  }
  .prob-row {
    display: grid;
    grid-template-columns: 40px 1fr auto;
    gap: 8px; align-items: center;
    padding: 6px 8px;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .dot {
    width: 32px; height: 32px; border-radius: 50%;
    border: 1px solid var(--border);
    background: #fff;
  }
  .digit {
    color: var(--ink); font-weight: 600; font-size: 14px;
  }
  .prob {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    font-weight: 500;
  }
  @media (max-width: 800px) {
    .wrap { grid-template-columns: 1fr; gap: 20px; }
    #canvas { width: 90vw; height: 90vw; max-width: 400px; max-height: 400px; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="card draw-card">
      <canvas id="canvas" width="28" height="28"></canvas>
      <div class="toolbar">
        <button id="clear" class="btn">Clear</button>
      </div>
    </div>

    <div class="card prob-card">
      <div class="prob-header">
        <h2>Probabilities (0–9)</h2>
        <span class="hint">darker = higher</span>
      </div>
      <div id="probs" class="prob-list"></div>
    </div>
  </div>

<script>
(function() {
  // ----- State -----
  const W = 28, H = 28;
  const pixels = new Uint8ClampedArray(W * H); // 0..255, 0 white, 255 black (ink)
  let isDown = false;
  let lastX = -1, lastY = -1;
  let inFlight = false; // prevent fetch pileup

  // ----- DOM -----
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const clearBtn = document.getElementById('clear');
  const probsEl = document.getElementById('probs');

  // Build the probabilities panel rows
  const rows = [];
  for (let d = 0; d < 10; d++) {
    const row = document.createElement('div');
    row.className = 'prob-row';
    const dot = document.createElement('div');
    dot.className = 'dot';
    const digit = document.createElement('div');
    digit.className = 'digit';
    digit.textContent = String(d);
    const prob = document.createElement('div');
    prob.className = 'prob';
    prob.textContent = '0.0%';
    row.appendChild(dot); row.appendChild(digit); row.appendChild(prob);
    probsEl.appendChild(row);
    rows.push({dot, digit, prob});
  }

  // ----- Helpers -----
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function posToCell(evt) {
    const rect = canvas.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * W;
    const y = ((evt.clientY - rect.top)  / rect.height) * H;
    return { x, y, i: clamp(Math.floor(x), 0, W-1), j: clamp(Math.floor(y), 0, H-1) };
  }

  function drawDot(i, j, radius, value, erase) {
    const r2 = radius * radius;
    const cx = i, cy = j;
    const i0 = clamp(Math.floor(cx - radius), 0, W-1);
    const i1 = clamp(Math.ceil (cx + radius), 0, W-1);
    const j0 = clamp(Math.floor(cy - radius), 0, H-1);
    const j1 = clamp(Math.ceil (cy + radius), 0, H-1);
    for (let y = j0; y <= j1; y++) {
      for (let x = i0; x <= i1; x++) {
        const dx = x - cx, dy = y - cy;
        if (dx*dx + dy*dy <= r2) {
          const idx = y * W + x;
          if (erase) {
            // fade out quickly
            pixels[idx] = Math.max(0, pixels[idx] - value);
          } else {
            // additive ink capped at 255
            pixels[idx] = Math.min(255, pixels[idx] + value);
          }
        }
      }
    }
  }

  function drawStroke(x0, y0, x1, y1) {
    const radius = 1.5; // fixed radius in cells
    const value  = 64; // per step intensity; multiple passes build toward 255
    const erase = false; // no eraser mode
    const steps = Math.max(Math.abs(x1-x0), Math.abs(y1-y0)) * 1.5 + 1;
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const xi = x0 + (x1 - x0) * t;
      const yi = y0 + (y1 - y0) * t;
      drawDot(xi, yi, radius, value, erase);
    }
  }

  function render() {
    // Draw pixels array to the 28×28 canvas. 0 -> white, 255 -> black
    const img = ctx.getImageData(0, 0, W, H);
    const data = img.data;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const v = pixels[j*W + i];          // 0..255 (ink)
        const s = 255 - v;                   // screen shade: 255 white background, 0 black ink
        const k = (j*W + i) * 4;
        data[k+0] = s; data[k+1] = s; data[k+2] = s; data[k+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    requestAnimationFrame(render);
  }

  // ----- Events -----
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    isDown = true;
    const {x, y} = posToCell(e);
    lastX = x; lastY = y;
    drawStroke(x, y, x, y);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!isDown) return;
    const {x, y} = posToCell(e);
    drawStroke(lastX, lastY, x, y);
    lastX = x; lastY = y;
  });
  window.addEventListener('pointerup', () => { isDown = false; });

  clearBtn.addEventListener('click', () => pixels.fill(0));

  // ----- Poll the server every 10ms (avoid piling requests) -----
  async function tick() {
    if (!inFlight) {
      inFlight = true;
      // Send normalized array 0..1 to server
      const arr = new Float32Array(W*H);
      for (let i = 0; i < pixels.length; i++) {
        const norm = pixels[i] / 255.0;            // 0..1 where 1 = black ink in UI
        arr[i] = norm;                             // send as-is
      }
      try {
        const res = await fetch("/api/forward", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ pixels: Array.from(arr) }),
          cache: "no-store",
        });
        if (res.ok) {
          const { probs } = await res.json();
          updateProbs(probs);
        }
      } catch (_) {
        // ignore transient errors
      } finally {
        inFlight = false;
      }
    }
  }
  setInterval(tick, 10);

  // ----- Update the right-hand panel -----
  function updateProbs(p) {
    if (!Array.isArray(p) || p.length !== 10) return;
    for (let d = 0; d < 10; d++) {
      const prob = Math.max(0, Math.min(1, Number(p[d]) || 0));
      // Darker = higher: shade = 255 - (prob * 255)
      const shade = Math.round(255 - prob * 255);
      rows[d].dot.style.backgroundColor = `rgb(${shade}, ${shade}, ${shade})`;
      rows[d].prob.textContent = (prob * 100).toFixed(1) + "%";
    }
  }

  // Kick things off
  render();
})();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

@app.post("/api/forward")
def api_forward():
    try:
        if PARAMS is None or PREDICT_PROBS is None:
            return jsonify({"error": "Params not loaded on server start. Run train/initialize first (params_jax.pkl)."}), 500

        payload = request.get_json(force=True, silent=False) or {}
        raw = payload.get("pixels", [])
        if not isinstance(raw, list) or len(raw) != 28 * 28:
            return jsonify({"error": "Expected 'pixels' as list[784]"}), 400

        # Normalize to [0,1]
        # Accept both 0..255 and 0..1
        arr = []
        for v in raw:
            try:
                f = float(v)
            except Exception:
                f = 0.0
            arr.append(f / 255.0 if f > 1.0 else f)

        x = jnp.asarray(arr, dtype=jnp.float32)  # (784,)

        # Probabilities via JIT'd function
        probs = PREDICT_PROBS(x)                 # (10,)

        # JSON-serializable
        probs_list = [float(p) for p in probs]

        # basic sanity clamp
        probs_list = [max(0.0, min(1.0, p)) for p in probs_list]
        if len(probs_list) != 10:
            return jsonify({"error": "network must return 10 probs"}), 500

        return jsonify({"probs": probs_list})
    except Exception as e:
        # print a concise error for your console
        import traceback
        print("ERROR /api/forward:", e)
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    # Optional: allow the same port picking you had
    import socket
    port = 6969
    for p in [6969, 6970, 6971]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", p))
            port = p
            break
        except OSError:
            continue

    # Flask dev server is single-threaded; the JS polls often.
    # inFlight on the client helps, but enable threading for smoother UX.
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)