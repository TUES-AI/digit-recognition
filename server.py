from __future__ import annotations
from flask import Flask, jsonify, request, Response
import json

# Import your network exactly as you provided it in the message.
# It must live beside this app.py (same directory) and expose forward_pass(inputs).
import network as net  # <-- your network.py

app = Flask(__name__)

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>28×28 Digit Test UI</title>
<style>
  :root {
    --bg: #f7f7f8;
    --panel: #ffffff;
    --ink: #111111;
    --muted: #9aa1a8;
    --border: #e5e7eb;
    --shadow: 0 10px 25px rgba(0,0,0,0.06), 0 2px 6px rgba(0,0,0,0.04);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji;
    color: #111;
    display: grid;
    place-items: center;
  }
  .wrap {
    display: grid;
    grid-template-columns: auto 340px;
    gap: 40px;
    align-items: center;
    width: min(1080px, 96vw);
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
  }

  /* Left: canvas + toolbar */
  .draw-card {
    padding: 18px 18px 12px;
    display: grid;
    gap: 14px;
    justify-items: center;
  }
  .toolbar {
    width: 100%;
    display: grid;
    grid-template-columns: 1fr auto auto;
    gap: 12px;
    align-items: center;
  }
  .toolbar .group {
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  }
  .toolbar label {
    color: var(--muted);
    font-size: 12px;
    letter-spacing: .02em;
    text-transform: uppercase;
  }
  .btn {
    appearance: none;
    border: 1px solid var(--border);
    background: #fff;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
  }
  .btn:active { transform: translateY(1px); }
  .chk { margin-left: 6px; }

  /* The 28x28 canvas (logical) scaled up cleanly */
  #canvas {
    width: 560px;   /* display size */
    height: 560px;
    image-rendering: pixelated;
    border-radius: 10px;
    border: 1px solid var(--border);
    display: block;
    background: #fff;
  }
  .hint { font-size: 12px; color: var(--muted); }

  /* Right: probabilities panel */
  .prob-card {
    padding: 18px;
    display: grid;
    gap: 16px;
  }
  .prob-header {
    display: flex; align-items: baseline; justify-content: space-between;
  }
  .prob-header h2 {
    font-size: 16px; margin: 0; letter-spacing: .02em;
  }
  .prob-list {
    display: grid; gap: 10px;
  }
  .prob-row {
    display: grid;
    grid-template-columns: 52px 1fr auto;
    gap: 12px; align-items: center;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 10px;
  }
  .dot {
    width: 36px; height: 36px; border-radius: 50%;
    border: 1px solid var(--border);
    background: #fff;
  }
  .digit {
    color: #111; font-weight: 700; font-size: 16px;
  }
  .prob {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }
  @media (max-width: 980px) {
    .wrap { grid-template-columns: 1fr; gap: 24px; }
    #canvas { width: 88vw; height: 88vw; max-width: 560px; max-height: 560px; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="card draw-card">
      <div class="toolbar">
        <div class="group">
          <label>Brush</label>
          <input id="brush" type="range" min="1" max="6" step="1" value="3" />
          <span id="brushVal" class="hint">3 px</span>
        </div>
        <div class="group">
          <label>Eraser</label>
          <input id="eraser" class="chk" type="checkbox" />
        </div>
        <button id="clear" class="btn">Clear</button>
      </div>
      <canvas id="canvas" width="28" height="28"></canvas>
      <div class="toolbar">
        <span class="hint">Left-drag = draw · Right-drag / “Eraser” = erase · Values are 0–255</span>
        <div class="group">
          <label>Invert input to network</label>
          <input id="invert" class="chk" type="checkbox" />
        </div>
        <div class="group">
          <label>Grid</label>
          <input id="grid" class="chk" type="checkbox" checked />
        </div>
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
  const eraser = document.getElementById('eraser');
  const invertChk = document.getElementById('invert');
  const gridChk = document.getElementById('grid');
  const brush = document.getElementById('brush');
  const brushVal = document.getElementById('brushVal');
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
    const radius = parseInt(brush.value, 10) / 2; // radius in cells
    const value  = 64; // per step intensity; multiple passes build toward 255
    const erase = eraser.checked;
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

    // Optional subtle grid overlay
    if (gridChk.checked) {
      ctx.save();
      ctx.strokeStyle = "rgba(0,0,0,0.06)";
      ctx.lineWidth = 1 / Math.max(canvas.clientWidth / W, canvas.clientHeight / H);
      for (let i = 1; i < W; i++) {
        ctx.beginPath(); ctx.moveTo(i+0.5, 0); ctx.lineTo(i+0.5, H); ctx.stroke();
      }
      for (let j = 1; j < H; j++) {
        ctx.beginPath(); ctx.moveTo(0, j+0.5); ctx.lineTo(W, j+0.5); ctx.stroke();
      }
      ctx.restore();
    }

    requestAnimationFrame(render);
  }

  // ----- Events -----
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
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
  brush.addEventListener('input', () => brushVal.textContent = brush.value + " px");

  // ----- Poll the server every 10ms (avoid piling requests) -----
  async function tick() {
    if (!inFlight) {
      inFlight = true;
      // Send normalized array 0..1 to server. If "invert" is checked, flip it.
      const invert = invertChk.checked;
      const arr = new Float32Array(W*H);
      for (let i = 0; i < pixels.length; i++) {
        const norm = pixels[i] / 255.0;            // 0..1 where 1 = black ink in UI
        arr[i] = invert ? (1.0 - norm) : norm;     // often networks expect white=1, black=0; toggle as needed
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
        payload = request.get_json(force=True, silent=False) or {}
        raw = payload.get("pixels", [])
        if not isinstance(raw, list) or len(raw) != 28*28:
            return jsonify({"error": "Expected 'pixels' as list[784]"}), 400

        # Accept either 0..255 or 0..1 input; normalize to 0..1 for your network.
        inp: list[float] = []
        for v in raw:
            try:
                f = float(v)
            except Exception:
                f = 0.0
            inp.append(f / 255.0 if f > 1.0 else f)

        # Call YOUR provided forward_pass from network.py
        probs = net.forward_pass(inp)

        # Be defensive: clamp to [0,1] and cast to float
        probs = [max(0.0, min(1.0, float(x))) for x in (probs or [])]
        if len(probs) != 10:
            return jsonify({"error": "forward_pass must return 10 values"}), 500

        return jsonify({"probs": probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Runs on port 6969 as requested.
    app.run(host="0.0.0.0", port=6969, debug=False)

