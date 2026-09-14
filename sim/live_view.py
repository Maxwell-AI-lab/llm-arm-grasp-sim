"""Live web view of the block-into-bowl sim (read-only observer).

Serves the three camera renders of a *separate* MuJoCo instance that mirrors
the running eval trial: it replays the same seed and the same action chunks
recorded by the file-handshake policy, so the page shows (a few seconds behind)
exactly what the evaluated trial is doing.

Run:  python live_view.py --port 8765 --exchange-dir agent_exchange
Open: http://127.0.0.1:8765
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

import mujoco

from yam_sim import YamBlockBowlEmbodiment, _ACT_LOW, _ACT_HIGH

_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>YAM block-into-bowl live</title>
<style>
 body{background:#14161a;color:#e8e8e8;font:14px/1.5 system-ui;margin:0;padding:16px}
 h1{font-size:16px;margin:0 0 4px} #meta{color:#9aa4b2;margin-bottom:12px}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:1400px}
 figure{margin:0} img{width:100%;border-radius:8px;border:1px solid #2a2f36}
 figcaption{color:#9aa4b2;text-align:center;margin-top:4px}
</style></head><body>
<h1>YAM block-into-bowl — live mirror</h1>
<div id="meta">loading…</div>
<div class="grid">
 <figure><img id="top" alt="top"><figcaption>top</figcaption></figure>
 <figure><img id="side" alt="side"><figcaption>side</figcaption></figure>
 <figure><img id="wrist" alt="wrist"><figcaption>wrist (base-fixed)</figcaption></figure>
</div>
<script>
async function tick(){
  const r = await fetch('/frame?t='+Date.now());
  const j = await r.json();
  document.getElementById('meta').textContent = j.meta;
  for (const cam of ['top','side','wrist'])
    document.getElementById(cam).src = 'data:image/png;base64,'+j[cam];
}
tick(); setInterval(tick, 800);
</script></body></html>"""


class Mirror:
    """Replays the evaluated trial's action chunks in a private sim instance."""

    def __init__(self, exchange_dir: Path, seed: int):
        self.emb = YamBlockBowlEmbodiment()
        self.exchange = exchange_dir
        self.seed = seed
        self.lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.emb.reset(_Scene(self.seed), seed=self.seed)
        self._queue: list[np.ndarray] = []
        self._seen_decisions = 0
        self._trial_dir = self._find_trial_dir()

    def _find_trial_dir(self):
        cands = sorted(self.exchange.glob("trial-*"))
        return cands[-1] if cands else None

    def _ingest(self):
        """Queue actions from newly completed decisions (result.txt present)."""
        if self._trial_dir is None or not self._trial_dir.exists():
            self._trial_dir = self._find_trial_dir()
            if self._trial_dir is None:
                return
        for dec in sorted(self._trial_dir.glob("decision_*")):
            n = int(dec.name.split("_")[1])
            if n <= self._seen_decisions:
                continue
            resp = dec / "response.json"
            # only mirror decisions the eval actually executed (result.txt
            # is written after toolset.execute succeeded)
            if not resp.exists() or not (dec / "result.txt").exists():
                continue
            self._seen_decisions = n
            try:
                payload = json.loads(resp.read_text())
            except Exception:
                continue
            args = payload.get("arguments", {})
            targets = args.get("targets")
            if not isinstance(targets, dict):
                continue
            labels = ("x", "y", "z", "gripper")
            cur = self.emb._eef_state()
            vec = cur.copy()
            for i, lab in enumerate(labels):
                if lab in targets:
                    vec[i] = float(targets[lab])
            action = np.clip(vec, _ACT_LOW, _ACT_HIGH)
            # mirror the official interpolation: straight line, safe speed
            self._queue.extend(_interp(cur, action))

    def step_to_realtime(self):
        with self.lock:
            self._ingest()
            budget = 6  # steps per poll (~0.6 s of sim time)
            while self._queue and budget:
                self.emb.step(_Action(self._queue.pop(0)))
                budget -= 1

    def render(self):
        with self.lock:
            images = {}
            for cam in ("top", "side", "wrist"):
                self.emb.renderer.update_scene(self.emb.data, camera=cam)
                images[cam] = self.emb.renderer.render().copy()
            st = self.emb._eef_state()
            meta = (
                f"seed {self.seed} | decisions consumed: {self._seen_decisions} | "
                f"eef x={st[0]:.3f} y={st[1]:.3f} z={st[2]:.3f} grip={st[3]:.2f} | "
                f"{self.emb._task_state()}"
            )
            return images, meta


def _interp(cur, target, hz=10.0, max_speed_frac=0.1):
    low, high = _ACT_LOW, _ACT_HIGH
    rng = high - low
    limits = max_speed_frac * rng / hz
    ratios = [abs(target[i] - cur[i]) / limits[i] for i in range(4) if limits[i] > 0]
    steps = max(1, int(np.ceil(max(ratios, default=0.0) / 0.9)))
    fr = np.linspace(1.0 / steps, 1.0, steps)
    return [np.clip(cur + (target - cur) * f, low, high) for f in fr]


class _Scene:
    def __init__(self, seed):
        self.id = f"layout-{seed}"
        self.instruction = "Pick up the red block from the table and place it inside the bowl."


class _Action:
    def __init__(self, data):
        self.data = data


class Handler(BaseHTTPRequestHandler):
    mirror: Mirror = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/frame"):
            self.mirror.step_to_realtime()
            images, meta = self.mirror.render()
            import base64
            import io
            payload = {"meta": meta}
            for cam, img in images.items():
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")
                payload[cam] = base64.b64encode(buf.getvalue()).decode()
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--exchange-dir", default="agent_exchange")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    mirror = Mirror(Path(args.exchange_dir), args.seed)
    Handler.mirror = mirror

    def pump():
        while True:
            mirror.step_to_realtime()
            time.sleep(0.1)

    threading.Thread(target=pump, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"live view: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
