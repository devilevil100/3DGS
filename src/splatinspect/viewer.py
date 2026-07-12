"""Interactive browser viewer — orbit/zoom a trained reconstruction live.

Server-side rendering: the page's sliders (azimuth / elevation / distance) fetch
freshly rendered frames from THIS engine, so the demo shows your own renderer
working in real time. Self-contained (no external JS libraries).

    from splatinspect.viewer import serve
    serve("outputs/recon/model.pt", port=8000)   # open http://localhost:8000
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from .cameras import view_camera
from .gaussians3d import load_model
from .viz import scene_extent


def build_app(model_path: str, size: int = 220, fov: float = 45.0):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_model(model_path, device)
    except RuntimeError:
        device = "cpu"
        model = load_model(model_path, "cpu")
    center, radius = scene_extent(model)
    app = FastAPI(title="splat-inspect viewer")

    def render_jpeg(az, el, dist):
        cam = view_camera(center, radius * dist, az, el, fov, size, size, device=device)
        with torch.no_grad():
            img, _ = model.render(cam)
        arr = (img.cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    @app.get("/render")
    def render(az: float = 0.0, el: float = 12.0, dist: float = 3.2):
        return Response(render_jpeg(az, el, dist), media_type="image/jpeg")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return f"""
<html><head><title>splat-inspect viewer</title><style>
body{{background:#0e0e12;color:#ddd;font-family:system-ui;text-align:center;margin:0;padding:18px}}
img{{width:min(80vw,560px);border-radius:10px;background:#000;box-shadow:0 8px 30px #0008}}
.ctl{{max-width:560px;margin:14px auto;display:grid;grid-template-columns:90px 1fr 54px;gap:10px 12px;align-items:center}}
input[type=range]{{width:100%}} h2{{font-weight:600}} code{{color:#9ad}}
</style></head><body>
<h2>🟣 splat-inspect — live 3D Gaussian reconstruction</h2>
<div style="color:#888;font-size:13px">{model.num} Gaussians · rendered on <code>{device}</code> · drag the sliders to orbit</div>
<img id="v" src="/render"/>
<div class="ctl">
  <label>azimuth</label><input id="az" type="range" min="0" max="360" value="0"><span id="azv">0&deg;</span>
  <label>elevation</label><input id="el" type="range" min="-40" max="60" value="12"><span id="elv">12&deg;</span>
  <label>distance</label><input id="di" type="range" min="1.8" max="6" step="0.1" value="3.2"><span id="div">3.2</span>
</div>
<script>
const v=document.getElementById('v');
const az=document.getElementById('az'),el=document.getElementById('el'),di=document.getElementById('di');
function upd(){{
  azv.textContent=az.value+'\\u00b0'; elv.textContent=el.value+'\\u00b0'; div.textContent=di.value;
  v.src='/render?az='+az.value+'&el='+el.value+'&dist='+di.value+'&t='+Date.now();
}}
for(const s of [az,el,di]) s.addEventListener('input',upd);
</script></body></html>"""

    return app


def serve(model_path: str, port: int = 8000, size: int = 220):
    import uvicorn
    print(f"serving viewer for {model_path} at http://localhost:{port}")
    uvicorn.run(build_app(model_path, size=size), host="0.0.0.0", port=port)
