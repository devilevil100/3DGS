"""Synthetic posed multi-view scenes — our stand-in for a phone walk-around.

We build a ground-truth 3D object (an ellipsoid "panel" with a colored surface
and a distinct DAMAGE patch), render it from a ring of known cameras, and save
the images + camera poses. Because poses are known, we skip COLMAP for the demo;
real footage plugs in via the COLMAP path (see README).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .cameras import Camera, orbit_cameras
from .gaussians3d import GaussianModel3D


def make_gt_object(n: int = 6000, seed: int = 0):
    """Points + colors on an ellipsoid, with a dark-red DAMAGE patch.

    Returns (points (n,3), colors (n,3), damage_dir (3,))."""
    rng = np.random.default_rng(seed)
    # sample directions on a sphere, squash into an ellipsoid
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    radii = np.array([0.9, 0.6, 0.9])
    pts = v * radii

    # base color: higher-frequency pattern so fine detail genuinely needs many
    # Gaussians (a smooth blob would only need a few thousand).
    f = 9.0
    col = 0.5 + 0.5 * np.stack([
        np.sin(f * pts[:, 0]) * np.cos(f * pts[:, 2]),
        np.sin(f * pts[:, 1] + 1.5) * np.cos(f * pts[:, 0]),
        np.sin(f * pts[:, 2] + 3.0) * np.cos(f * pts[:, 1]),
    ], axis=1)
    col = 0.3 + 0.55 * col  # keep away from pure black/white

    # DAMAGE: a patch around a fixed surface direction, colored dark red
    damage_dir = np.array([0.6, 0.1, 0.8]); damage_dir /= np.linalg.norm(damage_dir)
    cos = v @ damage_dir
    mask = cos > 0.92
    col[mask] = np.array([0.25, 0.02, 0.02])  # dark red "rust/crack"
    return pts.astype(np.float32), col.astype(np.float32), damage_dir.astype(np.float32)


def render_dataset(gt: GaussianModel3D, cams: list[Camera], device="cpu"):
    imgs = []
    with torch.no_grad():
        for cam in cams:
            img, _ = gt.render(cam.to(device))
            imgs.append((img.cpu().numpy() * 255).astype(np.uint8))
    return imgs


def save_scene(out_dir: str | Path, imgs, cams: list[Camera], meta: dict | None = None):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    from PIL import Image
    cam_list = []
    for i, (im, cam) in enumerate(zip(imgs, cams)):
        Image.fromarray(im).save(out / "images" / f"view_{i:03d}.png")
        cam_list.append({
            "image": f"images/view_{i:03d}.png",
            "R": cam.R.cpu().numpy().tolist(), "t": cam.t.cpu().numpy().tolist(),
            "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
            "width": cam.width, "height": cam.height,
        })
    payload = {"cameras": cam_list}
    if meta:
        payload["meta"] = meta
    (out / "cameras.json").write_text(json.dumps(payload, indent=2))


def load_scene(scene_dir: str | Path, device="cpu"):
    scene = Path(scene_dir)
    data = json.loads((scene / "cameras.json").read_text())
    from PIL import Image
    cams, imgs = [], []
    for c in data["cameras"]:
        cams.append(Camera(
            torch.tensor(c["R"], dtype=torch.float32, device=device),
            torch.tensor(c["t"], dtype=torch.float32, device=device),
            c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"]))
        arr = np.asarray(Image.open(scene / c["image"]).convert("RGB"), np.float32) / 255.0
        imgs.append(torch.tensor(arr, device=device))
    return imgs, cams, data.get("meta", {})
