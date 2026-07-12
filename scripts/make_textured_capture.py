"""Make a feature-rich synthetic 'photo capture' to test the COLMAP path.

Renders the GT object as sharp colored points (not soft splats) at high-res from
a ring of cameras, saving ONLY the images (poses withheld) — so pycolmap has to
recover the geometry from scratch, exactly as it would from real phone photos.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from splatinspect.cameras import orbit_cameras
from splatinspect.scene import make_gt_object


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=30)
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--points", type=int, default=16000)
    ap.add_argument("--out", default="data/capture_tex")
    args = ap.parse_args()

    import cv2
    import torch
    from PIL import Image

    pts, _col, _ = make_gt_object(args.points)
    rng = np.random.default_rng(1)
    col = (rng.uniform(0.15, 0.95, size=(len(pts), 3)) * 255).astype(np.uint8)  # rich texture
    P = torch.tensor(pts)

    cams = orbit_cameras(args.views, radius=3.0, target=[0, 0, 0], fov_deg=50,
                         width=args.size, height=args.size)
    out = Path(args.out) / "images"
    out.mkdir(parents=True, exist_ok=True)

    for i, cam in enumerate(cams):
        uv, depth, _ = cam.project(P)
        uv = uv.numpy(); depth = depth.numpy()
        order = np.argsort(-depth)                     # far -> near (painter's)
        img = np.full((args.size, args.size, 3), 130, np.uint8)
        for j in order:
            u, v = int(round(uv[j, 0])), int(round(uv[j, 1]))
            if 0 <= u < args.size and 0 <= v < args.size and depth[j] > 0:
                cv2.circle(img, (u, v), 2, tuple(int(x) for x in col[j]), -1)
        Image.fromarray(img).save(out / f"photo_{i:03d}.png")

    print(f"saved {args.views} synthetic photos ({args.size}px) -> {out}")


if __name__ == "__main__":
    main()
