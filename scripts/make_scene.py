"""Generate a synthetic posed multi-view scene (stand-in for a phone capture).

Renders a ground-truth object from a ring of known cameras and saves images +
poses. For real photos, use reconstruct_from_images.py (COLMAP) instead.

    python scripts/make_scene.py --views 24 --size 96 --out data/scene_synth
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from splatinspect.cameras import orbit_cameras
from splatinspect.gaussians3d import init_from_pointcloud
from splatinspect.scene import make_gt_object, render_dataset, save_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--fov", type=float, default=45.0)
    ap.add_argument("--gt-points", type=int, default=8000)
    ap.add_argument("--out", default="data/scene_synth")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"

    pts, col, dmg = make_gt_object(args.gt_points)
    gt = init_from_pointcloud(pts, col, device=device, init_scale=0.045)
    cams = orbit_cameras(args.views, radius=args.radius, target=[0, 0, 0],
                         fov_deg=args.fov, width=args.size, height=args.size, device=device)
    imgs = render_dataset(gt, cams, device=device)
    save_scene(args.out, imgs, cams, meta={"gt_points": args.gt_points})
    print(f"saved {len(imgs)} posed views ({args.size}x{args.size}) -> {args.out}")


if __name__ == "__main__":
    main()
