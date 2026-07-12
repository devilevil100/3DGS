"""Real-capture path: a folder of UNPOSED photos -> COLMAP poses -> our scene.

This is the capstone that lets the engine run on your own phone photos. COLMAP
(via pycolmap, no sudo) solves Structure-from-Motion: it finds features, matches
them across images, and recovers each camera's pose + a sparse 3D point cloud.
We convert those into our Camera format and a point-cloud init, then hand off to
reconstruct.py (which initializes the Gaussians from the sparse points, exactly
like the real 3DGS pipeline).

    python scripts/reconstruct_from_images.py --images my_photos/ --out data/scene_real --train

Capture tips (for your own object): 20-40 photos, orbit fully around it, keep it
sharp and well-lit, vary height, avoid blurry/duplicate frames.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401


def run_sfm(image_dir: Path, work: Path):
    import pycolmap

    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    print("[SfM] extracting features (SIFT)...")
    pycolmap.extract_features(db, image_dir)
    print("[SfM] matching features (exhaustive)...")
    pycolmap.match_exhaustive(db)
    print("[SfM] incremental mapping (recovering poses)...")
    maps = pycolmap.incremental_mapping(db, image_dir, work)
    if not maps:
        raise SystemExit("COLMAP could not register the images — need more overlap/texture.")
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    print(f"[SfM] registered {rec.num_reg_images()} images, "
          f"{len(rec.points3D)} sparse points")
    return rec


def K_of(camera):
    try:
        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    except Exception:
        p = list(camera.params)
        if len(p) == 3:      # SIMPLE_PINHOLE / SIMPLE_RADIAL: f, cx, cy
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]   # PINHOLE: fx, fy, cx, cy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/scene_real")
    ap.add_argument("--downscale", type=int, default=0,
                    help="target max image dimension in px (0 = keep original)")
    ap.add_argument("--train", action="store_true", help="run reconstruct.py after")
    args = ap.parse_args()

    from PIL import Image
    from splatinspect.cameras import Camera
    from splatinspect.scene import save_scene
    import torch

    image_dir = Path(args.images)
    out = Path(args.out)
    rec = run_sfm(image_dir, out / "colmap")

    cams, imgs = [], []
    for img_id, im in rec.images.items():
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = K_of(cam)
        W, H = cam.width, cam.height
        pil = Image.open(image_dir / im.name).convert("RGB")
        s = 1.0
        if args.downscale and max(W, H) > args.downscale:
            s = args.downscale / max(W, H)
            W, H = int(W * s), int(H * s)
            pil = pil.resize((W, H))
            fx, fy, cx, cy = fx * s, fy * s, cx * s, cy * s
        cfw = im.cam_from_world
        pose = cfw() if callable(cfw) else cfw    # method in some pycolmap versions
        R = np.asarray(pose.rotation.matrix(), dtype=np.float32)
        t = np.asarray(pose.translation, dtype=np.float32)
        cams.append(Camera(torch.tensor(R), torch.tensor(t), fx, fy, cx, cy, W, H))
        imgs.append(np.asarray(pil, dtype=np.uint8))

    save_scene(out, imgs, cams, meta={"source": "colmap", "images": str(image_dir)})

    # sparse point cloud -> Gaussian initialization (means + colors)
    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0
    np.savez(out / "points.npz", means=xyz, colors=rgb)
    print(f"saved scene ({len(cams)} views) + {len(xyz)} init points -> {out}")

    if args.train:
        print("\n[train] launching reconstruct.py on the COLMAP scene...")
        subprocess.run([sys.executable, str(Path(__file__).parent / "reconstruct.py"),
                        "--scene", str(out), "--out", "outputs/recon_real"], check=False)


if __name__ == "__main__":
    main()
