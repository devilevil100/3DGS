"""Export trained Gaussians to the standard 3D Gaussian Splatting .ply format.

The resulting file opens in common web viewers (SuperSplat, antimatter15/splat,
PlayCanvas) — so a recruiter can spin your reconstruction in the browser with no
setup. We write the INRIA-3DGS property layout (SH degree 0):

    x y z  nx ny nz  f_dc_0..2  opacity  scale_0..2  rot_0..3

Values are stored PRE-activation exactly as viewers expect: opacity as its logit,
scales as log, rotation as a (normalized) quaternion, and color as the SH DC term
f_dc = (rgb - 0.5) / C0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .gaussians3d import GaussianModel3D

_SH_C0 = 0.28209479177387814  # 0th-order spherical-harmonic constant


def save_ply(model: GaussianModel3D, path: str | Path,
             clean: bool = True, min_opacity: float = 0.02,
             scale_outlier: float = 8.0) -> int:
    """Write the Gaussians as a 3DGS-standard binary .ply. Returns #points.

    clean=True drops export junk so web viewers look tidy: near-transparent
    Gaussians (opacity < min_opacity) and runaway "floaters" whose size exceeds
    `scale_outlier`x the median Gaussian size.
    """
    with torch.no_grad():
        xyz = model.means.detach().cpu().numpy().astype(np.float32)
        rgb = torch.sigmoid(model.color_raw).detach().cpu().numpy().astype(np.float32)
        f_dc = (rgb - 0.5) / _SH_C0
        opacity = model.opacity_raw.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        scale = model.log_scales.detach().cpu().numpy().astype(np.float32)
        q = model.quats.detach()
        q = (q / q.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

    if clean:
        gsize = np.exp(scale).mean(axis=1)                       # per-Gaussian size
        med = np.median(gsize)
        op = 1.0 / (1.0 + np.exp(-opacity[:, 0]))               # sigmoid
        keep = (op >= min_opacity) & (gsize <= med * scale_outlier)
        xyz, f_dc, opacity, scale, q = xyz[keep], f_dc[keep], opacity[keep], scale[keep], q[keep]

    n = xyz.shape[0]
    normals = np.zeros((n, 3), np.float32)
    props = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([xyz, normals, f_dc, opacity, scale, q], axis=1).astype(np.float32)

    path = Path(path)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return n


def save_pointcloud_ply(points: np.ndarray, colors: np.ndarray, path: str | Path) -> int:
    """Plain colored point cloud (ASCII) — handy for quick sanity views."""
    points = np.asarray(points, np.float32)
    colors = (np.clip(np.asarray(colors), 0, 1) * 255).astype(np.uint8)
    n = len(points)
    path = Path(path)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {n}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    return n
