"""Pinhole camera model — how a 3D world point becomes a 2D pixel.

This is the bridge from Lesson 1 (2D) to real 3D splatting. A camera has:
  - intrinsics K: focal lengths (fx, fy) and principal point (cx, cy) in pixels
  - extrinsics [R | t]: a rotation + translation that map WORLD coords into
    CAMERA coords via  X_cam = R @ X_world + t

Projection (perspective divide):
    u = fx * X_cam.x / X_cam.z + cx
    v = fy * X_cam.y / X_cam.z + cy
    depth = X_cam.z         (used to sort/occlude Gaussians)

Everything is torch so it stays differentiable end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Camera:
    R: torch.Tensor          # (3,3) world->camera rotation
    t: torch.Tensor          # (3,)  world->camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to(self, device) -> "Camera":
        return Camera(self.R.to(device), self.t.to(device), self.fx, self.fy,
                      self.cx, self.cy, self.width, self.height)

    def project(self, points: torch.Tensor):
        """World points (N,3) -> (uv (N,2), depth (N,), cam_coords (N,3))."""
        cam = points @ self.R.T + self.t            # X_cam = R X + t
        z = cam[:, 2].clamp(min=1e-6)
        u = self.fx * cam[:, 0] / z + self.cx
        v = self.fy * cam[:, 1] / z + self.cy
        return torch.stack([u, v], dim=-1), cam[:, 2], cam


def look_at(eye, target, up, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Build world->camera (R, t) for a camera at `eye` looking at `target`.

    Uses the OpenCV convention: camera looks down +Z, x right, y down.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    z = target - eye
    z /= np.linalg.norm(z)                 # forward (camera +Z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)                 # right (camera +X)
    y = np.cross(z, x)                     # down  (camera +Y)

    R_cw = np.stack([x, y, z], axis=0)     # world->camera rotation
    t = -R_cw @ eye                        # world->camera translation
    return (torch.tensor(R_cw, dtype=torch.float32, device=device),
            torch.tensor(t, dtype=torch.float32, device=device))


def view_camera(center, radius: float, az_deg: float, el_deg: float,
                fov_deg: float, width: int, height: int, device="cpu") -> Camera:
    """A single camera orbiting `center` at (azimuth, elevation) — for viewers."""
    center = np.asarray(center, dtype=np.float64)
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    eye = center + radius * np.array([np.cos(el) * np.cos(az),
                                      -np.sin(el),
                                      np.cos(el) * np.sin(az)])
    R, t = look_at(eye, center, up=[0, -1, 0], device=device)
    f = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))
    return Camera(R, t, fx=f, fy=f, cx=width / 2, cy=height / 2,
                  width=width, height=height)


def orbit_cameras(n: int, radius: float, target, fov_deg: float,
                  width: int, height: int, elevation: float = 0.2,
                  device="cpu") -> list[Camera]:
    """A ring of `n` cameras orbiting `target` — our stand-in for a phone
    walk-around (real footage gets poses from COLMAP instead)."""
    target = np.asarray(target, dtype=np.float64)
    f = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_deg))   # focal length in pixels
    cams = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        eye = target + np.array([radius * np.cos(ang),
                                 -elevation * radius,
                                 radius * np.sin(ang)])
        R, t = look_at(eye, target, up=[0, -1, 0], device=device)
        cams.append(Camera(R, t, fx=f, fy=f, cx=width / 2, cy=height / 2,
                           width=width, height=height))
    return cams
