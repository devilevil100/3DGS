"""Visual deliverables: an animated turntable of a trained reconstruction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .cameras import view_camera


def scene_extent(model) -> tuple[np.ndarray, float]:
    """Center + radius of the Gaussian cloud, for framing an orbit camera."""
    m = model.means.detach().cpu().numpy()
    center = m.mean(0)
    radius = float(np.linalg.norm(m - center, axis=1).mean())
    return center, radius


def render_orbit(model, n_frames: int = 60, size: int = 160, elevation: float = 12.0,
                 fov: float = 45.0, dist_mult: float = 3.2):
    """Render `n_frames` around the scene -> list of uint8 (H,W,3) frames."""
    device = model.means.device
    center, radius = scene_extent(model)
    frames = []
    with torch.no_grad():
        for k in range(n_frames):
            az = 360.0 * k / n_frames
            cam = view_camera(center, radius * dist_mult, az, elevation,
                              fov, size, size, device=device)
            img, _ = model.render(cam)
            frames.append((img.cpu().numpy() * 255).astype(np.uint8))
    return frames


def save_gif(frames, path: str | Path, fps: int = 20):
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:],
                duration=int(1000 / fps), loop=0)
    return path


def save_mp4(frames, path: str | Path, fps: int = 20) -> str | None:
    try:
        import cv2
    except ImportError:
        return None
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    return str(path)
