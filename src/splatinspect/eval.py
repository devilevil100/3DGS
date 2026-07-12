"""Image-quality metrics for novel-view evaluation: PSNR and SSIM."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return -10.0 * np.log10(mse + 1e-12)


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.T @ g)                         # (size,size)
    return w.expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, size: int = 11, sigma: float = 1.5) -> float:
    """Structural similarity for two (H,W,3) images in [0,1]."""
    x = a.permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
    y = b.permute(2, 0, 1).unsqueeze(0)
    w = _gaussian_window(size, sigma, a.device)
    pad = size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=3)
    mu_y = F.conv2d(y, w, padding=pad, groups=3)
    mx2, my2, mxy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = F.conv2d(x * x, w, padding=pad, groups=3) - mx2
    sy = F.conv2d(y * y, w, padding=pad, groups=3) - my2
    sxy = F.conv2d(x * y, w, padding=pad, groups=3) - mxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mxy + c1) * (2 * sxy + c2)) / ((mx2 + my2 + c1) * (sx + sy + c2))
    return float(s.mean().item())


def evaluate_views(model, cams, imgs, idxs) -> dict:
    """Mean PSNR/SSIM over a set of views (e.g. held-out cameras)."""
    ps, ss = [], []
    with torch.no_grad():
        for i in idxs:
            r, _ = model.render(cams[i])
            ps.append(psnr(r, imgs[i]))
            ss.append(ssim(r, imgs[i]))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(idxs)}
