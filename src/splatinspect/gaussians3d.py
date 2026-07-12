"""From-scratch 3D Gaussian Splatting — model + differentiable EWA renderer.

This is the real algorithm (not the 2D toy), in pure PyTorch:

  * Each Gaussian has a 3D mean, an anisotropic covariance (from a scale + a
    rotation quaternion), a color, and an opacity.
  * To render from a camera we PROJECT each 3D Gaussian to a 2D Gaussian on the
    image plane. The projected 2D covariance is the "EWA splatting" formula
    Sigma_2D = J * Sigma_cam * J^T, where J is the Jacobian of the perspective
    projection. This is what lets a stretched 3D blob look correct in 2D.
  * We then DEPTH-SORT the Gaussians and do front-to-back ALPHA COMPOSITING, so
    near Gaussians occlude far ones (the thing the 2D lesson could not do).
  * The whole path is differentiable, so we optimize the Gaussians against photos.

Simplifications vs the CUDA reference (all honest, all defensible in interviews):
  * global per-view depth sort instead of per-tile sorting (slower, same result
    for our scene sizes);
  * plain RGB color instead of view-dependent spherical harmonics (add-on);
  * densification/pruning is a simple periodic step (see trainer), not the full
    gradient-threshold clone/split scheme.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from .cameras import Camera


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Normalized quaternions (N,4)=(w,x,y,z) -> rotation matrices (N,3,3)."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.empty(N, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class GaussianModel3D(nn.Module):
    def __init__(self, means: torch.Tensor, colors: torch.Tensor,
                 init_scale: float = 0.03):
        super().__init__()
        n = means.shape[0]
        self.means = nn.Parameter(means.clone())
        self.log_scales = nn.Parameter(torch.full((n, 3), float(np.log(init_scale))))
        q = torch.zeros(n, 4); q[:, 0] = 1.0            # identity quaternion
        self.quats = nn.Parameter(q)
        # colors given in [0,1] -> store as logits so sigmoid keeps them valid
        self.color_raw = nn.Parameter(torch.logit(colors.clamp(1e-3, 1 - 1e-3)))
        self.opacity_raw = nn.Parameter(torch.full((n,), 0.0))  # sigmoid(0)=0.5

    @property
    def num(self) -> int:
        return self.means.shape[0]

    def covariance3d(self) -> torch.Tensor:
        # clamp guards against runaway "floater" Gaussians whose scale would
        # explode to inf and poison the covariance (inf*inf -> NaN) on some views
        scales = torch.exp(self.log_scales.clamp(max=2.0))   # (N,3) > 0
        R = quat_to_rotmat(self.quats)                  # (N,3,3)
        M = R * scales.unsqueeze(1)                     # scale the columns
        return M @ M.transpose(1, 2)                    # Sigma = M M^T  (N,3,3)

    def render(self, cam: Camera, bg: float = 1.0):
        """Differentiable render from `cam` -> (image (H,W,3), alpha (H,W))."""
        device = self.means.device
        H, W = cam.height, cam.width
        uv, depth, camc = cam.project(self.means)       # (N,2),(N,),(N,3)

        # --- project 3D covariance to 2D (EWA) ---
        Sig3d = self.covariance3d()                     # (N,3,3)
        Rc = cam.R.unsqueeze(0)                          # (1,3,3)
        Sig_cam = Rc @ Sig3d @ Rc.transpose(1, 2)       # (N,3,3)
        x, y, z = camc[:, 0], camc[:, 1], camc[:, 2].clamp(min=1e-6)
        J = torch.zeros(self.num, 2, 3, device=device)
        J[:, 0, 0] = cam.fx / z
        J[:, 0, 2] = -cam.fx * x / (z * z)
        J[:, 1, 1] = cam.fy / z
        J[:, 1, 2] = -cam.fy * y / (z * z)
        Sig2d = J @ Sig_cam @ J.transpose(1, 2)         # (N,2,2)
        Sig2d[:, 0, 0] += 0.3                            # low-pass (anti-alias) blur
        Sig2d[:, 1, 1] += 0.3

        # invert 2x2 covariance in closed form
        a, b = Sig2d[:, 0, 0], Sig2d[:, 0, 1]
        c, d = Sig2d[:, 1, 0], Sig2d[:, 1, 1]
        det = (a * d - b * c).clamp(min=1e-6)
        i00, i01, i11 = d / det, -b / det, a / det

        # --- depth-sorted front-to-back alpha compositing, in CHUNKS ---
        # A dense (N,H,W) pass OOMs at realistic Gaussian counts. Instead we sort
        # all Gaussians by depth once, then composite them near->far in chunks,
        # carrying the running transmittance T and color C between chunks. Memory
        # is bounded by chunk*H*W regardless of N, so N can be hundreds of
        # thousands. Result is identical to the single-pass version.
        opacity = torch.sigmoid(self.opacity_raw)
        colors_all = torch.sigmoid(self.color_raw)
        valid = (depth > 1e-4).float()
        order = torch.argsort(depth)                       # near -> far
        uv, i00, i01, i11 = uv[order], i00[order], i01[order], i11[order]
        opacity, colors_all, valid = opacity[order], colors_all[order], valid[order]

        ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32),
                                indexing="ij")
        chunk = max(256, min(self.num, 12_000_000 // max(1, H * W)))
        # During training, checkpoint each chunk (recompute in backward) so peak
        # memory stays ~one chunk instead of O(N) — this is what lets us train
        # 100K+ Gaussians on 8 GB.
        use_ckpt = torch.is_grad_enabled() and self.means.requires_grad

        def _chunk(T_in, uvc, a0, a1, a2, opc, colc, valc):
            dx = xs.unsqueeze(0) - uvc[:, 0].view(-1, 1, 1)
            dy = ys.unsqueeze(0) - uvc[:, 1].view(-1, 1, 1)
            power = -0.5 * (a0.view(-1, 1, 1) * dx * dx
                            + 2 * a1.view(-1, 1, 1) * dx * dy
                            + a2.view(-1, 1, 1) * dy * dy)
            a = (opc.view(-1, 1, 1) * torch.exp(power.clamp(max=0.0))).clamp(0, 0.999)
            a = a * valc.view(-1, 1, 1)
            cp = torch.cumprod(1.0 - a + 1e-7, dim=0)
            t_before = torch.cat([torch.ones(1, H, W, device=device), cp[:-1]], dim=0)
            contrib = T_in.unsqueeze(0) * t_before * a
            dC = (contrib.unsqueeze(-1) * colc.view(-1, 1, 1, 3)).sum(0)
            return dC, T_in * cp[-1]

        T = torch.ones(H, W, device=device)
        C = torch.zeros(H, W, 3, device=device)
        for s in range(0, self.num, chunk):
            e = min(s + chunk, self.num)
            a = (T, uv[s:e], i00[s:e], i01[s:e], i11[s:e],
                 opacity[s:e], colors_all[s:e], valid[s:e])
            if use_ckpt:
                dC, T = torch.utils.checkpoint.checkpoint(_chunk, *a, use_reentrant=False)
            else:
                dC, T = _chunk(*a)
            C = C + dC
        img = torch.nan_to_num(C + T.unsqueeze(-1) * bg, nan=bg, posinf=1.0, neginf=0.0)
        return img.clamp(0, 1), torch.nan_to_num(1.0 - T).clamp(0, 1)

    # -- densification / pruning helpers used by the trainer --
    def prune(self, min_opacity: float = 0.02):
        keep = torch.sigmoid(self.opacity_raw).detach() > min_opacity
        if keep.all():
            return 0
        self._reindex(keep)
        return int((~keep).sum())

    @torch.no_grad()
    def clone_large(self, max_keep: int, frac: float = 0.1):
        """Duplicate the highest-opacity Gaussians with a small jitter (a simple
        stand-in for gradient-driven densification)."""
        if self.num >= max_keep:
            return 0
        op = torch.sigmoid(self.opacity_raw)
        k = min(int(self.num * frac), max_keep - self.num)
        if k <= 0:
            return 0
        idx = torch.topk(op, k).indices
        jitter = torch.randn(k, 3, device=self.means.device) * torch.exp(self.log_scales[idx]) * 0.5
        self._append(idx, jitter)
        return k

    def _reindex(self, keep):
        for name in ["means", "log_scales", "quats", "color_raw", "opacity_raw"]:
            p = getattr(self, name)
            setattr(self, name, nn.Parameter(p.detach()[keep]))

    def _append(self, idx, jitter):
        new_means = self.means.detach()[idx] + jitter
        parts = {
            "means": torch.cat([self.means.detach(), new_means]),
            "log_scales": torch.cat([self.log_scales.detach(), self.log_scales.detach()[idx]]),
            "quats": torch.cat([self.quats.detach(), self.quats.detach()[idx]]),
            "color_raw": torch.cat([self.color_raw.detach(), self.color_raw.detach()[idx]]),
            "opacity_raw": torch.cat([self.opacity_raw.detach(), self.opacity_raw.detach()[idx]]),
        }
        for name, val in parts.items():
            setattr(self, name, nn.Parameter(val))


def init_from_pointcloud(points: np.ndarray, colors: np.ndarray, device="cpu",
                         init_scale: float = 0.03) -> GaussianModel3D:
    m = torch.tensor(points, dtype=torch.float32)
    c = torch.tensor(colors, dtype=torch.float32)
    return GaussianModel3D(m, c, init_scale=init_scale).to(device)


def load_model(path, device="cpu") -> GaussianModel3D:
    """Rebuild a trained GaussianModel3D from a saved checkpoint."""
    d = torch.load(path, map_location=device)
    state = d["state"] if "state" in d else d
    n = state["means"].shape[0]
    model = GaussianModel3D(state["means"].clone(), torch.zeros(n, 3))
    model.load_state_dict(state)
    return model.to(device)
