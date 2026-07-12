"""
LESSON 1 — 2D Gaussian Splatting from scratch.

GOAL: fit a target image using a few hundred 2D "Gaussian blobs". We start them
as random colored smudges and let gradient descent move/stretch/recolor them
until they reconstruct the image. This is the ENTIRE optimization mechanic of real
3D Gaussian Splatting — just in 2D, so there is no camera math and no CUDA yet.

WHAT A GAUSSIAN IS HERE (each blob has these learnable parameters):
  - mean (x, y)      : where the blob sits on the image        (2 numbers)
  - scale (sx, sy)   : how wide/tall it is                      (2 numbers)
  - rotation theta   : how it's tilted                          (1 number)
  - color (r, g, b)  : its color                                (3 numbers)
  - opacity          : how strongly it contributes             (1 number)

KEY IDEA (same as real 3DGS): the renderer is DIFFERENTIABLE. We draw the image
from the blobs with pure math (no if-statements per pixel), compute how wrong the
drawing is (MSE vs target), and back-propagate that error into every blob's
parameters. You already know this loop from RNNs: params -> forward -> loss ->
loss.backward() -> optimizer.step(). The only new thing is the "forward" is a
renderer instead of a sequence model.

ACTIVATIONS (why we store raw params and transform them) — real 3DGS does this too:
  - scales are stored as log(scale) so exp() keeps them positive
  - opacity/color stored as raw logits so sigmoid() keeps them in [0, 1]
This lets the optimizer roam freely in unconstrained space while the values stay valid.

Run:
  <venv>/bin/python learn/lesson1_2d_splatting.py --gaussians 400 --steps 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def make_target(size: int, device) -> torch.Tensor:
    """A simple synthetic scene (gradient sky + sun + hills) to reconstruct."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    for y in range(size):  # vertical gradient sky
        t = y / size
        d.line([(0, y), (size, y)],
               fill=(int(60 + 120 * t), int(120 + 80 * t), int(220 - 60 * t)))
    d.ellipse([size * 0.62, size * 0.12, size * 0.86, size * 0.36], fill=(255, 214, 60))  # sun
    d.polygon([(0, size), (size * 0.35, size * 0.55), (size * 0.7, size)], fill=(34, 139, 34))  # hill
    d.polygon([(size * 0.45, size), (size * 0.78, size * 0.62), (size, size)], fill=(60, 160, 70))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)  # (H, W, 3)


class GaussianImage:
    """A bag of N 2D Gaussians plus a differentiable renderer."""

    def __init__(self, n: int, size: int, device):
        self.size = size
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(0)
        # raw (pre-activation) parameters — all leaf tensors we optimize
        self.mean = torch.rand(n, 2, generator=g).to(device).requires_grad_()
        self.log_scale = (torch.log(torch.rand(n, 2, generator=g) * 0.04 + 0.02)
                          ).to(device).requires_grad_()
        self.rot = torch.zeros(n, device=device).requires_grad_()
        self.color_raw = (torch.randn(n, 3, generator=g) * 0.5).to(device).requires_grad_()
        self.opacity_raw = torch.zeros(n, device=device).requires_grad_()
        # fixed pixel-coordinate grid, normalized to [0, 1]
        ys, xs = torch.meshgrid(torch.linspace(0, 1, size, device=device),
                                torch.linspace(0, 1, size, device=device), indexing="ij")
        self.pix = torch.stack([xs, ys], dim=-1)  # (H, W, 2)

    def params(self):
        return [self.mean, self.log_scale, self.rot, self.color_raw, self.opacity_raw]

    def render(self) -> torch.Tensor:
        N = self.mean.shape[0]
        H = W = self.size
        scales = torch.exp(self.log_scale)              # (N,2) positive  <- activation
        cos, sin = torch.cos(self.rot), torch.sin(self.rot)

        # vector from each Gaussian center to every pixel
        d = self.pix.view(1, H, W, 2) - self.mean.view(N, 1, 1, 2)  # (N,H,W,2)
        dx, dy = d[..., 0], d[..., 1]
        # rotate into each Gaussian's own frame (R^T d)
        dxr = cos.view(N, 1, 1) * dx + sin.view(N, 1, 1) * dy
        dyr = -sin.view(N, 1, 1) * dx + cos.view(N, 1, 1) * dy
        inv_s2 = 1.0 / (scales ** 2 + 1e-9)             # (N,2)  inverse covariance (diagonal in-frame)
        # Mahalanobis distance^2 -> Gaussian falloff exp(-0.5 * d^2)
        power = -0.5 * (dxr ** 2 * inv_s2[:, 0].view(N, 1, 1)
                        + dyr ** 2 * inv_s2[:, 1].view(N, 1, 1))
        g = torch.exp(power)                            # (N,H,W) each blob's footprint
        weight = torch.sigmoid(self.opacity_raw).view(N, 1, 1) * g  # (N,H,W)
        colors = torch.sigmoid(self.color_raw)          # (N,3) in [0,1]

        # Blend: normalized weighted sum of colors (order-independent).
        # NOTE: real 3DGS instead SORTS blobs by depth and does front-to-back
        # ALPHA COMPOSITING, so nearer blobs occlude farther ones. In 2D there's
        # no depth, so this simpler blend is enough — remember the difference,
        # it's a classic interview question.
        num = (weight.unsqueeze(-1) * colors.view(N, 1, 1, 3)).sum(0)  # (H,W,3)
        den = weight.sum(0).unsqueeze(-1) + 1e-6
        return (num / den).clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", type=int, default=400)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out", default="outputs/lesson1.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = make_target(args.size, device)
    model = GaussianImage(args.gaussians, args.size, device)
    opt = torch.optim.Adam(model.params(), lr=args.lr)
    print(f"device={device}  {args.gaussians} Gaussians  {args.steps} steps")

    snapshots = {}
    snap_at = {0, args.steps // 8, args.steps // 2, args.steps - 1}
    for step in range(args.steps):
        rendered = model.render()
        loss = torch.mean((rendered - target) ** 2)      # the same MSE idea as any regression
        opt.zero_grad()
        loss.backward()                                  # gradients flow into every blob
        opt.step()
        if step in snap_at:
            snapshots[step] = model.render().detach().cpu().numpy()
        if step % 50 == 0 or step == args.steps - 1:
            psnr = -10 * np.log10(loss.item() + 1e-12)
            print(f"  step {step:4d}  loss={loss.item():.5f}  PSNR={psnr:5.2f} dB")

    # save a filmstrip: target + snapshots over training
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(snapshots)
    fig, ax = plt.subplots(1, len(keys) + 1, figsize=(3 * (len(keys) + 1), 3))
    ax[0].imshow(target.cpu().numpy()); ax[0].set_title("TARGET"); ax[0].axis("off")
    for i, k in enumerate(keys):
        ax[i + 1].imshow(snapshots[k]); ax[i + 1].set_title(f"step {k}"); ax[i + 1].axis("off")
    fig.suptitle(f"Lesson 1 — {args.gaussians} 2D Gaussians learning to draw the image")
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nsaved filmstrip -> {out}")


if __name__ == "__main__":
    main()
