"""Reconstruct a 3D Gaussian scene from posed images (train the splats).

Holds out some cameras for NOVEL-VIEW evaluation (the honest test that the model
learned 3D geometry, not just memorized training pixels), reports PSNR, and
renders a 360 turntable.

    python scripts/reconstruct.py --scene data/scene_synth --steps 800
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from splatinspect.gaussians3d import GaussianModel3D
from splatinspect.scene import load_scene


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return -10.0 * np.log10(mse + 1e-12)


def random_init(n: int, scale: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Random point cloud in a ball + grey colors (no cheating from GT)."""
    g = torch.Generator().manual_seed(0)
    v = torch.randn(n, 3, generator=g)
    v = v / v.norm(dim=1, keepdim=True) * (torch.rand(n, 1, generator=g) ** (1 / 3)) * scale
    col = torch.full((n, 3), 0.5)
    return v.to(device), col.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/scene_synth")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--init-points", type=int, default=4000)
    ap.add_argument("--max-points", type=int, default=9000)
    ap.add_argument("--holdout", type=int, default=4, help="views reserved for novel-view eval")
    ap.add_argument("--densify-every", type=int, default=150, help="densify interval (steps)")
    ap.add_argument("--densify-frac", type=float, default=0.15, help="fraction of Gaussians cloned per densify")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default="outputs/recon")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    imgs, cams, meta = load_scene(args.scene, device=device)
    n = len(imgs)
    # deterministic held-out split (every k-th view) — tests generalization to new angles
    hold = set(range(0, n, max(1, n // args.holdout))) if args.holdout else set()
    hold = set(list(hold)[:args.holdout])
    train_idx = [i for i in range(n) if i not in hold]
    val_idx = sorted(hold)
    print(f"device={device}  {n} views -> {len(train_idx)} train / {len(val_idx)} held-out")

    pts_file = Path(args.scene) / "points.npz"
    if pts_file.exists():
        d = np.load(pts_file)
        m, c = d["means"], d["colors"]
        if len(m) > args.init_points:                     # subsample dense SfM clouds
            sel = np.random.default_rng(0).choice(len(m), args.init_points, replace=False)
            m, c = m[sel], c[sel]
        means = torch.tensor(m, dtype=torch.float32, device=device)
        colors = torch.tensor(c, dtype=torch.float32, device=device).clamp(0, 1)
        # scale init blob size to the scene extent (COLMAP scenes have arbitrary scale)
        extent = float(np.linalg.norm(m - m.mean(0), axis=1).mean())
        init_scale = max(extent * 0.02, 1e-3)
        print(f"init from COLMAP cloud: {len(means)} pts, extent~{extent:.2f}, blob={init_scale:.3f}")
        model = GaussianModel3D(means, colors, init_scale=init_scale).to(device)
    else:
        means, colors = random_init(args.init_points, scale=1.2, device=device)
        model = GaussianModel3D(means, colors, init_scale=0.08).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = int(rng.choice(train_idx))
        render, _ = model.render(cams[i])
        gt = imgs[i]
        loss = torch.abs(render - gt).mean() + torch.mean((render - gt) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()

        # simple densify/prune schedule
        if step % args.densify_every == 0 and step < args.steps * 0.8:
            model.prune(0.02)
            model.clone_large(args.max_points, frac=args.densify_frac)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)  # reset for new params

        if step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                tr = np.mean([psnr(model.render(cams[i])[0], imgs[i]) for i in train_idx[:6]])
                va = (np.mean([psnr(model.render(cams[i])[0], imgs[i]) for i in val_idx])
                      if val_idx else float("nan"))
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"train_PSNR={tr:5.2f}  novel_view_PSNR={va:5.2f}  N={model.num}")

    # ---- final metrics ----
    with torch.no_grad():
        val_psnr = float(np.mean([psnr(model.render(cams[i])[0], imgs[i]) for i in val_idx])) \
            if val_idx else float("nan")
        # render-latency benchmark (novel view)
        cam0 = cams[val_idx[0]] if val_idx else cams[0]
        for _ in range(3):
            model.render(cam0)                       # warmup
        if device == "cuda":
            torch.cuda.synchronize()
        tb = time.time()
        reps = 20
        for _ in range(reps):
            model.render(cam0)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - tb) / reps * 1000

    from splatinspect.eval import ssim as ssim_fn
    from splatinspect.io_ply import save_ply
    from splatinspect.viz import render_orbit, save_gif, save_mp4
    with torch.no_grad():
        val_ssim = float(np.mean([ssim_fn(model.render(cams[i])[0], imgs[i])
                                  for i in val_idx])) if val_idx else float("nan")

    out = Path(args.out); (out).mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "num": model.num}, out / "model.pt")
    _save_grid(model, cams, imgs, val_idx, out / "novel_views.png")
    save_ply(model, out / "point_cloud.ply")                     # standard 3DGS .ply
    frames = render_orbit(model, n_frames=48, size=min(cam0.width, 160))
    save_gif(frames, out / "turntable.gif")
    save_mp4(frames, out / "turntable.mp4")

    metrics = {
        "gaussians": model.num,
        "novel_view_psnr": round(val_psnr, 3),
        "novel_view_ssim": round(val_ssim, 4),
        "render_ms": round(ms, 2), "fps": round(1000 / ms, 1),
        "resolution": [cam0.width, cam0.height],
        "steps": args.steps, "train_seconds": round(time.time() - t0, 1),
        "device": device, "n_views": len(imgs), "n_holdout": len(val_idx),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n================ RESULTS ================")
    print(f"Gaussians:            {model.num}")
    print(f"Novel-view PSNR/SSIM: {val_psnr:.2f} dB / {val_ssim:.3f}   (held-out cameras)")
    print(f"Render latency:       {ms:.1f} ms/frame ({1000/ms:.0f} FPS) at "
          f"{cam0.width}x{cam0.height} on {device}")
    print(f"Total train time:     {time.time()-t0:.1f} s for {args.steps} steps")
    print(f"saved -> {out}/ : model.pt, metrics.json, novel_views.png, "
          f"point_cloud.ply, turntable.gif/mp4")


def _save_grid(model, cams, imgs, idxs, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not idxs:
        return
    fig, ax = plt.subplots(2, len(idxs), figsize=(3 * len(idxs), 6))
    ax = np.atleast_2d(ax)
    with torch.no_grad():
        for j, i in enumerate(idxs):
            r = model.render(cams[i])[0].cpu().numpy()
            ax[0, j].imshow(imgs[i].cpu().numpy()); ax[0, j].set_title(f"GT view {i}"); ax[0, j].axis("off")
            ax[1, j].imshow(r); ax[1, j].set_title("reconstruction"); ax[1, j].axis("off")
    fig.suptitle("Novel-view synthesis (held-out cameras the model never trained on)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _turntable(model, cams, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idxs = np.linspace(0, len(cams) - 1, 6).astype(int)
    fig, ax = plt.subplots(1, len(idxs), figsize=(3 * len(idxs), 3))
    with torch.no_grad():
        for j, i in enumerate(idxs):
            ax[j].imshow(model.render(cams[i])[0].cpu().numpy()); ax[j].axis("off")
    fig.suptitle("Turntable — reconstructed 3D scene rendered from around the orbit")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
