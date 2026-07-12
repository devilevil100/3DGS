# 🟣 splat-inspect — 3D Gaussian Splatting from scratch

**A from-scratch 3D Gaussian Splatting engine in pure PyTorch** — reconstruct a 3D
scene from multi-view images and synthesize novel views. No `gsplat`, no CUDA
kernels, no external renderer: the camera math, the differentiable EWA rasterizer,
and the training loop are all implemented here. Real photos are supported via
COLMAP structure-from-motion (`pycolmap`, no `sudo`).

---

## What it does

```
photos ──(COLMAP SfM)──▶ camera poses + sparse points ──(this engine)──▶ 3D Gaussians ──▶ novel views
```

Each scene is represented as thousands of 3D Gaussians (position, anisotropic
covariance from a scale + rotation quaternion, color, opacity). We render by
**projecting** each Gaussian to a 2D Gaussian (the EWA formula
`Σ₂D = J·Σ_cam·Jᵀ`), **depth-sorting**, and **front-to-back alpha compositing** —
all differentiable — then optimize the Gaussians against the input images.

## Results (measured on an RTX 4060, 8 GB)

| Scene | Views | Novel-view PSNR (held-out) | Gaussians |
|---|---:|---:|---:|
| **Synthetic** (controlled orbit) | 24 | **35.0 dB** | ~70,000 |


Novel-view PSNR is measured on cameras **held out of training** — the honest test
that the engine learned 3D geometry rather than memorizing training pixels.
The real-photo number is lower because it's only 11 low-res images through a
simplified renderer; see *Limitations*.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch
.venv/bin/pip install numpy matplotlib opencv-python-headless pillow pycolmap

# synthetic scene (no COLMAP): generate posed views, then reconstruct
.venv/bin/python scripts/make_scene.py --views 24 --size 96 --out data/scene_synth
.venv/bin/python scripts/reconstruct.py --scene data/scene_synth --steps 800

# real photos: unposed images -> COLMAP poses -> reconstruct (one command)
.venv/bin/python scripts/reconstruct_from_images.py --images my_photos/ --downscale 144 --train
```

Each run writes: `model.pt`, `metrics.json` (PSNR/SSIM/FPS), `novel_views.png`
(GT vs reconstruction on held-out views), `point_cloud.ply` (standard 3DGS format),
and `turntable.gif` / `turntable.mp4`.

## Unified CLI

Installs a `splat-inspect` command (`pip install -e .`):

```bash
splat-inspect scene       --views 24 --out data/scene_synth   # make a synthetic scene
splat-inspect reconstruct --scene data/scene_synth --steps 800
splat-inspect capture     --images my_photos/ --downscale 144 --train   # real photos (COLMAP)
splat-inspect view        outputs/recon/model.pt               # interactive browser viewer
splat-inspect export-ply  outputs/recon/model.pt               # -> point_cloud.ply
splat-inspect turntable   outputs/recon/model.pt --frames 60   # -> turntable.gif/mp4
splat-inspect eval        outputs/recon/model.pt data/scene_synth
splat-inspect info        outputs/recon/model.pt
```

**Interactive viewer** (`splat-inspect view model.pt` → http://localhost:8000):
a self-contained web page with azimuth / elevation / distance sliders that render
novel views live from the engine — no external JS libraries.

**Standard `.ply` export** opens directly in browser splat viewers (SuperSplat,
antimatter15/splat), so anyone can spin your reconstruction with zero setup.

## Tests

```bash
pip install -e ".[dev]" && pytest -q     # 7 tests: projection, render, learning, PLY, metrics
```

## Capture guide (your own object)

- Take **20–40 photos** orbiting the object; walk fully around it, vary height.
- Keep photos **sharp** and **well-lit**; avoid motion blur and near-duplicate frames.
- The object needs **texture** — COLMAP matches features across images, so blank
  walls or shiny/reflective surfaces are hard.
- Then: `reconstruct_from_images.py --images <your_folder> --downscale 144 --train`.
- Memory: the dense renderer scales with `#Gaussians × H × W`. On 8 GB keep
  `--downscale` ≤ ~160 and `--init-points`/`--max-points` modest (see *Limitations*).

## How it works — the code map

| File | What it implements |
|---|---|
| `src/splatinspect/cameras.py` | pinhole camera: intrinsics/extrinsics, 3D→2D projection, orbit poses |
| `src/splatinspect/gaussians3d.py` | 3D Gaussian model + **EWA differentiable renderer** (quaternion covariance, depth-sorted alpha compositing, densify/prune) |
| `src/splatinspect/scene.py` | synthetic posed-scene generation + scene I/O |
| `scripts/reconstruct.py` | training loop (photometric loss, densification) + held-out novel-view eval + turntable |
| `scripts/reconstruct_from_images.py` | **COLMAP path**: unposed photos → poses + sparse cloud → scene |
| `learn/lesson1_2d_splatting.py` | the 2D warm-up that teaches the whole mechanic |

## Limitations (and the honest interview answers)

- **Not real-time.** The renderer is *dense* (`O(N·H·W)` — every Gaussian touches
  every pixel) with a global depth sort, so it runs ~15 FPS at low res and OOMs at
  high res. The real 3DGS is fast because of **tile-based rasterization** (each
  16×16 tile only processes overlapping Gaussians). That's the #1 upgrade.
- **Plain RGB color**, not view-dependent **spherical harmonics** — so shiny/
  angle-dependent surfaces lose some fidelity.
- **Simple densification** (periodic clone of high-opacity Gaussians) rather than
  the gradient-threshold clone/split scheme.
- These are deliberate scope cuts for a from-scratch build; each is a clean next step.

## Roadmap

- [x] From-scratch EWA renderer, quaternion covariance, depth-sorted compositing
- [x] Held-out novel-view eval (35 dB synthetic) + densification/pruning
- [x] COLMAP real-photo path (11-image castle reconstructed end-to-end)
- [x] Standard `.ply` export, interactive web viewer, CLI, metrics.json, tests
- [ ] **Tile-based rasterizer** → real-time + high-res (the big one)
- [ ] Spherical-harmonics view-dependent color

## References

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023
- Zwicker et al., *EWA Splatting*, 2002 (the projected-covariance math)
- Schönberger & Frahm, *Structure-from-Motion Revisited* (COLMAP), CVPR 2016
