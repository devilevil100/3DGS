"""Sanity tests for the engine: rendering, projection, learning, and PLY I/O.

Run: .venv/bin/python -m pytest -q   (CPU-only, fast)
"""
import numpy as np
import torch

from splatinspect.cameras import Camera, look_at, view_camera
from splatinspect.eval import psnr, ssim
from splatinspect.gaussians3d import GaussianModel3D, quat_to_rotmat
from splatinspect.io_ply import save_ply


def _tiny_model(n=64):
    torch.manual_seed(0)
    means = torch.rand(n, 3) * 0.5
    colors = torch.rand(n, 3)
    return GaussianModel3D(means, colors, init_scale=0.05)


def _cam(size=32):
    R, t = look_at([0, 0, -3], [0, 0, 0], up=[0, -1, 0])
    f = 0.5 * size / np.tan(np.deg2rad(22.5))
    return Camera(R, t, f, f, size / 2, size / 2, size, size)


def test_quat_identity_is_identity():
    q = torch.tensor([[1.0, 0, 0, 0]])
    assert torch.allclose(quat_to_rotmat(q)[0], torch.eye(3), atol=1e-6)


def test_render_shape_and_range():
    img, acc = _tiny_model().render(_cam())
    assert img.shape == (32, 32, 3)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert acc.shape == (32, 32)


def test_projection_center_point_lands_center():
    cam = _cam(64)
    uv, depth, _ = cam.project(torch.zeros(1, 3))   # origin projects to principal point
    assert abs(uv[0, 0].item() - 32) < 1e-3 and abs(uv[0, 1].item() - 32) < 1e-3
    assert depth[0].item() > 0


def test_render_is_differentiable_and_learns():
    torch.manual_seed(0)
    model = _tiny_model(80)
    cam = _cam(24)
    target = torch.zeros(24, 24, 3); target[:, :, 0] = 1.0   # solid red frame
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    first = None
    for _ in range(40):
        loss = ((model.render(cam)[0] - target) ** 2).mean()
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first          # optimization reduced the loss


def test_metrics_bounds():
    a = torch.rand(16, 16, 3)
    assert psnr(a, a) > 60                # identical -> huge PSNR
    assert abs(ssim(a, a) - 1.0) < 1e-3   # identical -> SSIM 1


def test_ply_roundtrip(tmp_path):
    p = tmp_path / "m.ply"
    n = save_ply(_tiny_model(50), p)
    assert n == 50 and p.exists()
    head = p.read_bytes()[:64].decode("ascii", "ignore")
    assert head.startswith("ply") and "binary_little_endian" in head


def test_view_camera_orbit_distance():
    cam = view_camera([0, 0, 0], radius=3.0, az_deg=90, el_deg=0,
                      fov_deg=45, width=32, height=32)
    # camera center is at distance 3 from origin
    center = -cam.R.T @ cam.t
    assert abs(float(center.norm()) - 3.0) < 1e-4
