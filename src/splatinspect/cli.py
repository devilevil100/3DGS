"""Unified command-line interface: `splat-inspect <command>`.

Commands
  scene         generate a synthetic posed scene
  reconstruct   train Gaussians on a posed scene (+ novel-view eval)
  capture       real photos -> COLMAP poses -> reconstruct
  view          launch the interactive browser viewer for a trained model
  export-ply    export a trained model to a standard 3DGS .ply
  turntable     render an orbit GIF/MP4 of a trained model
  eval          PSNR/SSIM of a trained model against a scene's views
  info          summarize a trained model
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = _REPO / "scripts"


def _run_script(name: str, argv: list[str]) -> int:
    return subprocess.run([sys.executable, str(_SCRIPTS / name), *argv]).returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="splat-inspect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scene", add_help=False)
    sub.add_parser("reconstruct", add_help=False)
    sub.add_parser("capture", add_help=False)

    pv = sub.add_parser("view"); pv.add_argument("model"); pv.add_argument("--port", type=int, default=8000)
    pe = sub.add_parser("export-ply"); pe.add_argument("model"); pe.add_argument("--out", default=None)
    pt = sub.add_parser("turntable"); pt.add_argument("model"); pt.add_argument("--out", default=None)
    pt.add_argument("--frames", type=int, default=60); pt.add_argument("--size", type=int, default=160)
    pl = sub.add_parser("eval"); pl.add_argument("model"); pl.add_argument("scene")
    pi = sub.add_parser("info"); pi.add_argument("model")

    # passthrough commands keep their own argparse in the scripts
    if argv and argv[0] in {"scene", "reconstruct", "capture"}:
        mapping = {"scene": "make_scene.py", "reconstruct": "reconstruct.py",
                   "capture": "reconstruct_from_images.py"}
        return _run_script(mapping[argv[0]], argv[1:])

    args = p.parse_args(argv)

    import torch
    from .gaussians3d import load_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cmd == "view":
        from .viewer import serve
        serve(args.model, port=args.port)
        return 0

    try:
        model = load_model(args.model, dev)
    except RuntimeError:
        dev = "cpu"; model = load_model(args.model, "cpu")

    if args.cmd == "info":
        import numpy as np
        m = model.means.detach().cpu().numpy()
        print(f"model: {args.model}")
        print(f"  gaussians: {model.num}")
        print(f"  center:    {np.round(m.mean(0), 3).tolist()}")
        print(f"  extent:    {float(np.linalg.norm(m - m.mean(0), axis=1).mean()):.3f}")
        return 0

    if args.cmd == "export-ply":
        from .io_ply import save_ply
        out = args.out or str(Path(args.model).with_name("point_cloud.ply"))
        n = save_ply(model, out)
        print(f"exported {n} gaussians -> {out}")
        return 0

    if args.cmd == "turntable":
        from .viz import render_orbit, save_gif, save_mp4
        out = args.out or str(Path(args.model).with_name("turntable.gif"))
        frames = render_orbit(model, n_frames=args.frames, size=args.size)
        save_gif(frames, out)
        mp4 = save_mp4(frames, str(Path(out).with_suffix(".mp4")))
        print(f"wrote {out}" + (f" and {mp4}" if mp4 else ""))
        return 0

    if args.cmd == "eval":
        from .eval import evaluate_views
        from .scene import load_scene
        imgs, cams, _ = load_scene(args.scene, device=dev)
        m = evaluate_views(model, cams, imgs, list(range(len(cams))))
        print(f"PSNR {m['psnr']:.2f} dB   SSIM {m['ssim']:.3f}   over {m['n']} views")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
