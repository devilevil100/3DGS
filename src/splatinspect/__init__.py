"""splatinspect — 3D Gaussian-Splatting damage inspection.

A from-scratch (pure-PyTorch) 3D Gaussian Splatting engine plus a damage
inspection pipeline: reconstruct a 3D digital twin from posed images, segment
the damage, measure it in real-world units, and produce a report.

No gsplat, no CUDA compilation, no COLMAP required for the synthetic/posed path.
"""

__version__ = "0.1.0"
