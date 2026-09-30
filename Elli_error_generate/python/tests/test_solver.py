#!/usr/bin/env python3
"""Property and consistency tests for the compensation solver stack.

Run directly (no pytest required):
    python tests/test_solver.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elli_error_generate.ellipticity import cac_ellipticity, cac_ellipticity_full
from elli_error_generate.merit import ee80_radius, psf_morphology_diff
from elli_error_generate.simulation import (
    aberrated_pupil,
    build_pupil_grid,
    psf_from_pupil,
    wavefront_phase,
)
from elli_error_generate.solver import ForwardModel, SolverConfig

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def gaussian_psf(n: int, sx: float, sy: float, angle_deg: float = 0.0,
                 dx: int = 0, dy: int = 0) -> np.ndarray:
    """Elliptical Gaussian test PSF (sigma in pixels), peak near the center."""
    x = np.arange(n) - n // 2
    X, Y = np.meshgrid(x - dy, x - dx)  # rows=Y, cols=X; offsets keep peak in-bounds
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    xr = c * X + s * Y
    yr = -s * X + c * Y
    return np.exp(-(xr**2 / (2 * sx**2) + yr**2 / (2 * sy**2)))


def test_ellipticity_properties() -> None:
    # Axis-aligned ellipse: e2 == 0, e1 reflects x-vs-y; e invariant under rotation.
    psf = gaussian_psf(801, 20.0, 15.0)
    e, e1, e2 = cac_ellipticity_full(psf)
    # Code convention: rows are the x axis, columns the y axis. Sigma values
    # are chosen small relative to the ellipticity window (169 px radius) so
    # circular-mask truncation does not skew the quadrupole ratio.
    check("e1 axis-aligned matches analytic",
          abs(e1 - (15.0**2 - 20.0**2) / (20.0**2 + 15.0**2)) < 1e-3,
          f"e1={e1:.6f}")
    check("e2 axis-aligned ~ 0", abs(e2) < 1e-3, f"e2={e2:.2e}")

    psf_rot = gaussian_psf(801, 20.0, 15.0, angle_deg=30.0)
    e_r, _, e2_r = cac_ellipticity_full(psf_rot)
    check("e rotation-invariant", abs(e_r - e) < 1e-3, f"{e:.6f} vs {e_r:.6f}")
    check("rotated ellipse has e2 != 0", abs(e2_r) > 1e-3, f"e2={e2_r:.6f}")

    # Legacy single-value API agrees with the full computation.
    check("cac_ellipticity == cac_ellipticity_full[0]",
          abs(cac_ellipticity(psf) - e) < 1e-15)

    # Swapping axes flips e1.
    psf_swap = gaussian_psf(801, 15.0, 20.0)
    _, e1_s, _ = cac_ellipticity_full(psf_swap)
    check("axis swap flips e1", abs(e1_s + e1) < 1e-3, f"e1={e1:.6f} vs {e1_s:.6f}")


def test_ee80_brute_force() -> None:
    """EE80 via sorted cumulative sum vs independent ring integration."""
    rng = np.random.default_rng(0)
    psf = gaussian_psf(801, 55.0, 40.0, angle_deg=15.0)
    psf += 1e-4 * rng.random(psf.shape)  # break radial symmetry
    pixel_scale = 0.0004

    r_sort = ee80_radius(psf, pixel_scale, window=None)

    # Brute force: accumulate energy in integer-radius rings about the centroid.
    total = psf.sum()
    gx, gy = np.meshgrid(np.arange(psf.shape[1]), np.arange(psf.shape[0]))
    xc = (psf * gx).sum() / total
    yc = (psf * gy).sum() / total
    d = np.sqrt((gx - xc) ** 2 + (gy - yc) ** 2).ravel()
    ring_energy = np.bincount(np.floor(d).astype(int), weights=psf.ravel())
    cum = np.cumsum(ring_energy)
    r80_ring = float(np.argmax(cum >= 0.8 * total)) + 0.5  # ring mid-point
    check("ee80 sorted vs ring integration",
          abs(r_sort / pixel_scale - r80_ring) <= 1.0,
          f"sorted={r_sort / pixel_scale:.2f}px ring~{r80_ring:.2f}px")


def test_morphology_diff() -> None:
    a = np.ones((10, 10))
    check("morphology zero for identical", psf_morphology_diff(a, a.copy()) == 0.0)
    b = a.copy()
    b[0, 0] += 1.0
    check("morphology counts squared diff", psf_morphology_diff(a, b) == 1.0)


def test_batched_forward_matches_single() -> None:
    """ForwardModel batch evaluation must reproduce the verified single-forward path."""
    n = 127
    radius_ex = 25.0
    grid = build_pupil_grid(n, radius_ex, radius_in=0)
    rng = np.random.default_rng(1)
    base = rng.uniform(-0.05, 0.05, size=32)
    xs = rng.uniform(-0.1, 0.1, size=(3, 32))
    cfg = SolverConfig(radius_ex=radius_ex, chunk=2)

    model = ForwardModel(grid, base, cfg)
    batch = model.psf_batch(xs)
    for i in range(3):
        coeffs = base + xs[i]
        psf_ref = psf_from_pupil(aberrated_pupil(
            grid.mask_gxss, wavefront_phase(grid, coeffs, cfg.nflag)))
        diff = np.abs(batch[i] - psf_ref).max()
        check(f"batched forward[{i}] == single forward", diff < 1e-12,
              f"max_diff={diff:.2e}")

    # Peak normalization sanity.
    check("batched PSF peak == 1", np.allclose(batch.max(axis=(1, 2)), 1.0))


def main() -> None:
    test_ellipticity_properties()
    test_ee80_brute_force()
    test_morphology_diff()
    test_batched_forward_matches_single()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("All solver stack tests PASSED.")


if __name__ == "__main__":
    main()
