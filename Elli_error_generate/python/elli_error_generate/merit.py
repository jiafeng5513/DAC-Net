"""Merit-function quantities for the PSF-matching compensation solver.

Implements the three observables used by the paper's merit functions
(Applied Optics 63(3), 743-756, 2024, Eqs. 8-10):

- ellipticity components ``(e, e1, e2)`` from quadrupole moments (Eqs. 5-7),
  extending ``cac_ellipticity_fun.m`` (which returns only ``e``);
- the EE80 radius: radius of the circle centered on the PSF centroid that
  encloses 80% of the total energy;
- the PSF morphology difference: sum of squared pixel differences between
  two normalized PSFs (Eq. 10).
"""

from __future__ import annotations

import numpy as np

DEFAULT_PIXEL_SCALE = 0.0004  # mm per pixel (0.4 um)


def ee80_radius(psf, pixel_scale: float = DEFAULT_PIXEL_SCALE,
                window: int | None = 511) -> float:
    """Radius (in mm) of the circle around the PSF centroid enclosing 80% energy.

    Parameters
    ----------
    psf : array_like, 2-D
        PSF intensity (any normalization; the 80% fraction is scale-free).
    pixel_scale : float
        Physical size of one pixel in mm.
    window : int or None
        Side length of the square crop centered on the PSF peak used for
        the calculation. The crop captures effectively all of the energy
        for the simulated PSFs; ``None`` uses the full frame.
    """
    psf = np.ascontiguousarray(np.asarray(psf))
    psf = psf.real.astype(np.float64, copy=False)

    idx = int(np.argmax(psf.ravel(order="F")))
    x_max, y_max = np.unravel_index(idx, psf.shape, order="F")

    if window is None:
        work = psf.copy()
        x0_off = y0_off = 0
    else:
        h = window // 2
        x0, x1 = max(x_max - h, 0), min(x_max + h + 1, psf.shape[0])
        y0, y1 = max(y_max - h, 0), min(y_max + h + 1, psf.shape[1])
        work = np.zeros((window, window), dtype=np.float64)
        work[x0 - (x_max - h):x0 - (x_max - h) + (x1 - x0),
             y0 - (y_max - h):y0 - (y_max - h) + (y1 - y0)] = psf[x0:x1, y0:y1]
        x0_off, y0_off = x_max - h, y_max - h

    total = work.sum()
    if total <= 0:
        raise ValueError("PSF has non-positive total energy.")

    gx, gy = np.meshgrid(np.arange(work.shape[1]) + y0_off,
                         np.arange(work.shape[0]) + x0_off)
    xc = (work * gx).sum() / total
    yc = (work * gy).sum() / total

    target = 0.8 * total
    d2 = ((gx - xc) ** 2 + (gy - yc) ** 2).ravel()
    intensity = work.ravel()

    # The inner 80% of the energy lives in the smallest-distance pixels, so sort
    # only those instead of all ~window^2 pixels. Markov's inequality guarantees
    # a (2.5 x RMS-radius) cutoff encloses >= 84% of the energy; selecting every
    # pixel within that squared-radius keeps the sorted prefix - and thus the
    # interpolation below - identical to a full sort.
    r2 = max(6.25 * float((intensity * d2).sum() / total), 1.0)
    d2max = float(d2.max())
    while True:
        sel = d2 <= r2
        if intensity[sel].sum() >= target or r2 >= d2max:
            break
        r2 *= 2.0
    inten_s = intensity[sel]
    dist_sel = np.sqrt(d2[sel])
    order = np.argsort(dist_sel, kind="stable")
    dist_s = dist_sel[order]
    cum = np.cumsum(inten_s[order])

    hi = int(np.searchsorted(cum, target))
    if hi <= 0:
        return float(dist_s[0]) * pixel_scale
    if hi >= cum.size:
        return float(dist_s[-1]) * pixel_scale
    frac = (target - cum[hi - 1]) / (cum[hi] - cum[hi - 1])
    radius = dist_s[hi - 1] + frac * (dist_s[hi] - dist_s[hi - 1])
    return float(radius) * pixel_scale


def psf_morphology_diff(psf_a, psf_b) -> float:
    """Sum of squared pixel differences between two PSFs (paper Eq. 10)."""
    a = np.asarray(psf_a, dtype=np.float64).real
    b = np.asarray(psf_b, dtype=np.float64).real
    if a.shape != b.shape:
        raise ValueError(f"PSF shape mismatch: {a.shape} vs {b.shape}.")
    return float(np.sum((a - b) ** 2))
