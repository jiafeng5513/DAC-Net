"""Port of ``cac_ellipticity_fun.m`` — PSF ellipticity from quadrupole moments.

Crops an effective PSF window (radius ``Eff_Radius`` around the peak),
masks it to a circle, then computes the centroid and second central
moments. The ellipticity is ``sqrt(e1^2 + e2^2)`` with

    e1 = (Qxx - Qyy) / (Qxx + Qyy)
    e2 = 2 * Qxy / (Qxx + Qyy)
"""

from __future__ import annotations

import numpy as np


def cac_ellipticity_full(psf, k: int = 1) -> tuple[float, float, float]:
    """Ellipticity components ``(e, e1, e2)`` of a (real, non-negative) PSF image.

    Parameters
    ----------
    psf : array_like, 2-D
        Point spread function intensity. Complex input is accepted for
        parity with MATLAB (``fft .* conj(fft)``); only the real part
        is used, which is numerically exact since the imaginary part
        of such a product is exactly zero.
    k : int
        Pixel-scale divisor: ``FocInc = 0.0004 / k`` mm per pixel.

    Returns
    -------
    tuple of float
        ``(e, e1, e2)`` with ``e1 = (Qxx - Qyy)/(Qxx + Qyy)``,
        ``e2 = 2*Qxy/(Qxx + Qyy)``, ``e = sqrt(e1**2 + e2**2)``; the
        quadrupole moments are taken inside the same effective-radius
        window as the original MATLAB function.
    """
    psf = np.ascontiguousarray(np.asarray(psf))
    psf = psf.real.astype(np.float64, copy=False)

    foc_inc = 0.0004 / k            # mm per pixel (0.4 um / k)
    eff_radius = 0.0679             # mm, effective PSF radius (0.5 arcsec on 28 m)
    eff_radius_num = int(np.floor(eff_radius / foc_inc))

    # Peak location, first occurrence in column-major order (MATLAB `find`).
    idx = int(np.argmax(psf.ravel(order="F")))
    x_max, y_max = np.unravel_index(idx, psf.shape, order="F")

    n = eff_radius_num
    x0, x1 = x_max - n, x_max + n + 1
    y0, y1 = y_max - n, y_max + n + 1
    if x0 < 0 or y0 < 0 or x1 > psf.shape[0] or y1 > psf.shape[1]:
        raise ValueError(
            f"Effective PSF window [{x0}:{x1}, {y0}:{y1}] exceeds the PSF bounds "
            f"{psf.shape}; the PSF peak must be at least {n} pixels from the border."
        )
    eff_psf = psf[x0:x1, y0:y1].copy()

    x_eff = np.arange(-n, n + 1) * foc_inc  # window x coordinates (rows)
    y_eff = np.arange(-n, n + 1) * foc_inc  # window y coordinates (cols)

    # Mask everything outside the effective-radius circle.
    outside = x_eff[:, None] ** 2 + y_eff[None, :] ** 2 > eff_radius ** 2
    eff_psf[outside] = 0.0

    dx = dy = foc_inc
    flux = eff_psf.sum() * dx * dy

    x_center = (eff_psf * x_eff[:, None]).sum() * dx * dy / flux
    y_center = (eff_psf * y_eff[None, :]).sum() * dx * dy / flux

    qxx = (eff_psf * (x_eff[:, None] - x_center) ** 2).sum() * dx * dy
    qyy = (eff_psf * (y_eff[None, :] - y_center) ** 2).sum() * dx * dy
    qxy = (eff_psf * (x_eff[:, None] - x_center) * (y_eff[None, :] - y_center)).sum() * dx * dy

    e1 = (qxx - qyy) / (qxx + qyy)
    e2 = 2.0 * qxy / (qxx + qyy)
    return float(np.sqrt(e1**2 + e2**2)), float(e1), float(e2)


def cac_ellipticity(psf, k: int = 1) -> float:
    """Ellipticity magnitude ``sqrt(e1**2 + e2**2)`` of a PSF image."""
    return cac_ellipticity_full(psf, k)[0]
