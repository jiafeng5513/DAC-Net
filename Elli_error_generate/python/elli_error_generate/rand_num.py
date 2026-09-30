"""Port of ``RandNum.m`` — random Zernike coefficients with fixed RMS.

A uniform random vector in ``(-1, 1)`` is scaled so that its RMS matches
``RMS_Level``. For fringe Zernike (``nflag=1``) each coefficient is then
multiplied by ``sqrt(Coff_ZFC)``, the CodeV fringe-Zernike normalization
constant for Z5..Z36 — so the *surface* RMS of the resulting polynomial
(terms evaluated with unit peak) equals ``RMS_Level``.
"""

from __future__ import annotations

import numpy as np

# CodeV fringe Zernike coefficient normalization, Z5..Z36 (33rd entry of
# the original list dropped: the MATLAB code uses exactly these 32 values).
COFF_ZFC_Z5_Z36 = np.array(
    [6, 6, 8, 8, 5, 8, 8, 10, 10, 12, 12, 7, 10, 10, 12, 12,
     14, 14, 16, 16, 9, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 11],
    dtype=np.float64,
)


def rand_num(n: int, rms_level: float, nflag: int = 1,
             rng: np.random.Generator | None = None,
             raw_vec: np.ndarray | None = None) -> np.ndarray:
    """Generate ``n`` random Zernike coefficients with the given RMS level.

    Parameters
    ----------
    n : int
        Number of coefficients (32 for Z5..Z36).
    rms_level : float
        Target RMS of the wavefront contribution.
    nflag : int
        ``0`` = standard Zernike, ``1`` = fringe Zernike.
    rng : numpy.random.Generator, optional
        Random generator for reproducibility. Defaults to a fresh
        ``default_rng()``, matching MATLAB's ``rand`` in distribution
        but not in stream.
    raw_vec : numpy.ndarray, optional
        Pre-drawn uniform vector in ``(-1, 1)`` of length ``n``. When
        given, it is used instead of drawing from ``rng`` — useful for
        cross-validation against MATLAB/Octave runs.

    Returns
    -------
    numpy.ndarray
        Coefficient vector of shape ``(n,)``.
    """
    if nflag not in (0, 1):
        raise ValueError("nflag must be 0 or 1.")

    if raw_vec is None:
        if rng is None:
            rng = np.random.default_rng()
        rand_vec = (rng.random(n) - 0.5) * 2  # uniform in (-1, 1), like (rand-0.5)*2
    else:
        rand_vec = np.asarray(raw_vec, dtype=np.float64).ravel()
        if rand_vec.size != n:
            raise ValueError(f"raw_vec has {rand_vec.size} entries, expected n={n}.")
    sum_sq = float(np.sum(rand_vec**2))  # Coff_ZSC = ones, so sum(rand.^2 * 1)

    anm = rand_vec / np.sqrt(sum_sq) * rms_level
    if nflag == 1:
        if n != COFF_ZFC_Z5_Z36.size:
            raise ValueError(
                f"n must be {COFF_ZFC_Z5_Z36.size} (Z5..Z36) for fringe Zernike coefficients."
            )
        anm = anm * np.sqrt(COFF_ZFC_Z5_Z36)
    return anm
