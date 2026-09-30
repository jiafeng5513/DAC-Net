"""Vectorized Zernike polynomial evaluation.

Port of ``zernike_fun.m``. Two orderings are supported, selected by
``nflag`` (identical to the MATLAB convention):

- ``nflag=0``: standard Zernike ordering used by the original code
- ``nflag=1``: fringe Zernike ordering (the one used by the GXSS sim)

In both cases the coefficient vector ``Anm`` maps to terms Z5..Z36
(32 terms), exactly as in the MATLAB ``zernike=[ZRN_5 ... ZRN_36]`` row.
``r`` is the normalized radial coordinate in ``[0, 1]`` and ``t`` the
azimuth in radians.
"""

from __future__ import annotations

import numpy as np

NUM_TERMS = 32  # Z5..Z36


def _terms_standard(r: np.ndarray, t: np.ndarray) -> list[np.ndarray]:
    """Standard Zernike terms Z5..Z36 (nflag=0)."""
    return [
        2 * r**2 - 1,                                  # Z5
        r**2 * np.sin(2 * t),                          # Z6
        r**3 * np.cos(3 * t),                          # Z7
        (3 * r**3 - 2 * r) * np.cos(t),                # Z8
        (3 * r**3 - 2 * r) * np.sin(t),                # Z9
        r**3 * np.sin(3 * t),                          # Z10
        r**4 * np.cos(4 * t),                          # Z11
        (4 * r**4 - 3 * r**2) * np.cos(2 * t),         # Z12
        6 * r**4 - 6 * r**2 + 1,                       # Z13
        (4 * r**4 - 3 * r**2) * np.sin(2 * t),         # Z14
        r**4 * np.sin(4 * t),                          # Z15
        r**5 * np.cos(5 * t),                          # Z16
        (5 * r**5 - 4 * r**3) * np.cos(3 * t),         # Z17
        (10 * r**5 - 12 * r**3 + 3 * r) * np.cos(t),   # Z18
        (10 * r**5 - 12 * r**3 + 3 * r) * np.sin(t),   # Z19
        (5 * r**5 - 4 * r**3) * np.sin(3 * t),         # Z20
        r**5 * np.sin(5 * t),                          # Z21
        r**6 * np.cos(6 * t),                          # Z22
        (6 * r**6 - 5 * r**4) * np.cos(4 * t),         # Z23
        (15 * r**6 - 20 * r**4 + 6 * r**2) * np.cos(2 * t),  # Z24
        20 * r**6 - 30 * r**4 + 12 * r**2 - 1,         # Z25
        (15 * r**6 - 20 * r**4 + 6 * r**2) * np.sin(2 * t),  # Z26
        (6 * r**6 - 5 * r**4) * np.sin(4 * t),         # Z27
        r**6 * np.sin(6 * t),                          # Z28
        r**7 * np.cos(7 * t),                          # Z29
        (7 * r**7 - 6 * r**5) * np.cos(5 * t),         # Z30
        (21 * r**7 - 30 * r**5 + 10 * r**3) * np.cos(3 * t),  # Z31
        (35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r) * np.cos(t),  # Z32
        (35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r) * np.sin(t),  # Z33
        (21 * r**7 - 30 * r**5 + 10 * r**3) * np.sin(3 * t),      # Z34
        (7 * r**7 - 6 * r**5) * np.sin(5 * t),         # Z35
        r**7 * np.sin(7 * t),                          # Z36
    ]


def _terms_fringe(r: np.ndarray, t: np.ndarray) -> list[np.ndarray]:
    """Fringe Zernike terms Z5..Z36 (nflag=1)."""
    return [
        r**2 * np.cos(2 * t),                                # Z5
        r**2 * np.sin(2 * t),                                # Z6
        (3 * r**3 - 2 * r) * np.cos(t),                      # Z7
        (3 * r**3 - 2 * r) * np.sin(t),                      # Z8
        6 * r**4 - 6 * r**2 + 1,                             # Z9
        r**3 * np.cos(3 * t),                                # Z10
        r**3 * np.sin(3 * t),                                # Z11
        (4 * r**4 - 3 * r**2) * np.cos(2 * t),               # Z12
        (4 * r**4 - 3 * r**2) * np.sin(2 * t),               # Z13
        (10 * r**5 - 12 * r**3 + 3 * r) * np.cos(t),         # Z14
        (10 * r**5 - 12 * r**3 + 3 * r) * np.sin(t),         # Z15
        20 * r**6 - 30 * r**4 + 12 * r**2 - 1,               # Z16
        r**4 * np.cos(4 * t),                                # Z17
        r**4 * np.sin(4 * t),                                # Z18
        (5 * r**5 - 4 * r**3) * np.cos(3 * t),               # Z19
        (5 * r**5 - 4 * r**3) * np.sin(3 * t),               # Z20
        (15 * r**6 - 20 * r**4 + 6 * r**2) * np.cos(2 * t),  # Z21
        (15 * r**6 - 20 * r**4 + 6 * r**2) * np.sin(2 * t),  # Z22
        (35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r) * np.cos(t),  # Z23
        (35 * r**7 - 60 * r**5 + 30 * r**3 - 4 * r) * np.sin(t),  # Z24
        70 * r**8 - 140 * r**6 + 90 * r**4 - 20 * r**2 + 1,  # Z25
        r**5 * np.cos(5 * t),                                # Z26
        r**5 * np.sin(5 * t),                                # Z27
        (6 * r**6 - 5 * r**4) * np.cos(4 * t),               # Z28
        (6 * r**6 - 5 * r**4) * np.sin(4 * t),               # Z29
        (21 * r**7 - 30 * r**5 + 10 * r**3) * np.cos(3 * t),  # Z30
        (21 * r**7 - 30 * r**5 + 10 * r**3) * np.sin(3 * t),  # Z31
        (56 * r**8 - 105 * r**6 + 60 * r**4 - 10 * r**2) * np.cos(2 * t),  # Z32
        (56 * r**8 - 105 * r**6 + 60 * r**4 - 10 * r**2) * np.sin(2 * t),  # Z33
        (126 * r**9 - 280 * r**7 + 210 * r**5 - 60 * r**3 + 5 * r) * np.cos(t),  # Z34
        (126 * r**9 - 280 * r**7 + 210 * r**5 - 60 * r**3 + 5 * r) * np.sin(t),  # Z35
        252 * r**10 - 630 * r**8 + 560 * r**6 - 210 * r**4 + 30 * r**2 - 1,      # Z36
    ]


def zernike_terms(r, t, nflag: int = 1) -> np.ndarray:
    """Evaluate the Zernike basis terms Z5..Z36 as a stacked array.

    Returns an array of shape ``(32, *broadcast_shape)``; term ``i``
    multiplies coefficient ``anm[i]`` in :func:`zernike_poly`.
    """
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if nflag == 0:
        terms = _terms_standard(r, t)
    elif nflag == 1:
        terms = _terms_fringe(r, t)
    else:
        raise ValueError("nflag must be 0 or 1.")
    return np.stack(terms, axis=0)


def zernike_poly(r, t, anm, nflag: int = 1) -> np.ndarray:
    """Evaluate the weighted sum of Zernike terms Z5..Z36.

    Parameters
    ----------
    r, t : array_like
        Normalized radius (``[0, 1]``) and azimuth (rad), broadcastable
        to a common shape.
    anm : array_like
        Coefficient vector with 32 entries (Z5..Z36).
    nflag : int
        ``0`` = standard ordering, ``1`` = fringe ordering.

    Returns
    -------
    np.ndarray
        Weighted Zernike polynomial, same shape as the broadcast of
        ``r`` and ``t``.
    """
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    anm = np.asarray(anm, dtype=np.float64).ravel()

    terms = zernike_terms(r, t, nflag)
    if anm.size != terms.shape[0]:
        raise ValueError(
            f"anm has {anm.size} coefficients but {terms.shape[0]} terms (Z5..Z36) are defined."
        )
    return np.einsum("i,i...->...", anm, terms)
