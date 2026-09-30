"""Vectorized port of ``sim_mainV4_Monte_Carlo.m``.

Generates aberrated wavefronts for the collimator (parallel) and the
interferometer (GXSS), combines them, propagates through an FFT to get
the PSF, and measures its ellipticity — for a configurable number of
Monte-Carlo samples.

Index convention (kept identical to the MATLAB code so results match
bit-for-bit): every pupil-size array is indexed ``[i_x, i_y]`` with
``x_ep[i_x]`` and ``y_ep[i_y]``; the phase mask uses radius
``sqrt(x_ep[i_x]**2 + y_ep[i_y]**2)`` while the polar evaluation of the
Zernike terms uses ``r = sqrt(x_ep[i_y]**2 + y_ep[i_x]**2) / radius_ex``
and ``t = atan2(y_ep[i_x], x_ep[i_y])`` — this reproduces the
transpose built into the original ``cartesian2polar.m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from .ellipticity import cac_ellipticity
from .rand_num import rand_num
from .zernike import zernike_poly


@dataclass
class SimConfig:
    """Simulation parameters (defaults mirror the MATLAB script)."""

    sim_total_num: int = 2        # number of Monte-Carlo samples
    n_parallel: int = 32          # Zernike terms for the collimator (Z4-Z36 range)
    n_gxss: int = 32              # Zernike terms for the interferometer
    nflag: int = 1                # 0 = standard Zernike, 1 = fringe Zernike
    rms_parallel: float = 1 / 14  # collimator wavefront RMS (waves)
    rms_gxss: float = 1 / 14      # interferometer wavefront RMS (waves)
    radius_ex: float = 231        # outer pupil radius in pixels
    radius_in: float = 0          # inner pupil radius in pixels
    ellipticity_k: int = 1        # pixel scale divisor passed to cac_ellipticity
    seed: int | None = None       # RNG seed for reproducibility (no MATLAB equivalent)


@dataclass
class PupilGrid:
    """Precomputed pupil grid quantities (all arrays indexed [i_x, i_y])."""

    n: int                        # samplingpoint_N
    x_ep: np.ndarray              # ascending x coordinates, shape (n,)
    y_ep: np.ndarray              # descending y coordinates, shape (n,)
    mask_parallel: np.ndarray     # boolean pupil mask (meshgrid coordinates)
    mask_gxss: np.ndarray         # boolean pupil mask (meshgrid coordinates)
    phase_mask: np.ndarray        # boolean mask for the Zernike phase support
    polar_r: np.ndarray           # normalized radius for Zernike evaluation
    polar_t: np.ndarray           # azimuth for Zernike evaluation


def load_sampling_n(mat_path: str | Path) -> int:
    """Load ``ExitPupil.mat`` and derive ``samplingpoint_N`` from its size.

    The MATLAB script only uses the image dimensions; the pupil itself is
    rebuilt analytically below.
    """
    data = sio.loadmat(str(mat_path))
    image = data["ExitPupil"]
    image_x = image.shape[0]
    if image_x % 2 == 0:
        return image_x - 1  # even number of samples
    return (image_x - 1) // 2  # odd number of samples


def build_pupil_grid(n: int, radius_ex: float, radius_in: float) -> PupilGrid:
    """Build pupil masks and polar coordinates on the simulation grid."""
    x_ep = np.linspace(-n / 2 + 0.5, n / 2 - 0.5, n)
    y_ep = np.linspace(n / 2 - 0.5, -n / 2 + 0.5, n)

    # meshgrid with identical ascending axes, as in the MATLAB script.
    x_ap, y_ap = np.meshgrid(x_ep, x_ep)
    distance = np.sqrt(x_ap**2 + y_ap**2)
    mask_parallel = (distance >= radius_in) & (distance <= radius_ex)
    mask_gxss = (distance >= 0) & (distance <= radius_ex)

    # Phase support: r_temp = sqrt(x_ep[i_x]**2 + y_ep[i_y]**2) in [0, radius_ex).
    r_temp = np.sqrt(x_ep[:, None] ** 2 + y_ep[None, :] ** 2)
    phase_mask = (r_temp >= 0) & (r_temp < radius_ex)

    # Polar coordinates with the transposed convention of cartesian2polar.m:
    # r[i_x, i_y] = sqrt(x_ep[i_y]**2 + y_ep[i_x]**2) / radius_ex
    # t[i_x, i_y] = atan2(y_ep[i_x], x_ep[i_y])
    polar_r = np.sqrt(x_ep[None, :] ** 2 + y_ep[:, None] ** 2) / radius_ex
    polar_t = np.arctan2(y_ep[:, None], x_ep[None, :])

    return PupilGrid(
        n=n,
        x_ep=x_ep,
        y_ep=y_ep,
        mask_parallel=mask_parallel,
        mask_gxss=mask_gxss,
        phase_mask=phase_mask,
        polar_r=polar_r,
        polar_t=polar_t,
    )


def wavefront_phase(grid: PupilGrid, anm: np.ndarray, nflag: int) -> np.ndarray:
    """Evaluate the Zernike phase screen, zero outside the phase support."""
    phase = np.zeros((grid.n, grid.n), dtype=np.float64)
    poly = zernike_poly(grid.polar_r, grid.polar_t, anm, nflag)
    phase[grid.phase_mask] = poly[grid.phase_mask]
    return phase


def aberrated_pupil(mask: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Generalized pupil function: mask * exp(i * 2 * pi * phase)."""
    return mask.astype(np.complex128) * np.exp(1j * 2 * np.pi * phase)


def psf_from_pupil(pupil: np.ndarray) -> np.ndarray:
    """Normalized PSF: |fftshift(fft2(pupil))|^2, peak normalized to 1."""
    fft_field = np.fft.fftshift(np.fft.fft2(pupil))
    psf = (fft_field * np.conj(fft_field)).real
    return psf / psf.max()


def run_simulation(
    config: SimConfig,
    mat_path: str | Path,
    coeffs: dict[str, np.ndarray] | None = None,
) -> tuple[dict, list[dict]]:
    """Run the Monte-Carlo simulation.

    Parameters
    ----------
    config : SimConfig
        Simulation parameters.
    mat_path : str or Path
        Path to ``ExitPupil.mat`` (used only to derive the grid size).
    coeffs : dict, optional
        Fixed coefficient vectors for deterministic runs / cross-validation:
        ``Anm_parallel`` (32,), ``Anm_GXSS`` (32,) and ``Deta_error``
        (sim_total_num, 32). When given, the RNG is not used for these.

    Returns
    -------
    initial : dict
        Theoretical (error-free) wavefronts, aberrated pupils, PSFs and
        the GXSS ellipticity — mirrors the MATLAB ``initial`` struct.
    ansmat : list of dict
        Per-sample ``Deta_error``, ``ellipticity_parallel_GXSS`` and ``PSF``
        — mirrors the MATLAB ``ansmat`` struct array.
    """
    rng = np.random.default_rng(config.seed)
    n = load_sampling_n(mat_path)
    grid = build_pupil_grid(n, config.radius_ex, config.radius_in)

    def _draw(name: str, n_terms: int, rms: float, index: int | None = None) -> np.ndarray:
        if coeffs is not None and name in coeffs:
            arr = np.asarray(coeffs[name], dtype=np.float64)
            return arr[index] if index is not None else arr
        return rand_num(n_terms, rms, config.nflag, rng)

    initial: dict[str, np.ndarray | float] = {}
    ansmat: list[dict[str, np.ndarray | float]] = [{} for _ in range(config.sim_total_num)]

    # --- collimator (parallel) wavefront ---
    anm_parallel = _draw("Anm_parallel", config.n_parallel, config.rms_parallel)
    phase_parallel = wavefront_phase(grid, anm_parallel, config.nflag)
    parallel_aberrated = aberrated_pupil(grid.mask_parallel, phase_parallel)
    initial["parallel_zernike"] = anm_parallel
    initial["parallel_poly_zernike"] = phase_parallel
    initial["parallel_aberrated"] = parallel_aberrated

    # --- interferometer (GXSS) wavefront ---
    anm_gxss = _draw("Anm_GXSS", config.n_gxss, config.rms_gxss)
    phase_gxss = wavefront_phase(grid, anm_gxss, config.nflag)
    gxss_aberrated = aberrated_pupil(grid.mask_gxss, phase_gxss)
    initial["GXSS_zernike"] = anm_gxss
    initial["GXSS_poly_zernike"] = phase_gxss
    initial["GXSS_aberrated"] = gxss_aberrated

    # --- theoretical combined wavefront (no measurement error) ---
    anm_parallel_gxss = anm_parallel + anm_gxss
    phase_parallel_gxss = wavefront_phase(grid, anm_parallel_gxss, config.nflag)
    parallel_gxss_aberrated = aberrated_pupil(grid.mask_gxss, phase_parallel_gxss)
    psf_parallel_gxss = psf_from_pupil(parallel_gxss_aberrated)
    initial["parallel_GXSS_poly_zernike"] = phase_parallel_gxss
    initial["parallel_GXSS_aberrated"] = parallel_gxss_aberrated
    initial["parallel_GXSS_PSF"] = psf_parallel_gxss

    # --- Monte-Carlo loop with per-sample measurement error ---
    for step in range(config.sim_total_num):
        deta_error = _draw("Deta_error", config.n_gxss, config.rms_gxss, index=step)
        anm_total = anm_parallel + anm_gxss + deta_error
        phase_total = wavefront_phase(grid, anm_total, config.nflag)
        pupil_total = aberrated_pupil(grid.mask_parallel, phase_total)

        psf = psf_from_pupil(pupil_total)
        ellipticity = cac_ellipticity(psf, config.ellipticity_k)

        ansmat[step]["Deta_error"] = deta_error
        ansmat[step]["PSF"] = psf
        ansmat[step]["ellipticity_parallel_GXSS"] = ellipticity

    # --- GXSS-only PSF and ellipticity (diagnostic) ---
    psf_gxss = psf_from_pupil(gxss_aberrated)
    initial["ellipticity_GXSS"] = cac_ellipticity(psf_gxss, config.ellipticity_k)

    return initial, ansmat


def save_results(
    initial: dict,
    ansmat: list[dict],
    out_dir: str | Path,
    write_mat: bool = True,
) -> None:
    """Save results as ``initial.npz`` / ``ansmat.npz`` (and ``*.mat``).

    The ``.mat`` variants keep MATLAB compatibility: ``initial`` is a
    struct, ``ansmat`` a 1xn struct array with fields ``Deta_error``,
    ``PSF`` and ``ellipticity_parallel_GXSS``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- numpy-native npz ----
    np.savez_compressed(out / "initial.npz", **initial)
    np.savez_compressed(
        out / "ansmat.npz",
        Deta_error=np.stack([a["Deta_error"] for a in ansmat]),
        PSF=np.stack([a["PSF"] for a in ansmat]),
        ellipticity_parallel_GXSS=np.array(
            [a["ellipticity_parallel_GXSS"] for a in ansmat]
        ),
    )

    if not write_mat:
        return

    # ---- MATLAB-compatible mat (v7, compressed like MATLAB's default) ----
    initial_mat = {k: np.atleast_2d(np.asarray(v)) for k, v in initial.items()}
    sio.savemat(out / "initial.mat", {"initial": initial_mat}, do_compression=True)

    # A MATLAB struct array must be written as a structured object array
    # (a plain object array of dicts would load as a cell of structs).
    names = list(ansmat[0].keys())
    struct_arr = np.empty((1, len(ansmat)), dtype=[(name, object) for name in names])
    for j, sample in enumerate(ansmat):
        for name in names:
            struct_arr[0, j][name] = np.atleast_2d(np.asarray(sample[name]))
    sio.savemat(out / "ansmat.mat", {"ansmat": struct_arr}, do_compression=True)
