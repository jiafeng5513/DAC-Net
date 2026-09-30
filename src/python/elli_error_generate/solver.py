"""PSF-matching compensation solver (paper Secs. 2.E-4).

Solves for the Zernike coefficients of a compensation ("virtual lens")
surface so that the modeled bench PSF (PSF-S) matches a measured PSF
(PSF-M), then reports the ellipticity of the compensated system.

Pipeline per merit evaluation, matching the paper's three merit functions
(Eqs. 8-10, weights w1..w4 = 0.5 by default):

    PSF-S(x) = |FFT{ P * exp(i 2 pi (WF_col + WF_tel_meas + x))} |^2
    f_e = w1 |e_S - e_M|  + w2 |e1_S - e1_M|
    f_r = w3 |EE80_S - EE80_M|
    f_p = w4 sum (PSF-S - PSF-M)^2

with NSGA-II minimizing the three-objective vector (f_e, f_r, f_p).
The forward model is the verified Monte-Carlo generator kernel evaluated
in batches on the pupil support for speed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft2, fftshift

from .ellipticity import cac_ellipticity_full
from .merit import ee80_radius, psf_morphology_diff
from .simulation import PupilGrid, build_pupil_grid, wavefront_phase
from .zernike import zernike_terms


@dataclass
class SolverConfig:
    """NSGA-II and model settings (defaults follow the paper's Sec. 4.A)."""

    n_terms: int = 32           # compensation coefficients (Z5..Z36 fringe)
    nflag: int = 1
    radius_ex: float = 231
    radius_in: float = 0
    pop_size: int = 180
    n_gen: int = 120
    crossover_prob: float = 1.0     # simulated binary crossover probability
    mutation_prob: float = 0.1      # polynomial mutation probability
    coeff_bound: float = 0.2        # |x_i| bound (waves); paper Table 1 3σ ~ 0.15λ
    chunk: int = 16             # population chunk size for batched FFT
    w_e: float = 0.5
    w_e1: float = 0.5
    w_r: float = 0.5
    w_p: float = 0.5
    seed: int | None = None
    fast: bool = False          # complex64/float32 FFT path (~2x faster, less memory)
    fft_workers: int = -1       # scipy FFT threads (-1 = all cores; set lower when running samples in parallel)


@dataclass
class TargetPSF:
    """Measured PSF (PSF-M) with its precomputed merit observables."""

    psf: np.ndarray
    e: float
    e1: float
    ee80: float

    @classmethod
    def from_psf(cls, psf: np.ndarray) -> TargetPSF:
        e, e1, _ = cac_ellipticity_full(psf)
        return cls(psf=np.asarray(psf, dtype=np.float64), e=e, e1=e1,
                   ee80=ee80_radius(psf))


class ForwardModel:
    """Batched PSF forward model on the simulation pupil grid.

    Computes PSF-S for a batch of compensation coefficient vectors:
    phase = base_phase + X @ terms evaluated on the pupil support only.
    """

    def __init__(self, grid: PupilGrid, base_coeffs: np.ndarray,
                 config: SolverConfig):
        if base_coeffs.size != config.n_terms:
            raise ValueError(
                f"base_coeffs has {base_coeffs.size} entries, expected {config.n_terms}."
            )
        self.grid = grid
        self.config = config
        self.mask = grid.mask_gxss
        self.cdtype = np.complex64 if config.fast else np.complex128
        self.fdtype = np.float32 if config.fast else np.float64
        self.mask_c = self.mask.astype(self.cdtype)

        base_phase = wavefront_phase(grid, np.asarray(base_coeffs, dtype=np.float64),
                                     config.nflag)
        support = grid.phase_mask
        self.support_flat = np.flatnonzero(support)
        self.rows = self.support_flat // grid.n
        self.cols = self.support_flat % grid.n

        polar_r = grid.polar_r[support]
        polar_t = grid.polar_t[support]
        self.base_support = base_phase[support]
        self.terms_support = zernike_terms(polar_r, polar_t, config.nflag)

    def _psf_chunk(self, xb: np.ndarray) -> np.ndarray:
        """PSF-S for a small batch ``xb`` (shape (mb, n_terms))."""
        n = self.grid.n
        xb = np.asarray(xb, dtype=np.float64)
        mb = xb.shape[0]
        phase = self.base_support[None, :] + xb @ self.terms_support
        # Amplitude 1 across the mask (a few mask pixels lie outside the phase
        # support and stay at unit amplitude, zero phase); support pixels carry
        # exp(i*2*pi*phase). Support is a subset of the mask, so no extra
        # ``* mask`` is needed.
        pupil = np.repeat(self.mask_c[None], mb, axis=0)
        pupil[:, self.rows, self.cols] = np.exp(2j * np.pi * phase)
        field = fft2(pupil, axes=(1, 2), workers=self.config.fft_workers,
                     overwrite_x=True)
        # |fftshift(field)|^2 == fftshift(|field|^2); shift the real PSF, which
        # is half the bytes of the complex field.
        psf = field.real ** 2 + field.imag ** 2
        psf = fftshift(psf, axes=(1, 2))
        psf /= psf.max(axis=(1, 2), keepdims=True)
        return psf

    def iter_psf_chunks(self, x_batch: np.ndarray):
        """Yield ``(start, psf_chunk)`` pairs without materializing all PSFs."""
        m = x_batch.shape[0]
        chunk = max(1, self.config.chunk)
        for s in range(0, m, chunk):
            yield s, self._psf_chunk(x_batch[s:s + chunk])

    def psf_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """PSF-S for each row of ``x_batch`` (shape (m, n_terms))."""
        m = x_batch.shape[0]
        n = self.grid.n
        psfs = np.empty((m, n, n), dtype=self.fdtype)
        for s, psf in self.iter_psf_chunks(x_batch):
            psfs[s:s + psf.shape[0]] = psf
        return psfs

    def psf(self, x: np.ndarray) -> np.ndarray:
        """PSF-S for a single compensation vector."""
        return self.psf_batch(np.atleast_2d(x))[0]


class CompensationProblem:
    """pymoo Problem wrapper (lazy import so the rest of the package works without pymoo)."""

    def __init__(self, model: ForwardModel, target: TargetPSF, config: SolverConfig):
        from pymoo.core.problem import Problem

        self.model = model
        self.target = target
        self.config = config

        bound = np.broadcast_to(
            np.asarray(config.coeff_bound, dtype=float), (config.n_terms,))

        class _Problem(Problem):
            def __init__(self, outer: CompensationProblem):
                super().__init__(n_var=config.n_terms, n_obj=3,
                                 xl=-bound, xu=bound)
                self.outer = outer

            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = self.outer.evaluate(X)

        self._problem = _Problem(self)

    def evaluate(self, x_batch: np.ndarray) -> np.ndarray:
        """Three merit objectives for each row of ``x_batch``."""
        cfg = self.config
        tgt = self.target
        m = x_batch.shape[0]
        f = np.empty((m, 3), dtype=np.float64)
        # Evaluate PSFs chunk-by-chunk so the full (m, n, n) stack is never held.
        for s, psfs in self.model.iter_psf_chunks(x_batch):
            for j in range(psfs.shape[0]):
                psf = psfs[j]
                e, e1, _ = cac_ellipticity_full(psf)
                f[s + j, 0] = cfg.w_e * abs(e - tgt.e) + cfg.w_e1 * abs(e1 - tgt.e1)
                f[s + j, 1] = cfg.w_r * abs(ee80_radius(psf) - tgt.ee80)
                f[s + j, 2] = cfg.w_p * psf_morphology_diff(psf, tgt.psf)
        return f

    def aggregate(self, x_batch: np.ndarray) -> np.ndarray:
        """Comprehensive merit F = f_e + f_r + f_p (paper's aggregate)."""
        return self.evaluate(x_batch).sum(axis=1)


def _make_progress_callback(n_gen: int, label: str):
    """pymoo callback printing one updating progress line per generation."""
    from pymoo.core.callback import Callback

    class _Progress(Callback):
        def __init__(self):
            super().__init__()
            self.t0 = time.perf_counter()

        def notify(self, algorithm):
            gen = int(getattr(algorithm, "n_gen", 0) or 0)
            gen = max(1, min(gen, n_gen))
            F = algorithm.pop.get("F")
            best = float(F.sum(axis=1).min()) if F is not None else float("nan")
            elapsed = time.perf_counter() - self.t0
            frac = gen / n_gen
            eta = elapsed / frac - elapsed if frac > 0 else 0.0
            head = f"{label} " if label else ""
            print(f"\r{head}gen {gen:>3}/{n_gen} ({100 * frac:3.0f}%)  "
                  f"best_merit={best:.4g}  "
                  f"{elapsed:5.0f}s elapsed ~{eta:4.0f}s left   ",
                  end="", flush=True)

    return _Progress()


def run_nsga2(problem: CompensationProblem, config: SolverConfig,
              verbose: bool = False, progress: bool = False, label: str = ""):
    """Run NSGA-II with the paper's operators; returns the pymoo result."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    algorithm = NSGA2(
        pop_size=config.pop_size,
        crossover=SBX(prob=config.crossover_prob),
        mutation=PM(prob=config.mutation_prob),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", config.n_gen)
    # pymoo's verbose table and our one-line progress bar conflict; use one.
    # Only pass a callback when we have one: pymoo dereferences a None callback.
    kwargs = {}
    callback = None
    if progress and not verbose:
        callback = _make_progress_callback(config.n_gen, label)
        kwargs["callback"] = callback
    res = minimize(problem._problem, algorithm, termination, seed=config.seed,
                   verbose=verbose, **kwargs)
    if callback is not None:
        print()
    return res


@dataclass
class CompensationResult:
    """Outcome of one compensation optimization."""

    x_pareto: np.ndarray          # (n_pareto, n_terms) compensation coefficients
    f_pareto: np.ndarray          # (n_pareto, 3) objectives
    f_aggregate: np.ndarray       # (n_pareto,) comprehensive merit
    config: SolverConfig = field(repr=False)


def solve(target_psf: np.ndarray, base_coeffs: np.ndarray,
          n: int, config: SolverConfig | None = None,
          verbose: bool = False, progress: bool = False,
          label: str = "") -> tuple[CompensationResult, ForwardModel]:
    """Compensate ``base_coeffs`` so the modeled PSF matches ``target_psf``.

    Parameters
    ----------
    target_psf : ndarray
        Measured PSF (PSF-M).
    base_coeffs : ndarray
        Known modeled-system coefficients (collimator + measured telescope),
        which the compensation is added to.
    n : int
        Pupil grid size (samplingpoint_N, e.g. 1023 from ExitPupil.mat).
    """
    config = config or SolverConfig()
    grid = build_pupil_grid(n, config.radius_ex, config.radius_in)
    model = ForwardModel(grid, base_coeffs, config)
    target = TargetPSF.from_psf(target_psf)
    problem = CompensationProblem(model, target, config)
    res = run_nsga2(problem, config, verbose=verbose, progress=progress,
                    label=label)
    return CompensationResult(
        x_pareto=np.atleast_2d(res.X),
        f_pareto=np.atleast_2d(res.F),
        f_aggregate=np.atleast_2d(res.F).sum(axis=1),
        config=config,
    ), model
