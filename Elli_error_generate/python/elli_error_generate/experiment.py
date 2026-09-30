"""Synthetic validation protocol for the compensation solver.

Reproduces the paper's simulation design (Sec. 4): the measured PSF
(PSF-M) is the clean bench image; the model is built from *measured*
wavefront coefficients (true + measurement error, Eqs. 11-12); the
compensation surface absorbs the error; the ellipticity of the
compensated system (telescope equivalent + virtual lens, collimator
removed) is compared against the ground-truth telescope ellipticity.

Two data sources:

- ``synthetic``: fresh samples from the verified generator; measurement
  error is Gaussian per coefficient (paper Eq. 11) with per-order sigma
  derived from the paper's Table 1 3σ ranges (Z5-Z9) and a smaller floor
  for the stable high orders.
- ``archived``: PSF-M taken from ``matlab/ansmat.mat`` (which contains an
  unknown physical perturbation ``Deta_error`` in the beam); the model
  knows only the archived true coefficients, so the compensation must
  recover the perturbation. Validation targets: the ellipticity of PSF-M
  itself (the perturbed system's true ellipticity) and the coefficient
  recovery error ``||x - Deta_error||`` (possible because the archived
  ``Deta_error`` is known).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import scipy.io as sio

from .ellipticity import cac_ellipticity_full
from .rand_num import rand_num
from .simulation import (
    SimConfig,
    aberrated_pupil,
    build_pupil_grid,
    load_sampling_n,
    psf_from_pupil,
    wavefront_phase,
)
from .solver import ForwardModel, SolverConfig, solve

# Per-order measurement-error sigma (waves) for the Z5..Z36 fringe basis:
# Z5-Z9 converted from the paper's Table 1 3σ ranges (/6), high orders stable.
DEFAULT_MEAS_SIGMA = np.concatenate([
    np.array([0.010, 0.0075, 0.0075, 0.022, 0.015]),
    np.full(27, 0.005),
])


@dataclass
class ExperimentConfig:
    """One synthetic experiment: several measurement-error realizations."""

    n_samples: int = 4
    rms_true: float = 1 / 14        # telescope (GXSS) true wavefront RMS
    rms_collimator: float = 1 / 14  # collimator wavefront RMS (known exactly)
    meas_sigma: np.ndarray = field(default_factory=lambda: DEFAULT_MEAS_SIGMA.copy())
    solver: SolverConfig = field(default_factory=SolverConfig)
    seed: int | None = None
    bound_k: float = 6.0            # per-coefficient search bound = bound_k * meas_sigma


@dataclass
class SampleResult:
    """Compensation outcome for one PSF-M sample."""

    truth_e: float                  # ground-truth telescope ellipticity
    truth_e1: float
    initial_e: float                # ellipticity of the uncompensated modeled system
    comp_e: np.ndarray              # compensated-system ellipticity per Pareto point
    comp_e1: np.ndarray
    best_index: int                 # Pareto point with min |comp_e - truth_e|
    x_pareto: np.ndarray
    f_pareto: np.ndarray
    coef_ref: np.ndarray | None = None  # reference coefficients (archived Deta_error)


def _forward_single(grid, coeffs: np.ndarray, nflag: int) -> np.ndarray:
    phase = wavefront_phase(grid, coeffs, nflag)
    return psf_from_pupil(aberrated_pupil(grid.mask_gxss, phase))


def _evaluate_sample(psf_m: np.ndarray, model_base: np.ndarray,
                     answer_base: np.ndarray, truth_psf: np.ndarray,
                     n: int, solver_cfg: SolverConfig,
                     coef_ref: np.ndarray | None = None,
                     verbose: bool = False, progress: bool = False,
                     label: str = "") -> SampleResult:
    """Optimize compensation for one PSF-M and evaluate against ground truth."""
    grid = build_pupil_grid(n, solver_cfg.radius_ex, solver_cfg.radius_in)
    truth_e, truth_e1, _ = cac_ellipticity_full(truth_psf)
    initial_e, _, _ = cac_ellipticity_full(_forward_single(grid, model_base,
                                                           solver_cfg.nflag))

    result, _ = solve(psf_m, model_base, n, solver_cfg, verbose=verbose,
                      progress=progress, label=label)

    answer_model = ForwardModel(grid, answer_base, solver_cfg)
    comp_psfs = answer_model.psf_batch(result.x_pareto)
    comp_e = np.empty(len(comp_psfs))
    comp_e1 = np.empty(len(comp_psfs))
    for i, psf in enumerate(comp_psfs):
        comp_e[i], comp_e1[i], _ = cac_ellipticity_full(psf)
    best_index = int(np.argmin(np.abs(comp_e - truth_e)))
    return SampleResult(truth_e=truth_e, truth_e1=truth_e1, initial_e=initial_e,
                        comp_e=comp_e, comp_e1=comp_e1, best_index=best_index,
                        x_pareto=result.x_pareto, f_pareto=result.f_pareto,
                        coef_ref=coef_ref)


def run_synthetic_experiment(cfg: ExperimentConfig, mat_path: str | Path,
                             verbose: bool = False, progress: bool = False,
                             jobs: int = 1) -> list[SampleResult]:
    """Run the synthetic protocol for ``cfg.n_samples`` error realizations."""
    rng = np.random.default_rng(cfg.seed)
    n = load_sampling_n(mat_path)
    grid = build_pupil_grid(n, SimConfig().radius_ex, SimConfig().radius_in)
    nflag = cfg.solver.nflag

    # Match the search box to the measurement-error prior: per-coefficient
    # bounds of bound_k * sigma. The scalar +/-0.2 default is far wider than
    # |eps| (~0.01-0.04) and leaves NSGA-II unable to converge in 32-D.
    solver_cfg = replace(
        cfg.solver,
        coeff_bound=cfg.bound_k * np.asarray(cfg.meas_sigma, dtype=float),
    )

    # Draw every sample's inputs up front so the results never depend on ``jobs``.
    tasks: list[dict] = []
    for i in range(cfg.n_samples):
        a_true = rand_num(32, cfg.rms_true, nflag, rng)
        collimator = rand_num(32, cfg.rms_collimator, nflag, rng)
        eps = rng.normal(0.0, cfg.meas_sigma)           # paper Eq. 11
        a_meas = a_true + eps                           # paper Eq. 12
        tasks.append(dict(
            psf_m=_forward_single(grid, collimator + a_true, nflag),   # bench PSF-M
            model_base=collimator + a_meas,
            answer_base=a_meas,
            truth_psf=_forward_single(grid, a_true, nflag),            # telescope alone
            n=n, solver_cfg=solver_cfg,
            label=f"[样本 {i + 1}/{cfg.n_samples}]",
        ))

    if jobs and jobs > 1 and len(tasks) > 1:
        return _run_samples_parallel(tasks, jobs, verbose)
    return [_evaluate_sample(**t, verbose=verbose, progress=progress) for t in tasks]


def _run_samples_parallel(tasks: list[dict], jobs: int,
                          verbose: bool) -> list[SampleResult]:
    """Evaluate independent samples across processes (identical results to serial)."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from dataclasses import replace

    fft_workers = max(1, (os.cpu_count() or 1) // jobs)
    print(f"并行求解 {len(tasks)} 个样本: jobs={jobs}, 每进程 FFT 线程={fft_workers}",
          flush=True)
    results: list[SampleResult | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut_to_idx = {
            ex.submit(_evaluate_sample,
                      **{**t, "solver_cfg": replace(t["solver_cfg"],
                                                    fft_workers=fft_workers)},
                      verbose=verbose, progress=False): i
            for i, t in enumerate(tasks)
        }
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            results[i] = fut.result()
            print(f"  完成 {tasks[i]['label']}", flush=True)
    return results  # type: ignore[return-value]


def run_archived_experiment(mat_dir: str | Path, sample_index: int,
                            solver_cfg: SolverConfig,
                            verbose: bool = False,
                            progress: bool = False) -> SampleResult:
    """Compensation experiment on the archived MATLAB run (``matlab/*.mat``).

    PSF-M is ``ansmat(sample_index).PSF`` (contains the unknown physical
    perturbation ``Deta_error``); the model knows the archived true
    coefficients; ground truth is the archived theoretical PSF.
    """
    mat_dir = Path(mat_dir)
    init = sio.loadmat(mat_dir / "initial.mat")["initial"]
    ans = sio.loadmat(mat_dir / "ansmat.mat")["ansmat"]
    anm_parallel = np.asarray(init["parallel_zernike"][0, 0]).ravel()
    anm_gxss = np.asarray(init["GXSS_zernike"][0, 0]).ravel()
    base = anm_parallel + anm_gxss

    psf_m = np.asarray(ans["PSF"][0, sample_index], dtype=np.float64)
    deta = np.asarray(ans["Deta_error"][0, sample_index]).ravel()
    n = psf_m.shape[0]
    return _evaluate_sample(psf_m=psf_m, model_base=base, answer_base=base,
                            truth_psf=psf_m, n=n, solver_cfg=solver_cfg,
                            coef_ref=deta, verbose=verbose,
                            progress=progress, label=f"[归档样本 {sample_index + 1}]")


def summarize(samples: list[SampleResult]) -> dict:
    """Aggregate statistics across samples (paper Figs. 13/15 style)."""
    truth = np.array([s.truth_e for s in samples])
    initial_err = np.array([s.initial_e for s in samples]) - truth
    best_err = np.array([s.comp_e[s.best_index] - s.truth_e for s in samples])
    contains = np.array([
        s.comp_e.min() <= s.truth_e <= s.comp_e.max() for s in samples
    ])
    summary = {
        "truth_e_mean": float(truth.mean()),
        "initial_abs_err_mean": float(np.abs(initial_err).mean()),
        "comp_abs_err_best_mean": float(np.abs(best_err).mean()),
        "comp_interval_contains_truth_frac": float(contains.mean()),
        "per_sample": [
            {
                "truth_e": s.truth_e,
                "initial_e": s.initial_e,
                "comp_e_min": float(s.comp_e.min()),
                "comp_e_max": float(s.comp_e.max()),
                "comp_e_best": float(s.comp_e[s.best_index]),
                "n_pareto": len(s.comp_e),
            }
            for s in samples
        ],
    }
    if all(s.coef_ref is not None for s in samples):
        coef_err = np.array([
            np.linalg.norm(s.x_pareto[s.best_index] - s.coef_ref) for s in samples
        ])
        summary["coef_recovery_err_best_mean"] = float(coef_err.mean())
    return summary
