#!/usr/bin/env python3
"""Compensation solver CLI — reproduces the paper's optimization experiment.

Examples
--------
# Synthetic experiment, paper-scale NSGA-II settings:
python solve_compensation.py demo --samples 4 --pop 180 --gen 120 --seed 1

# Quick smoke test (small population/generation count):
python solve_compensation.py demo --samples 2 --pop 40 --gen 30

# Use the archived MATLAB run as PSF-M (sample 1):
python solve_compensation.py archived --sample 1 --pop 60 --gen 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_PARENT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_PARENT))

from elli_error_generate.experiment import (
    ExperimentConfig,
    run_archived_experiment,
    run_synthetic_experiment,
    summarize,
)
from elli_error_generate.solver import SolverConfig

DEFAULT_DATA = PACKAGE_PARENT.parent / "matlab" / "ExitPupil.mat"
DEFAULT_MAT_DIR = PACKAGE_PARENT.parent / "matlab"


def solver_config_from(args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(
        pop_size=args.pop,
        n_gen=args.gen,
        crossover_prob=args.cx_prob,
        mutation_prob=args.mut_prob,
        coeff_bound=args.bound,
        seed=args.seed,
        fast=args.fast,
        chunk=args.chunk,
    )


def print_summary(summary: dict) -> None:
    print("\n=== 汇总 (across samples) ===")
    print(f"真值椭率均值 truth_e_mean                = {summary['truth_e_mean']:.6f}")
    print(f"未补偿模型椭率误差 |Δe| 均值 (初始值)     = {summary['initial_abs_err_mean']:.6f}")
    print(f"补偿后椭率误差 |Δe| 均值 (Pareto 最优)   = {summary['comp_abs_err_best_mean']:.6f}")
    print(f"Pareto 区间包含真值的比例                = {summary['comp_interval_contains_truth_frac']:.2f}")
    if "coef_recovery_err_best_mean" in summary:
        print(f"系数恢复误差 ||x - Deta|| 均值           = {summary['coef_recovery_err_best_mean']:.6f}")
    print("\n逐样本:")
    for i, s in enumerate(summary["per_sample"], 1):
        print(f"  样本{i}: 真值={s['truth_e']:.6f}  初始={s['initial_e']:.6f} "
              f" 补偿区间=[{s['comp_e_min']:.6f}, {s['comp_e_max']:.6f}]"
              f" 最优={s['comp_e_best']:.6f} (n_pareto={s['n_pareto']})")


def cmd_demo(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig(n_samples=args.samples,
                           solver=solver_config_from(args), seed=args.seed,
                           bound_k=args.bound_k)
    t0 = time.perf_counter()
    samples = run_synthetic_experiment(cfg, DEFAULT_DATA, verbose=args.verbose,
                                       progress=args.progress, jobs=args.jobs)
    elapsed = time.perf_counter() - t0
    summary = summarize(samples)
    print_summary(summary)
    print(f"\nelapsed: {elapsed:.1f} s")
    _save(args, summary, samples)


def cmd_archived(args: argparse.Namespace) -> None:
    solver_cfg = solver_config_from(args)
    t0 = time.perf_counter()
    sample = run_archived_experiment(DEFAULT_MAT_DIR, args.sample, solver_cfg,
                                     verbose=args.verbose, progress=args.progress)
    elapsed = time.perf_counter() - t0
    summary = summarize([sample])
    print_summary(summary)
    print(f"\nelapsed: {elapsed:.1f} s")
    _save(args, summary, [sample])


def _save(args: argparse.Namespace, summary: dict, samples) -> None:
    if args.out is None:
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "compensation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        out / "compensation_results.npz",
        x_pareto=np.concatenate([s.x_pareto for s in samples]),
        f_pareto=np.concatenate([s.f_pareto for s in samples]),
        comp_e=np.concatenate([s.comp_e for s in samples]),
        comp_e1=np.concatenate([s.comp_e1 for s in samples]),
        truth_e=np.array([s.truth_e for s in samples]),
        initial_e=np.array([s.initial_e for s in samples]),
        sample_index=np.concatenate([
            np.full(len(s.comp_e), i) for i, s in enumerate(samples)
        ]),
    )
    print(f"saved: {out / 'compensation_summary.json'}, {out / 'compensation_results.npz'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def add_solver_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--pop", type=int, default=180, help="NSGA-II population (paper: 180)")
        sp.add_argument("--gen", type=int, default=120, help="generations (paper: 120)")
        sp.add_argument("--cx-prob", type=float, default=1.0, help="SBX crossover probability (paper: 1)")
        sp.add_argument("--mut-prob", type=float, default=0.1, help="polynomial mutation probability (paper: 0.1)")
        sp.add_argument("--bound", type=float, default=0.2, help="|compensation coefficient| bound (waves); archived mode / fallback")
        sp.add_argument("--bound-k", type=float, default=6.0,
                        help="demo search bound per coefficient = bound_k * meas_sigma (default 6)")
        sp.add_argument("--fast", action="store_true",
                        help="complex64/float32 FFT path (~2x faster, less memory)")
        sp.add_argument("--chunk", type=int, default=16,
                        help="PSF batch size per FFT (higher = more parallelism/memory)")
        sp.add_argument("--jobs", type=int, default=1,
                        help="parallel worker processes across samples (demo only)")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--verbose", action="store_true", help="pymoo per-generation output")
        sp.add_argument("--no-progress", dest="progress", action="store_false",
                        help="disable the per-generation progress line (on by default)")
        sp.set_defaults(progress=True)
        sp.add_argument("--out", type=Path, default=None, help="save results here")

    sp_demo = sub.add_parser("demo", help="synthetic experiment (fresh MC samples)")
    sp_demo.add_argument("--samples", type=int, default=4)
    add_solver_args(sp_demo)
    sp_demo.set_defaults(func=cmd_demo)

    sp_arch = sub.add_parser("archived", help="use matlab/ansmat.mat sample as PSF-M")
    sp_arch.add_argument("--sample", type=int, default=1, help="ansmat sample index (1-based)")
    add_solver_args(sp_arch)
    sp_arch.set_defaults(func=cmd_archived)

    args = p.parse_args()
    if hasattr(args, "sample"):
        args.sample -= 1  # convert to 0-based
    args.func(args)


if __name__ == "__main__":
    main()
