#!/usr/bin/env python3
"""Command-line entry point — port of ``sim_mainV4_Monte_Carlo.m``.

Example
-------
    python sim_main_v4.py --n 2 --seed 42 --out-dir ./output
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_PARENT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_PARENT))

from elli_error_generate import SimConfig, run_simulation, save_results

DEFAULT_DATA = PACKAGE_PARENT.parent / "matlab" / "ExitPupil.mat"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte-Carlo simulation of CSST PSF ellipticity errors "
        "(Python port of sim_mainV4_Monte_Carlo.m)."
    )
    p.add_argument("--n", type=int, default=2, help="number of Monte-Carlo samples")
    p.add_argument("--nflag", type=int, default=1, choices=[0, 1],
                   help="0 = standard Zernike, 1 = fringe Zernike")
    p.add_argument("--rms-parallel", type=float, default=1 / 14,
                   help="collimator wavefront RMS (waves)")
    p.add_argument("--rms-gxss", type=float, default=1 / 14,
                   help="interferometer wavefront RMS (waves)")
    p.add_argument("--radius-ex", type=float, default=231, help="pupil outer radius (px)")
    p.add_argument("--radius-in", type=float, default=0, help="pupil inner radius (px)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="path to ExitPupil.mat")
    p.add_argument("--out-dir", type=Path, default=Path("."),
                   help="directory for ansmat/initial outputs")
    p.add_argument("--no-mat", action="store_true",
                   help="only write .npz outputs (skip MATLAB-compatible .mat)")
    p.add_argument("--coeffs", type=Path, default=None,
                   help="optional .npz with fixed Anm_parallel / Anm_GXSS / "
                        "Deta_error for deterministic reproduction")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise SystemExit(f"ExitPupil data file not found: {args.data}")

    coeffs = None
    if args.coeffs is not None:
        data = np.load(args.coeffs)
        coeffs = {k: data[k] for k in ("Anm_parallel", "Anm_GXSS", "Deta_error")}

    config = SimConfig(
        sim_total_num=args.n,
        nflag=args.nflag,
        rms_parallel=args.rms_parallel,
        rms_gxss=args.rms_gxss,
        radius_ex=args.radius_ex,
        radius_in=args.radius_in,
        seed=args.seed,
    )

    t0 = time.perf_counter()
    initial, ansmat = run_simulation(config, args.data, coeffs)
    elapsed = time.perf_counter() - t0

    save_results(initial, ansmat, args.out_dir, write_mat=not args.no_mat)

    ell = [a["ellipticity_parallel_GXSS"] for a in ansmat]
    print(f"Done in {elapsed:.1f} s — {args.n} sample(s), grid "
          f"{initial['parallel_GXSS_PSF'].shape[0]}x{initial['parallel_GXSS_PSF'].shape[1]}")
    print(f"ellipticity_GXSS (initial)      = {initial['ellipticity_GXSS']:.6f}")
    for i, e in enumerate(ell, 1):
        print(f"ellipticity_parallel_GXSS ({i}) = {e:.6f}")
    print(f"outputs written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
