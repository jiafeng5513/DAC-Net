#!/usr/bin/env python3
"""Compare the Python port against Octave-run original MATLAB components.

Expects ``/tmp/octave_components.mat`` produced by ``verify_components.m``
and the injected-coefficient run outputs. Exits non-zero on mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elli_error_generate import (
    SimConfig,
    cac_ellipticity,
    rand_num,
    run_simulation,
    zernike_poly,
)

MATLAB_DIR = Path("/Users/anna/WorkSpace/Projects/CSST/Elli_error_generate/matlab")
FAILURES = []


def check(name: str, got, expected, rtol=1e-12, atol=1e-14) -> None:
    got = np.asarray(got, dtype=np.float64).squeeze()
    expected = np.asarray(expected, dtype=np.float64).squeeze()
    same_shape = got.shape == expected.shape
    ok = same_shape and np.allclose(got, expected, rtol=rtol, atol=atol)
    max_diff = np.abs(got - expected).max() if same_shape else np.inf
    print(f"{'PASS' if ok else 'FAIL'}  {name:42s} max_abs_diff={max_diff:.3e}")
    if not ok:
        FAILURES.append(name)


def main() -> None:
    comp = sio.loadmat("/tmp/octave_components.mat")
    rng_vec = comp["rng_vec"].ravel()

    # ---- 1. zernike_fun: scalar + grid, both orderings ----
    check("zernike scalar nflag=0", zernike_poly(comp["r_s"][0, 0], comp["t_s"][0, 0], rng_vec, 0), comp["z0_scalar"])
    check("zernike scalar nflag=1", zernike_poly(comp["r_s"][0, 0], comp["t_s"][0, 0], rng_vec, 1), comp["z1_scalar"])
    rr, tt = comp["rr"].ravel(), comp["tt"].ravel()
    R, T = np.meshgrid(rr, tt)  # match [i_angle, i_radius] layout
    check("zernike grid nflag=0", zernike_poly(R, T, rng_vec, 0), comp["z0_grid"])
    check("zernike grid nflag=1", zernike_poly(R, T, rng_vec, 1), comp["z1_grid"])

    # ---- 2. cac_ellipticity_fun on synthetic PSFs ----
    check("ellipticity axis-aligned gaussian", cac_ellipticity(comp["psf_a"], 1), comp["ell_a"])
    check("ellipticity rotated gaussian", cac_ellipticity(comp["psf_b"], 1), comp["ell_b"])

    # ---- 3. RandNum formula with injected raw vector ----
    check("rand_num nflag=1 (fringe)", rand_num(32, 1 / 14, 1, raw_vec=comp["raw42"].ravel()), comp["anm_fringe"].ravel())
    check("rand_num nflag=0 (standard)", rand_num(32, 1 / 14, 0, raw_vec=comp["raw43"].ravel()), comp["anm_std"].ravel())
    # RMS property: sqrt(sum((Anm/sqrt(Coff))^2)) == RMS_Level
    from elli_error_generate.rand_num import COFF_ZFC_Z5_Z36
    anm = rand_num(32, 1 / 14, 1, raw_vec=comp["raw42"].ravel())
    rms_actual = float(np.sqrt(np.sum((anm / np.sqrt(COFF_ZFC_Z5_Z36)) ** 2)))
    check("rand_num fringe RMS property", rms_actual, 1 / 14, rtol=1e-12, atol=1e-15)

    # ---- 4. Full pipeline vs the real MATLAB run (injected coefficients) ----
    ref = np.load("/tmp/matlab_coeffs.npz")
    coeffs = {k: ref[k] for k in ("Anm_parallel", "Anm_GXSS", "Deta_error")}
    config = SimConfig(sim_total_num=2, nflag=1)
    initial, ansmat = run_simulation(config, MATLAB_DIR / "ExitPupil.mat", coeffs)

    init_mat = sio.loadmat(MATLAB_DIR / "initial.mat")["initial"]
    getf = lambda name: np.asarray(init_mat[name][0, 0])

    # Tolerance covers cross-libm exp() noise ~2*pi*1e-14 on phase-derived fields.
    phase_tol = {"rtol": 1e-10, "atol": 1e-12}
    check("initial.parallel_poly_zernike", initial["parallel_poly_zernike"], getf("parallel_poly_zernike"), **phase_tol)
    check("initial.GXSS_poly_zernike", initial["GXSS_poly_zernike"], getf("GXSS_poly_zernike"), **phase_tol)
    check("initial.parallel_GXSS_poly_zernike", initial["parallel_GXSS_poly_zernike"], getf("parallel_GXSS_poly_zernike"), **phase_tol)
    check("initial.parallel_aberrated (real)", initial["parallel_aberrated"].real, getf("parallel_aberrated").real, **phase_tol)
    check("initial.parallel_aberrated (imag)", initial["parallel_aberrated"].imag, getf("parallel_aberrated").imag, **phase_tol)
    check("initial.GXSS_aberrated (real)", initial["GXSS_aberrated"].real, getf("GXSS_aberrated").real, **phase_tol)
    check("initial.GXSS_aberrated (imag)", initial["GXSS_aberrated"].imag, getf("GXSS_aberrated").imag, **phase_tol)
    check("initial.parallel_GXSS_aberrated (real)", initial["parallel_GXSS_aberrated"].real, getf("parallel_GXSS_aberrated").real, **phase_tol)
    check("initial.parallel_GXSS_aberrated (imag)", initial["parallel_GXSS_aberrated"].imag, getf("parallel_GXSS_aberrated").imag, **phase_tol)
    check("initial.parallel_GXSS_PSF", initial["parallel_GXSS_PSF"], getf("parallel_GXSS_PSF"))
    check("initial.ellipticity_GXSS", initial["ellipticity_GXSS"], np.asarray(getf("ellipticity_GXSS")).ravel()[0])

    ans_mat = sio.loadmat(MATLAB_DIR / "ansmat.mat")["ansmat"]
    for i in range(2):
        psf_ref = np.asarray(ans_mat["PSF"][0, i])
        ell_ref = float(np.asarray(ans_mat["ellipticity_parallel_GXSS"][0, i]).ravel()[0])
        deta_ref = np.asarray(ans_mat["Deta_error"][0, i]).ravel()
        check(f"ansmat[{i}].PSF", ansmat[i]["PSF"], psf_ref)
        check(f"ansmat[{i}].ellipticity", ansmat[i]["ellipticity_parallel_GXSS"], ell_ref)
        check(f"ansmat[{i}].Deta_error", ansmat[i]["Deta_error"], deta_ref)

    # Reference PSFs for the Octave-side .mat roundtrip test.
    sio.savemat(
        "/tmp/matlab_coeffs_reference.mat",
        {
            "psf_mc_matlab_1": ref["psf_mc_matlab"][0],
            "psf_mc_matlab_2": ref["psf_mc_matlab"][1],
        },
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("All component and pipeline comparisons PASSED.")


if __name__ == "__main__":
    main()
