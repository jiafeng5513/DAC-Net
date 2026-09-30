% Verify the Python-written .mat files load as proper structs in Octave/MATLAB
% and that the stored ellipticity is consistent with the stored PSF.
addpath('/Users/anna/WorkSpace/Projects/CSST/Elli_error_generate/matlab');
d = '/Users/anna/WorkSpace/Projects/CSST/Elli_error_generate/python/test_output';
S = load(fullfile(d, 'ansmat.mat'));
I = load(fullfile(d, 'initial.mat'));

assert(isstruct(S.ansmat) && size(S.ansmat, 2) == 2, ...
       'ansmat must be a 1x2 struct array');
assert(isstruct(I.initial), 'initial must be a struct');

% Recompute ellipticity from the stored PSF with the original function.
ell_recalc = cac_ellipticity_fun(S.ansmat(1).PSF, 1);
ell_stored = S.ansmat(1).ellipticity_parallel_GXSS;
printf('ansmat(1): stored ellipticity = %.10f, recalculated = %.10f, diff = %.2e\n', ...
       ell_stored, ell_recalc, abs(ell_stored - ell_recalc));
assert(abs(ell_stored - ell_recalc) < 1e-12, 'stored vs recalculated ellipticity mismatch');

% The PSF written by Python must equal the real MATLAB-run PSF bit-for-bit
% when deterministic coefficients were injected.
R = load('/tmp/matlab_coeffs_reference.mat');
diff0 = max(max(abs(S.ansmat(1).PSF - R.psf_mc_matlab_1)));
diff1 = max(max(abs(S.ansmat(2).PSF - R.psf_mc_matlab_2)));
printf('ansmat(1).PSF vs MATLAB run: max abs diff = %.3e\n', diff0);
printf('ansmat(2).PSF vs MATLAB run: max abs diff = %.3e\n', diff1);
assert(diff0 < 1e-12 && diff1 < 1e-12, 'PSF mismatch vs original MATLAB run');

printf('OK: ansmat fields: %s\n', strjoin(fieldnames(S.ansmat), ', '));
printf('OK: initial fields: %s\n', strjoin(fieldnames(I.initial), ', '));
