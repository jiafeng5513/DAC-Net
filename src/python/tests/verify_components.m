% Cross-language component verification harness (run with Octave).
% Exercises the ORIGINAL MATLAB functions with fixed inputs and saves
% the results for comparison against the Python port.
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'matlab'));
out = '/tmp/octave_components.mat';

%% 1. zernike_fun: scalar and small grid, both orderings
rng_vec = linspace(-0.9, 0.9, 32)';   % fixed deterministic coefficient vector
r_s = 0.37; t_s = 1.23;
z0_scalar = zernike_fun(r_s, t_s, rng_vec, 0);
z1_scalar = zernike_fun(r_s, t_s, rng_vec, 1);

rr = linspace(0.05, 0.95, 5);         % 5 radii
tt = linspace(-pi, pi, 7);            % 7 angles
z0_grid = zeros(numel(tt), numel(rr));
z1_grid = zeros(numel(tt), numel(rr));
for i = 1:numel(tt)
  for j = 1:numel(rr)
    z0_grid(i, j) = zernike_fun(rr(j), tt(i), rng_vec, 0);
    z1_grid(i, j) = zernike_fun(rr(j), tt(i), rng_vec, 1);
  end
end

%% 2. cac_ellipticity_fun on synthetic PSFs
[X, Y] = meshgrid(-200:200, -200:200);
% axis-aligned elliptical Gaussian
psf_a = exp(-(X.^2 / (2 * 60^2) + Y.^2 / (2 * 45^2)));
% rotated (30 deg) elliptical Gaussian with slightly offset peak
c = cosd(30); s = sind(30);
Xr = c * X + s * Y;  Yr = -s * X + c * Y;
psf_b = exp(-(((Xr - 3).^2) / (2 * 60^2) + ((Yr + 2).^2) / (2 * 45^2)));
ell_a = cac_ellipticity_fun(psf_a, 1);
ell_b = cac_ellipticity_fun(psf_b, 1);

%% 3. RandNum with captured raw uniform vector (fringe & standard)
rand('twister', 42); raw42 = (rand(32, 1) - 0.5) * 2;
rand('twister', 42); anm_fringe = RandNum(32, 1 / 14, 1);  % same draw sequence
rand('twister', 43); raw43 = (rand(32, 1) - 0.5) * 2;
rand('twister', 43); anm_std = RandNum(32, 1 / 14, 0);

save('-mat7-binary', out, 'rng_vec', 'r_s', 't_s', 'z0_scalar', 'z1_scalar', ...
     'rr', 'tt', 'z0_grid', 'z1_grid', ...
     'psf_a', 'psf_b', 'ell_a', 'ell_b', ...
     'raw42', 'raw43', 'anm_fringe', 'anm_std');
printf('components saved to %s\n', out);
