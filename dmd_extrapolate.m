function [pdir, lam, ok, info] = dmd_extrapolate(dX, dG, rk, sv_tol, amp_max)
%DMD_EXTRAPOLATE  Fixed-point extrapolation direction from exact DMD on snapshot pairs.
%
%   [pdir, lam, ok, info] = dmd_extrapolate(dX, dG, rk, sv_tol, amp_max)
%
% Consider a fixed-point iteration x -> G(x) that is approximately affine near its
% fixed point x*:  G(x) ~ x* + J (x - x*).  For ANY two points x_j, x_{j+1}
%       dG_j = G(x_{j+1}) - G(x_j) ~ J (x_{j+1} - x_j) = J dX_j ,
% and for the current residual r = G(x) - x one has  x* = G(x) + J (I - J)^{-1} r.
%
% INPUTS
%   dX, dG   p x q matrices whose columns are the pairs (x_{j+1}-x_j, G(x_{j+1})-G(x_j))
%   rk       current residual r = G(x) - x  (p x 1)
%   sv_tol   singular values of dX below sv_tol*sigma_1 are discarded      (e.g. 1e-6)
%   amp_max  cap on the modal gain |lam/(1-lam)|                            (e.g. 1e3)
%
% OUTPUTS
%   pdir     direction p such that x* ~ G(x) + p  (real, p x 1); the caller should
%            still safeguard the step, e.g. by a line search along p
%   lam      DMD eigenvalues (estimated eigenvalues of J on the snapshot subspace)
%   ok       false if the window was degenerate (then pdir = [])
%   info     struct with U (POD basis), W (eigenvectors of the reduced operator),
%            gain (applied modal gains) and coef (residual in modal coordinates)
%
% Exact DMD (Tu et al. 2014): dX = U S V',  Atil = U' dG V S^{-1} = W diag(lam) W^{-1}.
% The extrapolation uses the projected operator J ~ U Atil U' and the Neumann series
% J (I-J)^{-1} = sum_{i>=1} J^i, i.e. a modal gain lam/(1-lam).  Modes with |lam| >= 1
% do not converge and are dropped; the gain is capped at amp_max to limit the influence
% of spuriously slow modes (the line search of the caller fixes the overall step length).
pdir = []; lam = []; ok = false; info = struct();
[U, S, V] = svd(dX, 'econ');
s = diag(S);
if isempty(s) || s(1) <= 0, return; end
r = sum(s > sv_tol * s(1));
U = U(:, 1:r); V = V(:, 1:r); s = s(1:r);
Atil = (U' * dG * V) ./ s';                      % r x r reduced operator U' J U
[W, L] = eig(Atil);
lam = diag(L);
if any(~isfinite(lam)) || rcond(W) < 1e-14, return; end
coef = W \ (U' * rk);                            % residual in DMD-mode coordinates
gain = lam ./ (1 - lam);
gain(abs(lam) >= 1) = 0;                         % non-converging modes dropped
big = abs(gain) > amp_max;
gain(big) = gain(big) .* (amp_max ./ abs(gain(big)));
pdir = real(U * (W * (gain .* coef)));
ok = all(isfinite(pdir)) && any(pdir ~= 0);
info = struct('U', U, 'W', W, 'gain', gain, 'coef', coef, 'Atil', Atil);
end
