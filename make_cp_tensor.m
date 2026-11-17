function [X, Atrue, Xclean] = make_cp_tensor(I, R, congr, noise_rel, seed)
%MAKE_CP_TENSOR  Synthetic rank-R CP tensor with controllable factor collinearity.
%
%   [X, Atrue, Xclean] = make_cp_tensor(I, R, congr, noise_rel, seed)
%
%   I          vector of mode sizes, e.g. [50 50 50]
%   R          rank
%   congr      target pairwise column correlation (congruence) of the factor columns,
%              the same in every mode (Tomasi & Bro 2006 style).  0 = independent
%              columns, 0.9 = strongly collinear ("swamp" regime for ALS).
%   noise_rel  ||E||_F / ||Xclean||_F of the additive Gaussian noise (0 = none)
%   seed       rng seed
%
%   Columns of every factor have unit norm (all rank-1 terms have equal weight).
if nargin >= 5 && ~isempty(seed), rng(seed); end
N = numel(I); Atrue = cell(1, N);
Sigma = (1 - congr) * eye(R) + congr * ones(R);
L = chol(Sigma, 'lower');
for n = 1:N
    A = randn(I(n), R) * L';                 % rows ~ N(0, Sigma) -> column correlation congr
    A = A ./ sqrt(sum(A.^2, 1));
    Atrue{n} = A;
end
Xclean = cp_full(Atrue);
X = Xclean;
if noise_rel > 0
    E = randn(size(X));
    X = X + noise_rel * (norm(Xclean(:)) / norm(E(:))) * E;
end
end

function X = cp_full(A)
N = numel(A); R = size(A{1}, 2); Is = cellfun(@(M) size(M, 1), A);
K = A{2};
for n = 3:N
    K = reshape(reshape(K, [], 1, R) .* reshape(A{n}, 1, [], R), [], R);
end
X = reshape(A{1} * K', Is);
end
