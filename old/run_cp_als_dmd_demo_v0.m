function run_cp_als_dmd_demo()
%RUN_CP_ALS_DMD_DEMO  Quick driver to test CP-ALS + DMD extrapolation
%
% This script:
%   1) Generates a random low-rank CP tensor of chosen size
%   2) Adds optional Gaussian noise
%   3) Runs cp_als_dmd(X, R, opts)
%   4) Plots objective curves and reports reconstruction error
%
% REQUIREMENTS
%   - cp_als_dmd.m should be on your MATLAB path (same folder is fine)
%

%% User settings
I = [40, 35, 30];   % tensor size (3-way example)
R_true = 5;         % true rank used to synthesize data
R_est  = 5;         % rank used in factorization (can mismatch to test robustness)
noise_sigma = 0; % std of additive Gaussian noise (0 = noiseless)
max_iters   = 500;

% DMD settings
opts = struct();
opts.max_iters = max_iters;
opts.tol       = 1e-8;
opts.damp      = 1e-10;
opts.normalize = true;
opts.dmd_win   = 10;        % window size (>=3)
opts.dmd_every = 1;        % apply every sweep
opts.accept_rule = 'decrease';
opts.init = 'rand';
opts.use_els   = true;     % exact model-space line search after DMD
opts.verbose   = 1;        % print per-iter f values
opts.seed      = 1;        % for reproducibility

%% 1) Build a random CP model and dense tensor
fprintf('Generating synthetic CP tensor of size %s with rank %d...\n', mat2str(I), R_true);
Atrue = cell(1, numel(I));
for n=1:numel(I)
    Atrue{n} = randn(I(n), R_true);
    % normalize columns for a well-scaled model
    for r=1:R_true
        Atrue{n}(:,r) = Atrue{n}(:,r) / norm(Atrue{n}(:,r));
    end
end
% random positive weights (absorbed into mode-1)
lambda = abs(randn(1, R_true));
Atrue{1} = Atrue{1} .* lambda;  %#ok<NASGU>

X = full_from_cp(Atrue);   % dense tensor
if noise_sigma > 0
    X = X + noise_sigma * randn(size(X));
end

%% 2) Factorize with our algorithm
fprintf('Running cp_als_dmd (R_est = %d)...\n', R_est);
opts.init = 'rand';
[A_est, f_hist, info] = cp_als_dmd(X, R_est, opts);

%% 3) Report numbers
fprintf('Finished in %d sweeps, final f = %.6e\n', numel(f_hist), f_hist(end));
rel_err = sqrt(2*f_hist(end)) / norm(X(:));
fprintf('Relative reconstruction error: %.3e\n', rel_err);

%% 4) Plot objective curve
figure; plot(f_hist, 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Objective f = 1/2 ||X - M||_F^2');
title('CP-ALS + DMD extrapolation'); grid on;

%% 5) (optional) Compare factors up to permutation/sign
%if R_est == R_true
    fprintf('Computing simple correlation scores (mode-wise, max over columns)...\n');
    for n=1:numel(I)
        C = corr(A_est{n},Atrue{n}); %#ok<CORR>
        s = max(C(:));
        fprintf('  Mode %d: max |corr| = %.3f\n', n, s);
    end
%end

end

%% ---------------- local helpers ----------------
function X = full_from_cp(A)
%FULL_FROM_CP Build a dense tensor from CP factors (weights absorbed in mode-1)
    N = numel(A);
    Is = cellfun(@(Ai) size(Ai,1), A);
    R = size(A{1},2);
    X = zeros(Is);
    for r=1:R
        outer = A{1}(:,r);
        for n=2:N
            outer = ktimes(outer, A{n}(:,r));
        end
        X = X + outer;
    end
end

function Y = ktimes(a, b)
%KTIMES n-D outer product accumulator
    if isvector(a) && isvector(b)
        Y = reshape(a, [], 1) * reshape(b, 1, []);
    else
        % expand a by outer product with vector b on a new trailing mode
        %Y = bsxfun(@times, a, reshape(b, 1, [])); %#ok<BSXFUN>
        Y = bsxfun(@times, reshape(b, [], 1),reshape(a,[1 size(a)])); %needs fix
        Y = permute(Y,[2 3 1]);
    end
end
