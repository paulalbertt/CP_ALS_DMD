function [A, fit_hist, info] = cp_als_dmd(X, R, opts)
%CP_ALS_DMD  CP/PARAFAC via ALS with Dynamic Mode Decomposition extrapolation.
%
%   [A, fit_hist, info] = cp_als_dmd(X, R, opts)
%
%INPUTS
%   X      : N-way dense array (double). (This reference implementation
%            expects a dense tensor; adapt MTTKRP for sparse tensors if needed.)
%   R      : Target CP rank (positive integer)
%   opts   : (struct) optional fields
%            .max_iters   - maximum ALS sweeps (default 1000)
%            .tol         - relative objective decrease tolerance (1e-4)
%            .damp        - Tikhonov damping for normal eqs (1e-8)
%            .init        - 'rand' (default) or cell array of factors {A1..AN}
%            .seed        - RNG seed for reproducible init (default [] = no set)
%            .normalize   - true (default): unit-norm columns with weights λ
%            .dmd_win     - DMD window size m (default 5; start when k >= m)
%            .dmd_every   - apply DMD every t sweeps (default 1)
%            .accept_rule - 'decrease' (default) or 'always'
%            .use_els     - true/false (default true): after DMD, do exact
%                           model-space line search between ALS and DMD models
%            .verbose     - 0/1 (default 1)
%
%OUTPUTS
%   A         : cell array {A{1},...,A{N}} of factor matrices (I_n x R)
%               with column normalization and weights λ absorbed into A{1}.
%   fit_hist  : objective history f_k = 1/2 ||X - M_k||_F^2 (vector)
%   info      : struct with details:
%               .lambdas (per-iter, optional), .niters, .fval, .time, counts
%
%NOTES
% - DMD here is a simple least-squares propagator K ≈ Y * pinv(X) built from
%   packed factor snapshots over a sliding window. We align permutation & signs
%   between successive ALS models before packing so snapshots live in a common
%   coordinate system.
% - Exact line search (ELS) is performed in model space between the ALS model
%   M_als and the DMD-predicted model M_dmd: alpha* = <R_als, ΔM> / ||ΔM||^2.
%   This needs only one MTTKRP to evaluate <X, M_dmd>; all other terms come
%   from Gram/cross-Gram matrices.
%
%Reference-quality, dependency-free (no toolboxes) MATLAB code.
%
%--------------------------------------------------------------------------

% ---- defaults ----
if nargin < 3, opts = struct; end
arg = @(name,def) (isfield(opts,name) && ~isempty(opts.(name))) * opts.(name) + ...
                   (~(isfield(opts,name) && ~isempty(opts.(name)))) * def; %#ok<NASGU>
max_iters   = arg('max_iters', 1000);
rtol        = arg('tol', 1e-4);
damp        = arg('damp', 1e-8);
init        = arg('init', 'rand');
seed        = arg('seed', []);
use_norm    = arg('normalize', true);
dmd_win     = max(3, arg('dmd_win', 5));
dmd_every   = max(1, arg('dmd_every', 1));
accept_rule = arg('accept_rule', 'decrease');
use_els     = arg('use_els', true);
verbose     = arg('verbose', 1);

% ---- setup ----
N = ndims(X);
Is = size(X);
if numel(Is) < N, Is = [Is, ones(1,N-numel(Is))]; end

if ~isempty(seed), rng(seed); end

% init factors
A = cell(1,N);
if ischar(opts.init) && strcmpi(opts.init,'rand')
    for n=1:N, A{n} = rand(Is(n), R); end
elseif iscell(init)
    assert(numel(init)==N, 'init cell must have N factors');
    A = init;
    for n=1:N, assert(size(A{n},2)==R, 'factor %d must have R columns', n); end
else
    error('opts.init must be ''rand'' or {A1..AN}');
end

% column normalization and weights λ in A{1}
if use_norm
    [A, lambda] = normalize_columns_with_lambda(A);
else
    lambda = ones(1,R);
end
A{1} = A{1} .* lambda; lambda = ones(1,R); % absorb λ into mode-1

AtA = cellfun(@(Ai) (Ai.'*Ai), A, 'uni', 0);
Xnorm2 = sum(X(:).^2);

% storage
fit_hist = zeros(max_iters,1);
info = struct('niters',0,'fval',NaN,'time',NaN,'accepted_dmd',0,'tried_dmd',0);

% snapshot buffer for DMD
p = sum(Is.*R);                % packed parameter length (A{n} concatenated)
snap_buf = zeros(p, dmd_win);  % columns are snapshots
snap_k = 0;                    % how many stored so far

% helpers for packing/unpacking
pack = @(Acell) pack_factors(Acell);
unpack = @(vec) unpack_factors(vec, Is, R);

% objective helpers
% model norm ||M||^2 from Gram Hadamards
model_norm2 = @(AtAcells) ones(1,R) * hadamard_chain(AtAcells) * ones(R,1);

% ALS bookkeeping for exact values this sweep
xXm_old = NaN;  % <X, M_k> before sweep (captured at first subproblem)
xXm_new = NaN;  % <X, M_{k+1}> after sweep (captured at last subproblem)

% timer
t0 = tic;

for k = 1:max_iters
    A_prev  = A; AtA_prev = AtA;

    % ===== ALS sweep =====
    for n = 1:N
        Gamma = hadamard_chain_exclude(AtA, n);
        M = mttkrp(X, A, n);  % X_(n) * KhatriRao(all factors except n)
        if n==1, xXm_old = trace(A{n}.' * M); end
        % Solve A{n} * Gamma = M  --> A{n} = M / (Gamma + damp*I)
        if damp > 0, G = Gamma + damp*eye(R); else, G = Gamma; end
        A{n} = M / G;
        AtA{n} = A{n}.' * A{n};
        if n==N, xXm_new = trace(A{n}.' * M); end
    end

    % normalize and push weights to mode-1
    if use_norm
        [A, lambda] = normalize_columns_with_lambda(A);
        A{1} = A{1} .* lambda; lambda = ones(1,R);
        for n=1:N, AtA{n} = A{n}.'*A{n}; end
    end

    % objective (ALS model)
    H_pp = hadamard_chain(AtA);
    Mk1_2 = ones(1,R) * H_pp * ones(R,1);
    f_als = 0.5 * (Xnorm2 - 2*xXm_new + Mk1_2);

    % permutation/sign alignment to previous model (for consistent snapshots)
    [A_aligned, P, sgn] = align_to_reference(A, A_prev);
    if any(P ~= 1:R) || any(sgn ~= 1)
        A = A_aligned;
        for n=1:N, AtA{n} = A{n}.'*A{n}; end
    end

    % pack snapshot and update buffer
    xk = pack(A);
    if snap_k < dmd_win
        snap_k = snap_k + 1;
        snap_buf(:, snap_k) = xk;
    else
        snap_buf = [snap_buf(:,2:end), xk]; %#ok<AGROW>
    end

    tried_dmd = false; accepted = false;
    if (k >= dmd_win) && mod(k, dmd_every)==0
        tried_dmd = true;
        Xs = snap_buf(:,1:end-1);  % p x (m-1)
        Ys = snap_buf(:,2:end);
        % least-squares propagator
        Kls = Ys / Xs;             % p x p (implicitly low-rank if m small)
        x_pred = Kls * xk;

        % unpack candidate and (optionally) ELS blend with ALS model
        A_dmd = unpack(x_pred);
        if use_norm
            [A_dmd, lambda_d] = normalize_columns_with_lambda(A_dmd);
            A_dmd{1} = A_dmd{1} .* lambda_d; % absorb
        end
        % compute candidate objective using one MTTKRP for <X, M_dmd>
        AtA_d = cellfun(@(Ai) (Ai.'*Ai), A_dmd, 'uni', 0);
        H_dd  = hadamard_chain(AtA_d);
        Md2   = ones(1,R)*H_dd*ones(R,1);
        M1_d  = mttkrp(X, A_dmd, 1);          % one MTTKRP
        xXm_d = trace(A_dmd{1}.' * M1_d);
        f_dmd = 0.5*(Xnorm2 - 2*xXm_d + Md2);

        if use_els
            % build scalars for exact line search between ALS and DMD models
            % need ||M_als||^2, ||M_dmd||^2, <M_dmd, M_als>, <X, M_als>, <X, M_dmd>
            % we already have Mk1_2 (=||M_als||^2), Md2, xXm_new, xXm_d
            % cross-term via Hadamard of cross-Grams
            H_da = hadamard_chain( cellfun(@(Ad, Aa) (Ad.'*Aa), A_dmd, A, 'uni', 0) );
            dot_da = ones(1,R) * H_da * ones(R,1);
            normDelta2 = Md2 + Mk1_2 - 2*dot_da;
            rDotDelta  = (xXm_d - xXm_new) - (Mk1_2 - dot_da);
            if normDelta2 <= 1e-16
                alpha = 1.0;
            else
                alpha = max(0, min(1.5, rDotDelta / normDelta2));
            end
            % Blend factors in factor space and evaluate f(alpha) cheaply
            A_try = cell(1,N);
            for n=1:N
                A_try{n} = A{n} + alpha * (A_dmd{n} - A{n});
            end
            if use_norm
                [A_try, lambda_t] = normalize_columns_with_lambda(A_try);
                A_try{1} = A_try{1} .* lambda_t;
            end
            AtA_t = cellfun(@(Ai) (Ai.'*Ai), A_try, 'uni', 0);
            Mt2   = ones(1,R) * hadamard_chain(AtA_t) * ones(R,1);
            M1_t  = mttkrp(X, A_try, 1);
            xXm_t = trace(A_try{1}.' * M1_t);
            f_try = 0.5*(Xnorm2 - 2*xXm_t + Mt2);
            if f_try < f_als
                A = A_try; AtA = AtA_t; f_als = f_try; accepted = true;
            elseif f_dmd < f_als
                A = A_dmd; AtA = AtA_d; f_als = f_dmd; accepted = true;
            end
        else
            % simple rule: accept DMD only if it decreases f
            if strcmpi(accept_rule,'always') || (f_dmd < f_als)
                A = A_dmd; AtA = AtA_d; f_als = f_dmd; accepted = true;
            end
        end
    end

    fit_hist(k) = f_als;
    if verbose
        if tried_dmd
            fprintf('iter %4d  f=%.6e  DMD:%s\n', k, f_als, tern(accepted,'accepted','rejected'));
        else
            fprintf('iter %4d  f=%.6e\n', k, f_als);
        end
    end

    % stopping
    if k > 1
        rel = abs(fit_hist(k) - fit_hist(k-1)) / max(1, fit_hist(k-1));
        if rel < rtol
            fit_hist = fit_hist(1:k);
            break;
        end
    end
end

info.niters = numel(fit_hist);
info.fval   = fit_hist(end);
info.time   = toc(t0);
info.tried_dmd     = sum(~isnan(fit_hist)); %#ok<NASGU>
info.accepted_dmd  = NaN; % (left for extension)

end % main

% ======================================================================
% ------------------------- Helper functions ---------------------------
% ======================================================================

function Y = mttkrp(X, A, n)
%MTTKRP Matricized-Tensor Times Khatri-Rao Product (mode-n)
% X: dense tensor, A: cell {A1..AN}
% Returns: Y = X_(n) * (kr(A_N,...,A_{n+1},A_{n-1},...,A_1))  (I_n x R)
    N = ndims(X);
    Is = size(X); if numel(Is) < N, Is = [Is, ones(1,N-numel(Is))]; end
    R = size(A{1},2);
    % Build Khatri-Rao of all modes except n (in reverse mode order matching unfolding)
    other = [N:-1:n+1, n-1:-1:1];
    KR = A{other(1)};
    for k = 2:numel(other)
        KR = khatrirao(KR, A{other(k)});
    end
    % Unfold X in mode n
    perm = [n, 1:n-1, n+1:N];
    Xn = reshape(permute(X, perm), Is(n), []);
    Y = Xn * KR;
end

function C = khatrirao(A, B)
%KHATRIRAO Column-wise Kronecker product (A ⊙ B)
    assert(size(A,2)==size(B,2), 'KR: column counts must match');
    R = size(A,2);
    C = zeros(size(A,1)*size(B,1), R);
    for r=1:R
        C(:,r) = kron(A(:,r), B(:,r));
    end
end

function H = hadamard_chain(Gs)
%HADAMARD_CHAIN elementwise product of R-by-R matrices in Gs (cell)
    H = ones(size(Gs{1}));
    for i=1:numel(Gs)
        H = H .* Gs{i};
    end
end

function H = hadamard_chain_exclude(Gs, idx)
%HADAMARD_CHAIN_EXCLUDE hadamard of all except idx
    R = size(Gs{1},1);
    H = ones(R,R);
    for i=1:numel(Gs)
        if i==idx, continue; end
        H = H .* Gs{i};
    end
end

function [A, lambda] = normalize_columns_with_lambda(A)
% Normalize columns of each factor to unit 2-norm, collect weights in λ
    N = numel(A); R = size(A{1},2);
    lambda = ones(1,R);
    for r=1:R
        prodn = 1;
        for n=1:N
            nr = norm(A{n}(:,r));
            if nr==0
                A{n}(:,r) = randn(size(A{n},1),1)*1e-6; nr = norm(A{n}(:,r));
            end
            A{n}(:,r) = A{n}(:,r) / nr;
            prodn = prodn * nr;
        end
        lambda(r) = prodn;
    end
end

function x = pack_factors(A)
% Concatenate all factor matrices column-wise into a vector
    N = numel(A);
    x = [];
    for n=1:N
        x = [x; A{n}(:)]; %#ok<AGROW>
    end
end

function A = unpack_factors(x, Is, R)
% Reconstruct factors from packed vector x given sizes Is and rank R
    N = numel(Is);
    A = cell(1,N);
    s = 0;
    for n=1:N
        m = Is(n)*R;
        A{n} = reshape(x(s+(1:m)), Is(n), R);
        s = s + m;
    end
end

function [A2, P, sgn] = align_to_reference(A, Aref)
%ALIGN_TO_REFERENCE Align permutation and signs of A to Aref.
% Greedy assignment maximizing product of mode-wise absolute correlations.
    N = numel(A);
    R = size(A{1},2);
    C = zeros(R,R);
    for i=1:R
        for j=1:R
            val = 1;
            for n=1:N
                ai = A{n}(:,i); aj = Aref{n}(:,j);
                ci = (ai.'*aj) / (norm(ai)*norm(aj) + eps);
                val = val * abs(ci);
            end
            C(i,j) = val;
        end
    end
    % greedy matching
    P = zeros(1,R); used_r = false(1,R); used_c = false(1,R);
    for r=1:R %#ok<FXSET>
        Ctemp = C; Ctemp(used_r,:) = -Inf; Ctemp(:,used_c) = -Inf;
        [mx, idx] = max(Ctemp(:)); %#ok<ASGLU>
        if ~isfinite(mx) 
            break; 
        end
        [i,j] = ind2sub([R,R], idx);
        P(i) = j; used_r(i)=true; used_c(j)=true;
    end
    % apply permutation and sign flips
    A2 = cell(size(A)); sgn = ones(1,R);
    for i=1:R
        j = P(i); if j==0, j=i; end
        s = 1;
        % choose sign based on mode-1 correlation
        if (A{1}(:,i).'*Aref{1}(:,j)) < 0, s = -1; end
        sgn(i) = s;
        for n=1:N
            A2{n}(:,i) = s * A{n}(:,i);
        end
    end
    % reorder columns to match reference order
    for n=1:N
        A2{n} = A2{n}(:, 1:R);
        A2{n} = A2{n}(:, argsort(P));
    end
end

function idx = argsort(v)
%ARGSORT return indices that sort v ascending
    [~, idx] = sort(v, 'ascend');
end

function out = tern(cond, a, b)
%TERN Ternary helper
    if cond, out = a; else, out = b; end
end
