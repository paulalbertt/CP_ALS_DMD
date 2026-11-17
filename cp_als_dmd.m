function [A, out] = cp_als_dmd(X, R, opts)
%CP_ALS_DMD  CP/PARAFAC by alternating least squares with safeguarded acceleration.
%
%   [A, out] = cp_als_dmd(X, R)          plain ALS
%   [A, out] = cp_als_dmd(X, R, opts)    opts.accel selects the accelerator
%
% Fits the rank-R CP model  M(A) = sum_r a1_r o a2_r o ... o aN_r  to the dense N-way
% array X by minimising  f(A) = 1/2 ||X - M(A)||_F^2.  One iteration is one ALS sweep
% over the N modes (one MTTKRP per mode) followed by an optional, safeguarded
% acceleration step.  An accelerated point is accepted only if its objective is lower
% than the objective reached by the ALS sweep, so every method is monotone in f.
%
% THEORY (why DMD can be used here).  Write one ALS sweep plus gauge fixing (unit-norm
% columns in modes 2..N, weights in mode 1) as the fixed-point map  x -> G(x)  on the
% packed factor vector x = [A1(:); ...; AN(:)].  Near a minimiser x*, G is approximately
% affine, G(x) ~ x* + J (x - x*), so the residual r(x) = G(x) - x = (J - I)(x - x*) and
%
%        x* = G(x) + J (I - J)^{-1} r(x).                                   (1)
%
% All accelerators below are approximations of the operator J (I - J)^{-1} acting on
% the current residual r_k = G(x_k) - x_k:
%
%   'none'  plain ALS                                              (operator = 0)
%   'bro'   Harshman/Bro line search as in the N-way toolbox:  x_k + s r_k with
%           s = k^(1/acc_pow), accepted if f decreases                (scalar s-1)
%   'els'   exact line search (Rajih, Comon & Harshman 2008):  x_k + a r_k, a = global
%           minimiser of the degree-2N polynomial f(a)                 (scalar a-1)
%   'aa'    Anderson acceleration (Walker & Ni 2011), window opts.win, safeguarded and
%           restarted on failure            (least-squares combination of past G's)
%   'dmd'   DMD fixed-point extrapolation: exact DMD (Tu et al. 2014) fitted to the pairs
%           dX_j = x_{j+1}-x_j,  dG_j = G(x_{j+1})-G(x_j)  of the last opts.win iterations
%           (dG_j ~ J dX_j holds for ANY pair of points, so the window may contain
%           accelerated steps).  With dX = U S V', Atil = U' dG V S^{-1} = W diag(lam) W^{-1}
%           the extrapolation direction is, from (1),
%                 p = U W diag( lam./(1-lam) ) W^{-1} U' r_k ,
%           where modes with |lam| >= 1 are dropped and |lam/(1-lam)| is capped at
%           opts.amp_max.  The new point is G(x_k) + a p with a from an exact line search
%           along p (opts.linesearch = 'els') or a in {1, 1/2, 1/4} ('backtrack').
%           Note that 'els' is the special case of a single mode whose gain lam/(1-lam)
%           is fitted by the line search instead of from the snapshot history.
%
% INPUTS
%   X      dense N-way array (N >= 3 recommended; N = 2 also works)
%   R      number of components
%   opts   struct, all fields optional:
%     .accel       'none' (default) | 'bro' | 'els' | 'aa' | 'dmd'
%     .max_iters   maximum number of iterations (sweeps)              [1000]
%     .tol         stop when |f_k - f_{k-1}| / f_{k-1} < tol          [1e-8]
%     .floor       stop when f_k / (1/2 ||X||^2) < floor (noise-free data reaches
%                  the rounding floor of the objective evaluation)     [1e-12]
%     .damp        relative Tikhonov damping of the normal equations   [0]
%     .init        'randn' (default) | 'rand' | cell {A1,...,AN}
%     .seed        rng seed used for the initialisation                [] (none)
%     .win         window (number of stored (x, G(x)) pairs) for 'aa'/'dmd'  [6]
%     .start       first iteration at which acceleration is attempted [3]
%     .every       attempt acceleration every 'every' iterations      [1]
%     .linesearch  'els' | 'backtrack'   (for 'dmd')                   ['els']
%     .sv_tol      relative singular-value cut-off of the DMD snapshot matrix [1e-6]
%     .amp_max     cap on |lam/(1-lam)| of a DMD mode                  [1e3]
%     .restart_on_reject  clear the 'aa'/'dmd' history after a rejected step [true]
%     .acc_pow, .max_fail   parameters of the 'bro' line search       [2, 4]
%     .store_eigs  store the DMD eigenvalues of every attempt in out.eigs [false]
%     .monitor     true: run the DMD convergence diagnostic every iteration (works with
%                  any accel, including 'none') and return it in out.diag        [false]
%     .monitor_forecast  also evaluate the objective at the DMD-predicted fixed point
%                  (one extra MTTKRP per iteration) -> out.diag.f_inf            [true]
%     .nonneg      logical, scalar or 1 x N: nonnegativity constraint per mode (sub-problems
%                  solved row-wise by the N-way toolbox fastnnls if on the path, else by 10
%                  HALS passes); accelerated candidates are projected onto the constraint
%                  and accepted only if the exact objective decreases                [false]
%     .nnls        'auto' | 'fastnnls' | 'hals'                                       ['auto']
%     Missing values: NaN entries of X are handled by expectation-maximisation (refilled
%                  with the model after every sweep); f_hist is then the misfit over the
%                  observed entries only.
%     .verbose     0 (silent) | 1 (one line per iteration) | 2 (+ DMD eigenvalues)
%
% OUTPUTS
%   A      cell {A1,...,AN}; columns of A2..AN have unit norm, A1 carries the weights
%   out    struct: f_hist, t_hist (cumulative seconds), niters, fval, time, stop,
%          n_accept, n_try, n_mttkrp (cost in MTTKRP equivalents), alpha_hist,
%          eigs (if store_eigs), opts, and with opts.monitor the struct diag with per-
%          iteration series: rho (dominant |lambda| of the ALS map), n_slow (#modes with
%          |lambda| > 0.9), lam (all eigenvalues), rate_emp (||r_k||/||r_{k-1}||, the
%          empirical rate), step_rel (||predicted step||/||x||, the predicted relative
%          distance to the fixed point), f_inf (objective at the predicted fixed point),
%          rem_dec_1mode (predicted remaining decrease of f from the dominant mode),
%          iters_to_tol (predicted iterations until the relative-change rule fires),
%          part (share of the predicted step carried by each component), pair (the two
%          components that carry most of it), pair_congr (their signed congruence
%          product over the modes; -> -1 signals a two-factor degeneracy), weights
%          (column norms of A1)
%
% References
%   Rajih, Comon, Harshman (2008) SIAM J. Matrix Anal. Appl. 30, 1128-1147 (ELS)
%   Walker & Ni (2011) SIAM J. Numer. Anal. 49, 1715-1735 (Anderson acceleration)
%   Tu, Rowley, Luchtenburg, Brunton, Kutz (2014) J. Comput. Dyn. 1, 391-421 (exact DMD)
%   Andersson (2016) Computers & Fluids 133, 32-42 (DMD acceleration of iterative solvers)
%   Sidi (2017) Vector Extrapolation Methods with Applications, SIAM (RRE/MPE)

% ---------------------------------------------------------------- options
if nargin < 3 || isempty(opts), opts = struct; end
def = struct('accel','none', 'max_iters',1000, 'tol',1e-8, 'floor',1e-12, 'damp',0, ...
             'init','randn', 'seed',[], 'win',6, 'start',3, 'every',1, 'linesearch','els', ...
             'sv_tol',1e-6, 'amp_max',1e3, 'restart_on_reject',true, 'acc_pow',2, ...
             'max_fail',4, 'store_eigs',false, 'monitor',false, 'monitor_forecast',true, 'verbose',0, ...
             'nonneg',false, 'nnls','auto');
fn = fieldnames(opts);
for i = 1:numel(fn)
    if ~isfield(def, fn{i}), warning('cp_als_dmd:opt', 'Unknown option "%s" ignored', fn{i}); end
end
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}) || isempty(opts.(fn{i})), opts.(fn{i}) = def.(fn{i}); end
end
accel = lower(opts.accel);
assert(any(strcmp(accel, {'none','bro','els','aa','dmd'})), 'unknown opts.accel "%s"', accel);
opts.win = max(2, round(opts.win));

% ---------------------------------------------------------------- setup
N  = ndims(X);
Is = size(X);
miss = isnan(X); has_miss = any(miss(:));
if has_miss, X(miss) = mean(X(~miss)); end          % initial fill; refined by EM below
Xn = unfoldings(X, N, Is);
Xnorm2 = sum(X(:).^2);
p = sum(Is) * R;
nonneg = logical(opts.nonneg); if isscalar(nonneg), nonneg = repmat(nonneg, 1, N); end
constrained = any(nonneg);
nnls = lower(opts.nnls);
if strcmp(nnls, 'auto'), if exist('fastnnls', 'file'), nnls = 'fastnnls'; else, nnls = 'hals'; end, end

if ~isempty(opts.seed), rng(opts.seed); end
if iscell(opts.init)
    assert(numel(opts.init) == N, 'opts.init must contain N factor matrices');
    A = opts.init;
elseif strcmpi(opts.init, 'randn')
    A = arrayfun(@(n) randn(Is(n), R), 1:N, 'uni', 0);
elseif strcmpi(opts.init, 'rand')
    A = arrayfun(@(n) rand(Is(n), R), 1:N, 'uni', 0);
else
    error('opts.init must be ''randn'', ''rand'' or a cell array of factors');
end
[A, AtA] = normalize_factors(A);

% history of (x_j, G(x_j)) pairs for 'aa' and 'dmd'
Hx = zeros(p, opts.win); Hg = zeros(p, opts.win); hcount = 0;
% 'bro' state
acc_pow = opts.acc_pow; n_fail = 0;

f_hist = zeros(opts.max_iters, 1); t_hist = zeros(opts.max_iters, 1);
alpha_hist = nan(opts.max_iters, 1); eigs_hist = cell(opts.max_iters, 1);
if opts.monitor
    K = opts.max_iters;
    dg = struct('rho', nan(K,1), 'n_slow', nan(K,1), 'rate_emp', nan(K,1), 'step_rel', nan(K,1), ...
                'f_inf', nan(K,1), 'rem_dec_1mode', nan(K,1), 'iters_to_tol', nan(K,1), ...
                'part', nan(K,R), 'pair', nan(K,2), 'pair_congr', nan(K,1), 'weights', nan(K,R));
    dg.lam = cell(K, 1); r_prev_norm = NaN;
end
n_accept = 0; n_try = 0; n_mttkrp = 0; stop = 'max_iters';
f_prev = Inf;
t0 = tic;

% ---------------------------------------------------------------- main loop
for k = 1:opts.max_iters
    % ---- one ALS sweep: x_k -> g_k = G(x_k), objective f_g comes for free
    [Ag, AtAg, f_g] = als_sweep(Xn, A, AtA, opts.damp, Xnorm2, nonneg, nnls);
    n_mttkrp = n_mttkrp + N;
    if has_miss                                       % EM: refill the missing entries with the model
        Mfull = reshape(Ag{1} * kr_except(Ag, 1)', Is);
        X(miss) = Mfull(miss);
        Xn = unfoldings(X, N, Is); Xnorm2 = sum(X(:).^2);
        f_g = 0.5 * sum((X(:) - Mfull(:)).^2);        % = misfit over the observed entries
        n_mttkrp = n_mttkrp + 1;
    end

    A_new = Ag; AtA_new = AtAg; f_new = f_g; accepted = false; tried = false; msg = '';

    try_accel = ~strcmp(accel, 'none') && k >= opts.start && mod(k - opts.start, opts.every) == 0;

    % ---- push the pair (x_k, G(x_k)) into the history
    if any(strcmp(accel, {'aa','dmd'})) || opts.monitor
        if hcount < opts.win
            hcount = hcount + 1;
        else
            Hx(:, 1:end-1) = Hx(:, 2:end); Hg(:, 1:end-1) = Hg(:, 2:end);
        end
        Hx(:, hcount) = pack_factors(A); Hg(:, hcount) = pack_factors(Ag);
    end

    if try_accel
        switch accel
            % ---------------------------------------------------------- Bro
            case 'bro'
                tried = true;
                s = k^(1/acc_pow);
                A_try = project_nonneg(cellfun(@(a, g) a + s*(g - a), A, Ag, 'uni', 0), nonneg);
                f_try = objective(Xn, A_try, Xnorm2); n_mttkrp = n_mttkrp + 1;
                if f_try < f_g
                    accepted = true; A_new = A_try; f_new = f_try; alpha_hist(k) = s;
                else
                    n_fail = n_fail + 1;
                    if n_fail >= opts.max_fail, acc_pow = acc_pow + 2; n_fail = 0; end
                end
                msg = sprintf('bro s=%.2f', s);
            % ---------------------------------------------------------- ELS
            case 'els'
                tried = true;
                D = cellfun(@(a, g) g - a, A, Ag, 'uni', 0);
                [alpha, q_best, q_base, nm] = exact_line_search(Xn, A, D, 1);
                n_mttkrp = n_mttkrp + nm;
                if q_best < q_base && alpha ~= 1
                    A_try = cellfun(@(a, d) a + alpha*d, A, D, 'uni', 0);
                    if constrained
                        A_try = project_nonneg(A_try, nonneg);
                        f_try = objective(Xn, A_try, Xnorm2); n_mttkrp = n_mttkrp + 1;
                    else
                        f_try = f_g + (q_best - q_base);
                    end
                    if f_try < f_g, accepted = true; A_new = A_try; f_new = f_try; alpha_hist(k) = alpha; end
                end
                msg = sprintf('els a=%.3f', alpha);
            % ---------------------------------------------------------- Anderson
            case 'aa'
                if hcount >= 2
                    tried = true;
                    Rr = Hg(:, 1:hcount) - Hx(:, 1:hcount);          % residuals
                    dR = Rr(:, 2:end) - Rr(:, 1:end-1);
                    dG = Hg(:, 2:hcount) - Hg(:, 1:hcount-1);
                    gamma = dR \ Rr(:, end);
                    x_try = Hg(:, hcount) - dG * gamma;
                    A_try = project_nonneg(unpack_factors(x_try, Is, R), nonneg);
                    f_try = objective(Xn, A_try, Xnorm2); n_mttkrp = n_mttkrp + 1;
                    if f_try < f_g
                        accepted = true; A_new = A_try; f_new = f_try;
                    end
                    msg = sprintf('aa m=%d', hcount);
                end
            % ---------------------------------------------------------- DMD
            case 'dmd'
                if hcount >= 3
                    tried = true;
                    dX = Hx(:, 2:hcount) - Hx(:, 1:hcount-1);
                    dG = Hg(:, 2:hcount) - Hg(:, 1:hcount-1);
                    rk = Hg(:, hcount) - Hx(:, hcount);
                    [pdir, lam, ok] = dmd_extrapolate(dX, dG, rk, opts.sv_tol, opts.amp_max);
                    if opts.store_eigs, eigs_hist{k} = lam; end
                    if ok
                        Pd = unpack_factors(pdir, Is, R);
                        if strcmpi(opts.linesearch, 'els')
                            [alpha, q_best, q_base, nm] = exact_line_search(Xn, Ag, Pd, 0);
                            n_mttkrp = n_mttkrp + nm;
                            if q_best < q_base && alpha ~= 0
                                A_try = cellfun(@(g, d) g + alpha*d, Ag, Pd, 'uni', 0);
                                if constrained
                                    A_try = project_nonneg(A_try, nonneg);
                                    f_try = objective(Xn, A_try, Xnorm2); n_mttkrp = n_mttkrp + 1;
                                else
                                    f_try = f_g + (q_best - q_base);
                                end
                                if f_try < f_g, accepted = true; A_new = A_try; f_new = f_try; alpha_hist(k) = alpha; end
                            end
                        else
                            alpha = 1;
                            for trial = 1:3
                                A_try = project_nonneg(cellfun(@(g, d) g + alpha*d, Ag, Pd, 'uni', 0), nonneg);
                                f_try = objective(Xn, A_try, Xnorm2); n_mttkrp = n_mttkrp + 1;
                                if f_try < f_g
                                    accepted = true; A_new = A_try; f_new = f_try; alpha_hist(k) = alpha;
                                    break
                                end
                                alpha = alpha / 2;
                            end
                        end
                        msg = sprintf('dmd r=%d |lam|max=%.4f a=%.3f', numel(lam), max(abs(lam)), alpha);
                    else
                        msg = 'dmd (skipped: degenerate window)';
                    end
                end
        end
        if tried
            n_try = n_try + 1;
            if accepted
                n_accept = n_accept + 1;
                [A_new, AtA_new] = normalize_factors(A_new);
            elseif opts.restart_on_reject && any(strcmp(accel, {'aa','dmd'}))
                % keep only the current pair
                Hx(:, 1) = Hx(:, hcount); Hg(:, 1) = Hg(:, hcount); hcount = 1;
            end
        end
    end

    % ---- DMD convergence diagnostic (independent of the accelerator)
    if opts.monitor
        rk = Hg(:, hcount) - Hx(:, hcount);
        dg.rate_emp(k) = norm(rk) / r_prev_norm; r_prev_norm = norm(rk);
        dg.weights(k, :) = sqrt(sum(Ag{1}.^2, 1));
        if hcount >= 3
            dX = Hx(:, 2:hcount) - Hx(:, 1:hcount-1);
            dG = Hg(:, 2:hcount) - Hg(:, 1:hcount-1);
            [pm, lamm, okm] = dmd_extrapolate(dX, dG, rk, opts.sv_tol, opts.amp_max);
            if okm
                dg.lam{k} = lamm;
                rho = max(abs(lamm)); dg.rho(k) = rho; dg.n_slow(k) = sum(abs(lamm) > 0.9);
                dg.step_rel(k) = norm(pm) / norm(Hg(:, hcount));
                Pm = unpack_factors(pm, Is, R);
                e = zeros(1, R); for n = 1:N, e = e + sum(Pm{n}.^2, 1); end
                dg.part(k, :) = e / max(sum(e), eps);
                [~, order] = sort(e, 'descend'); r1 = order(1); r2 = order(min(2, R));
                dg.pair(k, :) = [r1 r2];
                cg = 1;
                for n = 1:N
                    cg = cg * (Ag{n}(:, r1)' * Ag{n}(:, r2)) / (norm(Ag{n}(:, r1)) * norm(Ag{n}(:, r2)) + eps);
                end
                dg.pair_congr(k) = cg;
                if k > 1 && rho < 1
                    dfk = max(abs(f_prev - f_g), realmin);
                    dg.rem_dec_1mode(k) = dfk * rho^2 / (1 - rho^2);
                    dg.iters_to_tol(k) = max(0, log(opts.tol * f_g / dfk) / (2 * log(rho)));
                end
                if opts.monitor_forecast
                    dg.f_inf(k) = objective(Xn, cellfun(@(g, d) g + d, Ag, Pm, 'uni', 0), Xnorm2);
                    n_mttkrp = n_mttkrp + 1;
                end
            end
        end
    end

    A = A_new; AtA = AtA_new;
    f_hist(k) = f_new; t_hist(k) = toc(t0);

    if opts.verbose
        if tried
            fprintf('it %5d  f=%.10e  %-40s %s\n', k, f_new, msg, tern(accepted, 'ACCEPT', 'reject'));
        else
            fprintf('it %5d  f=%.10e\n', k, f_new);
        end
        if opts.verbose >= 2 && strcmp(accel, 'dmd') && tried && ok
            fprintf('          DMD eigenvalues: %s\n', mat2str(lam(:).', 5));
        end
    end

    % ---- stopping tests
    if f_new / (0.5*Xnorm2) < opts.floor, stop = 'floor'; break; end
    if k > 1 && abs(f_prev - f_new) / f_prev < opts.tol, stop = 'tol'; break; end
    f_prev = f_new;
end

niters = min(k, opts.max_iters);
out = struct('f_hist', f_hist(1:niters), 't_hist', t_hist(1:niters), 'niters', niters, ...
             'fval', f_hist(niters), 'time', t_hist(niters), 'stop', stop, ...
             'n_accept', n_accept, 'n_try', n_try, 'n_mttkrp', n_mttkrp, ...
             'alpha_hist', alpha_hist(1:niters), 'opts', opts);
if opts.store_eigs, out.eigs = eigs_hist(1:niters); end
if opts.monitor
    fn = fieldnames(dg);
    for i = 1:numel(fn), dg.(fn{i}) = dg.(fn{i})(1:niters, :); end
    out.diag = dg;
end
end

% ====================================================================== helpers
function [A, AtA, f] = als_sweep(Xn, A, AtA, damp, Xnorm2, nonneg, nnls)
% One ALS sweep over all modes; returns the normalised factors and the exact objective
N = numel(A); R = size(A{1}, 2);
for n = 1:N
    G = ones(R);
    for m = [1:n-1, n+1:N], G = G .* AtA{m}; end
    M = Xn{n} * kr_except(A, n);                          % MTTKRP
    if damp > 0, G = G + damp * (trace(G)/R) * eye(R); end
    if nonneg(n)
        A{n} = nnls_rows(M, G, nnls, A{n});
    else
        A{n} = M / G;
    end
    AtA{n} = A{n}' * A{n};
    if n == N, xXm = sum(sum(A{n} .* M)); end           % <X, M(A)> for free
end
H = ones(R); for n = 1:N, H = H .* AtA{n}; end
f = 0.5 * (Xnorm2 - 2*xXm + sum(H(:)));
[A, AtA] = normalize_factors(A);
end

function Xn = unfoldings(X, N, Is)
Xn = cell(1, N);
for n = 1:N, Xn{n} = reshape(permute(X, [n, 1:n-1, n+1:N]), Is(n), []); end
end

function A = project_nonneg(A, nonneg)
for n = find(nonneg), A{n} = max(A{n}, 0); end
end

function A = nnls_rows(M, G, method, A_old)
% rows of A solve min ||x_i - Z a_i||^2 s.t. a_i >= 0 given G = Z'Z and M(i,:)' = Z'x_i
R = size(G, 1);
switch method
    case 'fastnnls'
        A = zeros(size(M)); tol = 10 * eps * norm(G, 1) * R;
        for i = 1:size(M, 1), A(i, :) = fastnnls(G, M(i, :)', tol)'; end
    otherwise                                             % HALS (Cichocki & Phan 2009)
        A = max(A_old, 0); d = max(diag(G)', eps);
        for pass = 1:10
            for r = 1:R
                A(:, r) = max(0, (M(:, r) - A * G(:, r) + A(:, r) * G(r, r)) / d(r));
            end
        end
end
end

function f = objective(Xn, A, Xnorm2)
% f = 1/2 ||X - M(A)||^2 with one MTTKRP
N = numel(A); R = size(A{1}, 2);
xXm = sum(sum(A{1} .* (Xn{1} * kr_except(A, 1))));
H = ones(R); for n = 1:N, H = H .* (A{n}' * A{n}); end
f = 0.5 * (Xnorm2 - 2*xXm + sum(H(:)));
end

function [A, AtA] = normalize_factors(A)
% gauge fixing: unit-norm columns in modes 2..N, all weights in mode 1
N = numel(A);
for n = 2:N
    nrm = sqrt(sum(A{n}.^2, 1)); nrm(nrm == 0) = 1;
    A{n} = A{n} ./ nrm;
    A{1} = A{1} .* nrm;
end
AtA = cellfun(@(M) M' * M, A, 'uni', 0);
end

function K = kr_except(A, n)
% Khatri-Rao product of all factors except mode n, ordered to match the mode-n unfolding
N = numel(A); R = size(A{1}, 2);
others = [1:n-1, n+1:N];
K = A{others(1)};
for o = others(2:end)
    K = reshape(reshape(K, [], 1, R) .* reshape(A{o}, 1, [], R), [], R);
end
end

function x = pack_factors(A)
x = cell2mat(cellfun(@(M) M(:), A(:), 'uni', 0));
end

function A = unpack_factors(x, Is, R)
N = numel(Is); A = cell(1, N); s = 0;
for n = 1:N
    A{n} = reshape(x(s + (1:Is(n)*R)), Is(n), R); s = s + Is(n)*R;
end
end

function [alpha, q_best, q_base, n_mttkrp] = exact_line_search(Xn, B, D, alpha_base)
% Global minimiser over the real line of f(a) = 1/2||X - M(B + a D)||^2  (Rajih et al. 2008)
% f(a) = 1/2||X||^2 + q(a),  q(a) = -sum_j c_j a^j + 1/2 sum_l g_l a^l  (degree 2N).
% Returns q at the best a and at alpha_base (same rounding, so differences are consistent).
N = numel(B); R = size(B{1}, 2);
c = zeros(1, N+1);                       % c(j+1) = sum_{|S|=j} <X, M_S>
for mask = 0:2^(N-1)-1                   % subsets of modes 2..N, one MTTKRP each
    U = B; j = 0;
    for n = 2:N
        if bitget(mask, n-1), U{n} = D{n}; j = j + 1; end
    end
    Y = Xn{1} * kr_except(U, 1);
    c(j+1) = c(j+1) + sum(sum(B{1} .* Y));
    c(j+2) = c(j+2) + sum(sum(D{1} .* Y));
end
n_mttkrp = 2^(N-1);
BB = cell(1,N); BD = cell(1,N); DD = cell(1,N);
for n = 1:N, BB{n} = B{n}'*B{n}; BD{n} = B{n}'*D{n}; DD{n} = D{n}'*D{n}; end
g = zeros(1, 2*N+1);                     % g(l+1) = sum_{|S|+|T|=l} <M_S, M_T>
for s = 0:2^N-1
    for t = 0:2^N-1
        H = ones(R); l = 0;
        for n = 1:N
            sb = bitget(s, n); tb = bitget(t, n); l = l + sb + tb;
            if ~sb && ~tb,     H = H .* BB{n};
            elseif ~sb && tb,  H = H .* BD{n};
            elseif sb && ~tb,  H = H .* BD{n}';
            else,              H = H .* DD{n};
            end
        end
        g(l+1) = g(l+1) + sum(H(:));
    end
end
q = 0.5 * g; q(1:N+1) = q(1:N+1) - c;    % increasing powers
dq = q(2:end) .* (1:2*N);
rts = roots(fliplr(dq));
rts = rts(abs(imag(rts)) <= 1e-8 * max(1, abs(real(rts))));
cand = [real(rts(:)); alpha_base];
qv = polyval(fliplr(q), cand);
[q_best, i] = min(qv); alpha = cand(i);
q_base = polyval(fliplr(q), alpha_base);
end


function out = tern(c, a, b)
if c, out = a; else, out = b; end
end
