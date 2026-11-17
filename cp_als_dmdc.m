function [A, out] = cp_als_dmdc(X, R, opts)
%CP_ALS_DMDC  Damped ALS whose damping is chosen by a DMD-with-control (DMDc) controller.
%
% Research prototype for steering ALS out of a degenerate swamp (two-factor degeneracy).
% ALS with Tikhonov damping mu on the mode-1 (weight) sub-problem is block coordinate
% descent on 1/2||X - M(A)||^2 + mu/2 ||A1||_F^2 with unit-norm columns in modes 2..N, so mu
% is an actuator that changes the ALS map and its fixed point.  Around the current
% trajectory the iterates are modelled as  x_{k+1} = A x_k + b mu_k + c  (DMDc, Proctor,
% Brunton & Kutz 2016), identified from the last opts.win triples (x_j, mu_j, G(x_j)) via
% the consecutive differences  dG_j = A dX_j + b dmu_j.  With A = V diag(lam) V^{-1} and left
% eigenvector w1 of the slow mode (lam1 ~ 1 in a degenerate swamp), the amplitude of that
% mode is a_k = w1'(x_k - x_ref) and the proportional law
%
%      mu_k = clip( mu0 + alpha a_k ,  mu_min, mu_max ),   alpha = (lam_target - lam1) / (w1' b)
%
% places the closed-loop eigenvalue of the slow mode at lam_target (rank-one pole placement
% along a left eigenvector leaves the other eigenvalues unchanged): the damping grows with
% the displacement along the degenerate direction until the growth stops.
%
% opts.control  'none' | 'fixed' | 'decay' | 'trigger' | 'dmdc'
%   'none'     mu = 0 (plain ALS)
%   'fixed'    mu = mu0
%   'decay'    mu = mu0 * decay^(k-1)
%   'trigger'  mu = mu_high once the DMD monitor has seen rho > rho_crit with a negative
%              slow-mode pair congruence (< congr_crit) for 3 iterations, else mu_low
%   'ratefb'   switched on as 'trigger'; then every win iterations the open-loop DMD rate
%              rho at the current constant mu is read and mu is multiplied by 3 if
%              rho > lam_target + 0.02, divided by 1.5 if rho < lam_target - 0.1
%              (parametric pole placement by feedback on the measured rate, no b needed)
%   'fixedpath' classic continuation: mu = mu0 until the stage converges (stage_tol), then
%              mu <- mu * cont_mu_factor, repeated until a stop rule fires (see below)
%   'dmdc'     switched on as 'trigger'; then n_id iterations with a pseudo-random binary
%              dither of mu around mu0 (persistent excitation) to identify (A, b, c) by
%              DMDc; the slow mode is oriented so that its open-loop drift c1 > 0 and the
%              model is accepted only if b1 < 0 (damping opposes the drift); then the
%              proportional law above with the FROZEN model; re-identified only if mu
%              saturates at mu_max for 5 iterations (closed-loop data are not used for
%              identification).  Falls back to 'ratefb' when identification fails twice.
% Continuation for 'dmdc' (opts.continuation = true): after every converged stage the base
%   damping is lowered, mu_base <- mu * cont_mu_factor, and optionally the target rate raised,
%   1 - lam_t <- (1 - lam_t) * cont_lam_factor (1 keeps it), the model is re-identified and the
%   feedback re-balances the damping at the new position on the regularisation path.  A stage ends when the
%   relative changes of f and x fall below stage_tol / stage_xtol.  Stop rules (both for
%   'dmdc' continuation and 'fixedpath'): pair congruence below congr_bound, relative fit
%   gain of the stage below min_gain, lam_t >= lam_t_max, or max_stages reached; the run then
%   finishes the last stage to the full tolerance.  out.stages records the path.
% other opts (defaults): mu0 1e-3, decay 0.995, mu_high 1e-2, mu_low 0, rho_crit 0.99,
%   congr_crit -0.3, lam_target 0.9, mu_min 0, mu_max 1, dither 0.75, n_id 12, rank_cap 2
%   (number of DMD modes kept in the reduced model; 2-3 is robust, 4 is not), win 6,
%   sv_tol 1e-6, max_iters 1500, tol 1e-8, xtol 1e-7, init 'randn', seed [], continuation false,
%   cont_lam_factor 0.5, cont_mu_factor 0.5, stage_tol 1e-6, stage_xtol 1e-5, congr_bound -0.9,
%   min_gain 0.02, lam_t_max 0.99, max_stages 8
% out: f_hist (unregularised objective 1/2||X-M||^2), mu_hist, rho_hist (dominant |lambda|
%   of the open-loop DMD model), congr_hist (signed congruence of the slow-mode pair),
%   wmax_hist (largest weight), amp_hist, ident (struct array of the identified models:
%   k, lam1, b1, c1, alpha), t_on, niters, stop, phase ('off' | 'ident' | 'control' | 'ratefb')
if nargin < 3, opts = struct; end
def = struct('control','none', 'mu0',1e-3, 'decay',0.995, 'mu_high',1e-2, 'mu_low',0, ...
             'rho_crit',0.99, 'congr_crit',-0.3, 'lam_target',0.9, 'mu_min',0, 'mu_max',1, ...
             'dither',0.75, 'n_id',12, 'rank_cap',2, 'win',6, 'sv_tol',1e-6, 'max_iters',1500, 'tol',1e-8, 'xtol',1e-7, ...
             'init','randn', 'seed',[], 'continuation',false, 'cont_lam_factor',0.5, 'cont_mu_factor',0.5, ...
             'stage_tol',1e-6, 'stage_xtol',1e-5, 'congr_bound',-0.9, 'min_gain',0.02, 'lam_t_max',0.99, 'max_stages',8);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}) || isempty(opts.(fn{i})), opts.(fn{i}) = def.(fn{i}); end
end
control = lower(opts.control);
N = ndims(X); Is = size(X); p = sum(Is) * R;
Xn = cell(1, N);
for n = 1:N, Xn{n} = reshape(permute(X, [n, 1:n-1, n+1:N]), Is(n), []); end
Xnorm2 = sum(X(:).^2);
if ~isempty(opts.seed), rng(opts.seed); end
if iscell(opts.init), A = opts.init; else, A = arrayfun(@(n) randn(Is(n), R), 1:N, 'uni', 0); end
[A, AtA] = normalize_factors(A);
x = pack_factors(A);

K = opts.max_iters; m = opts.win;
Hx = zeros(p, m); Hg = zeros(p, m); Hu = zeros(1, m); hcount = 0;
f_hist = nan(K,1); mu_hist = nan(K,1); rho_hist = nan(K,1); congr_hist = nan(K,1); wmax_hist = nan(K,1);
amp_hist = nan(K,1);
on = false; t_on = NaN; n_trig = 0; phase = 'off'; mdl = []; ident = struct('k', {}, 'ok', {}, 'reason', {}, 'lam1', {}, 'b1', {}, 'c1', {}, 'alpha', {}, 'lam', {});
Xid = []; Gid = []; Uid = []; prbs = []; n_sat = 0; n_fail = 0; mu_ctrl = opts.mu0; last_mu_change = 0;
rho = NaN; congr = NaN; f_prev = Inf; stop = 'max_iters';
lam_t = opts.lam_target; stage = 1; stage_k0 = 1; f_stage_prev = Inf; finishing = false;
stages = struct('stage', {}, 'mu', {}, 'lam_t', {}, 'k_end', {}, 'iters', {}, 'f', {}, 'wmax', {}, 'congr', {}, 'stop_reason', {});
if strcmp(control, 'fixedpath'), on = true; t_on = 1; mu_ctrl = opts.mu0; end
for k = 1:K
    % ---------------------------------------------------------------- choose mu_k
    switch control
        case 'none',  mu = 0;
        case 'fixed', mu = opts.mu0;
        case 'decay', mu = opts.mu0 * opts.decay^(k-1);
        case 'fixedpath', mu = mu_ctrl;
        case {'trigger', 'dmdc', 'ratefb'}
            if ~on
                mu = opts.mu_low;
            elseif strcmp(control, 'trigger')
                mu = opts.mu_high;
            elseif strcmp(phase, 'ratefb')
                if k - last_mu_change >= m && isfinite(rho)
                    if rho > opts.lam_target + 0.02,     mu_ctrl = min(opts.mu_max, 3 * mu_ctrl);   last_mu_change = k;
                    elseif rho < opts.lam_target - 0.1,  mu_ctrl = max(opts.mu_min, mu_ctrl / 1.5); last_mu_change = k;
                    end
                end
                mu = mu_ctrl;
            elseif strcmp(phase, 'ident')
                j = k - t_on;                                        % 1..n_id
                mu = mu_ctrl * (1 + opts.dither * prbs(j));
            else                                                     % 'control'
                a = real(mdl.w1' * (mdl.Uh' * x)) - mdl.a_on;
                mu = min(opts.mu_max, max(opts.mu_min, mu_ctrl + mdl.alpha * a));
                amp_hist(k) = a;
                if mu >= opts.mu_max, n_sat = n_sat + 1; else, n_sat = 0; end
            end
        otherwise, error('unknown control "%s"', control);
    end
    % ---------------------------------------------------------------- damped sweep
    [Ag, AtAg, f] = als_sweep(Xn, A, AtA, mu, Xnorm2);
    g = pack_factors(Ag);
    % ---------------------------------------------------------------- window
    if hcount < m, hcount = hcount + 1; else, Hx(:,1:end-1) = Hx(:,2:end); Hg(:,1:end-1) = Hg(:,2:end); Hu(1:end-1) = Hu(2:end); end
    Hx(:, hcount) = x; Hg(:, hcount) = g; Hu(hcount) = mu;
    % ---------------------------------------------------------------- identification
    if hcount >= 3
        dX = Hx(:, 2:hcount) - Hx(:, 1:hcount-1);
        dG = Hg(:, 2:hcount) - Hg(:, 1:hcount-1);
        dU = Hu(2:hcount) - Hu(1:hcount-1);
        rk = g - x;
        [pm, lam, ok] = dmd_extrapolate(dX, dG, rk, opts.sv_tol, 1e3);     % open-loop monitor
        if ok
            rho = max(abs(lam));
            Pm = unpack_factors(pm, Is, R);
            e = zeros(1, R); for n = 1:N, e = e + sum(Pm{n}.^2, 1); end
            [~, o] = sort(e, 'descend'); r1 = o(1); r2 = o(min(2, R));
            congr = 1;
            for n = 1:N
                congr = congr * (Ag{n}(:, r1)' * Ag{n}(:, r2)) / (norm(Ag{n}(:, r1)) * norm(Ag{n}(:, r2)) + eps);
            end
        end
    end
    % ---------------------------------------------------------------- DMDc identification phase
    if strcmp(control, 'dmdc') && strcmp(phase, 'ident')
        Xid = [Xid x]; Gid = [Gid g]; Uid = [Uid mu]; %#ok<AGROW>
        if size(Xid, 2) >= opts.n_id
            mdl_new = dmdc_identify(Xid, Gid, Uid, opts.sv_tol, lam_t, p, opts.rank_cap);
            Xid = []; Gid = []; Uid = [];
            ident(end+1) = struct('k', k, 'ok', mdl_new.ok, 'reason', mdl_new.reason, 'lam1', mdl_new.lam1, ...
                                  'b1', mdl_new.b1, 'c1', mdl_new.c1, 'alpha', mdl_new.alpha, 'lam', mdl_new.lam); %#ok<AGROW>
            if mdl_new.ok
                mdl = mdl_new; mdl.a_on = real(mdl.w1' * (mdl.Uh' * g)); phase = 'control'; n_sat = 0;
                if stage == 1, f_stage_prev = Inf; end
            else
                n_fail = n_fail + 1;
                if n_fail >= 2, phase = 'ratefb'; last_mu_change = k; else, t_on = k; prbs = sign(randn(opts.n_id, 1)); end
            end
        end
    end
    if strcmp(phase, 'control') && n_sat >= 5          % saturated: identify again at the current level
        mu_ctrl = min(opts.mu_max / 4, mu); phase = 'ident'; t_on = k; prbs = sign(randn(opts.n_id, 1)); n_sat = 0;
    end
    % ---------------------------------------------------------------- trigger
    if any(strcmp(control, {'trigger', 'dmdc', 'ratefb'})) && ~on
        if rho > opts.rho_crit && congr < opts.congr_crit, n_trig = n_trig + 1; else, n_trig = 0; end
        if n_trig >= 3
            on = true; t_on = k; last_mu_change = k; mu_ctrl = opts.mu0;
            if strcmp(control, 'dmdc'), phase = 'ident'; rs = rng; rng(5); prbs = sign(randn(opts.n_id, 1)); rng(rs); end
            if strcmp(control, 'ratefb'), phase = 'ratefb'; end
        end
    end
    % ---------------------------------------------------------------- record / update / stop
    f_hist(k) = f; mu_hist(k) = mu; rho_hist(k) = rho; congr_hist(k) = congr; wmax_hist(k) = max(sqrt(sum(Ag{1}.^2, 1)));
    dxrel = norm(g - x) / norm(x);
    A = Ag; AtA = AtAg; x = g;
    settled = ~any(strcmp(control, {'trigger','dmdc','ratefb','fixedpath'})) || (~on && k > 50) || ...
              (on && strcmp(control, 'trigger') && k > t_on + 5) || ...
              (on && strcmp(phase, 'ratefb') && k - last_mu_change > m + 2) || ...
              (on && strcmp(phase, 'control') && k > t_on + opts.n_id + 5) || ...
              (strcmp(control, 'fixedpath') && k > stage_k0 + 2);
    df_rel = abs(f_prev - f) / f_prev;
    staged = (strcmp(control, 'dmdc') && opts.continuation && strcmp(phase, 'control')) || strcmp(control, 'fixedpath');
    if k > 1 && settled && staged && ~finishing && df_rel < opts.stage_tol && dxrel < opts.stage_xtol
        % ---- a continuation stage has converged: record it and decide
        gain = (f_stage_prev - f) / f_stage_prev;
        reason = '';
        if congr < opts.congr_bound,          reason = 'congruence bound';
        elseif stage > 1 && gain < opts.min_gain, reason = 'diminishing gain';
        elseif stage >= opts.max_stages,      reason = 'max stages';
        elseif strcmp(control, 'dmdc') && 1 - (1 - lam_t) * opts.cont_lam_factor > opts.lam_t_max, reason = 'lam_t_max';
        elseif strcmp(control, 'fixedpath') && mu_ctrl * opts.cont_mu_factor < 1e-6, reason = 'mu floor';
        end
        stages(end+1) = struct('stage', stage, 'mu', mu, 'lam_t', lam_t, 'k_end', k, 'iters', k - stage_k0 + 1, ...
                               'f', f, 'wmax', wmax_hist(k), 'congr', congr, 'stop_reason', reason); %#ok<AGROW>
        if ~isempty(reason)
            finishing = true;                                % finish this stage to the full tolerance
        else
            stage = stage + 1; stage_k0 = k + 1; f_stage_prev = f;
            if strcmp(control, 'fixedpath')
                mu_ctrl = mu_ctrl * opts.cont_mu_factor;
            else
                lam_t = 1 - (1 - lam_t) * opts.cont_lam_factor;
                mu_ctrl = max(mu * opts.cont_mu_factor, 1e-5);   % base damping lowered; the feedback re-balances
                phase = 'ident'; t_on = k; rs = rng; rng(5 + stage); prbs = sign(randn(opts.n_id, 1)); rng(rs); n_fail = 0;
            end
        end
    elseif k > 1 && settled && ~staged && df_rel < opts.tol && dxrel < opts.xtol
        stop = 'tol'; break;
    elseif k > 1 && settled && staged && finishing && df_rel < opts.tol && dxrel < opts.xtol
        stop = 'tol'; break;
    end
    f_prev = f;
end
niters = min(k, K);
out = struct('f_hist', f_hist(1:niters), 'mu_hist', mu_hist(1:niters), 'rho_hist', rho_hist(1:niters), ...
             'congr_hist', congr_hist(1:niters), 'wmax_hist', wmax_hist(1:niters), 'amp_hist', amp_hist(1:niters), ...
             'ident', ident, 't_on', t_on, 'niters', niters, 'stop', stop, 'fval', f_hist(niters), 'phase', phase, ...
             'stages', stages, 'lam_t_final', lam_t, 'opts', opts);
end

function mdl = dmdc_identify(Xid, Gid, Uid, sv_tol, lam_target, p, rank_cap)
% DMDc on the identification triples: dG = A dX + b du; reduced model on the POD basis of dG.
mdl = struct('ok', false, 'reason', '', 'lam1', NaN, 'b1', NaN, 'c1', NaN, 'alpha', NaN, 'lam', []);
dX = Xid(:, 2:end) - Xid(:, 1:end-1); dG = Gid(:, 2:end) - Gid(:, 1:end-1); dU = Uid(2:end) - Uid(1:end-1);
if ~any(dU), mdl.reason = 'no input variation'; return; end
s = norm(dX) / norm(dU);                                   % input row scaled like sigma_1(dX)
Om = [dX; s * dU];
[Uo, So, Vo] = svd(Om, 'econ'); so = diag(So); r = sum(so > sv_tol * so(1));
Uo = Uo(:, 1:r); Vo = Vo(:, 1:r); so = so(1:r);
[Uh, Sh, ~] = svd(dG, 'econ'); sh = diag(Sh); rh = min(rank_cap, sum(sh > sv_tol * sh(1))); Uh = Uh(:, 1:rh);
T = (Uh' * dG * Vo) ./ so';
At = T * (Uo(1:p, :)' * Uh);                               % reduced A (rh x rh)
bt = T * Uo(p+1, :)' * s;                                  % reduced b (rh x 1)
[V, L] = eig(At); lam = diag(L); mdl.lam = lam;
if ~all(isfinite(lam)) || rcond(V) < 1e-12, mdl.reason = 'eig failed'; return; end
[~, i1] = max(abs(lam)); W = inv(V)'; w1 = W(:, i1); lam1 = lam(i1); mdl.lam1 = real(lam1);
if abs(imag(lam1)) > 1e-3 || real(lam1) > 1.2 || real(lam1) < 0.5, mdl.reason = sprintf('lam1 = %s out of range', num2str(lam1, 4)); return; end
lam1 = real(lam1); w1 = real(w1 / norm(w1));
% affine term of the slow-mode amplitude:  a^g_j = lam1 a_j + b1 u_j + c1
ax = w1' * (Uh' * Xid); ag = w1' * (Uh' * Gid);
b1 = w1' * bt;
c1 = mean(ag - lam1 * ax - b1 * Uid);
if c1 < 0, w1 = -w1; b1 = -b1; c1 = -c1; end           % orient: open-loop drift positive
mdl.b1 = b1; mdl.c1 = c1; mdl.alpha = (lam_target - lam1) / b1;
if b1 >= 0 || abs(b1) < 1e-12, mdl.reason = sprintf('b1 = %+.2e does not oppose the drift c1 = %+.2e', b1, c1); return; end
mdl = struct('ok', true, 'reason', 'ok', 'Uh', Uh, 'w1', w1, 'lam1', lam1, 'b1', b1, 'c1', c1, 'alpha', (lam_target - lam1) / b1, 'a_on', 0, 'lam', lam);
end

% ====================================================================== helpers
function [A, AtA, f] = als_sweep(Xn, A, AtA, mu, Xnorm2)
% mode 1 (weights) damped with mu, the other modes plain; f is the unregularised objective
N = numel(A); R = size(A{1}, 2);
for n = 1:N
    G = ones(R);
    for q = [1:n-1, n+1:N], G = G .* AtA{q}; end
    M = Xn{n} * kr_except(A, n);
    if n == 1 && mu > 0, G = G + mu * eye(R); end
    A{n} = M / G;
    AtA{n} = A{n}' * A{n};
    if n == N, xXm = sum(sum(A{n} .* M)); end
end
H = ones(R); for n = 1:N, H = H .* AtA{n}; end
f = 0.5 * (Xnorm2 - 2*xXm + sum(H(:)));
[A, AtA] = normalize_factors(A);
end

function [A, AtA] = normalize_factors(A)
N = numel(A);
for n = 2:N
    nrm = sqrt(sum(A{n}.^2, 1)); nrm(nrm == 0) = 1;
    A{n} = A{n} ./ nrm; A{1} = A{1} .* nrm;
end
AtA = cellfun(@(M) M' * M, A, 'uni', 0);
end

function Kr = kr_except(A, n)
N = numel(A); R = size(A{1}, 2); others = [1:n-1, n+1:N];
Kr = A{others(1)};
for o = others(2:end), Kr = reshape(reshape(Kr, [], 1, R) .* reshape(A{o}, 1, [], R), [], R); end
end

function x = pack_factors(A)
x = cell2mat(cellfun(@(M) M(:), A(:), 'uni', 0));
end

function A = unpack_factors(x, Is, R)
N = numel(Is); A = cell(1, N); s = 0;
for n = 1:N, A{n} = reshape(x(s + (1:Is(n)*R)), Is(n), R); s = s + Is(n)*R; end
end
