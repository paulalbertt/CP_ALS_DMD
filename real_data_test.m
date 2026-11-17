function results = real_data_test(name, cfg)
%REAL_DATA_TEST  Accelerators and the DMD diagnostic on a real EEM dataset.
%
%   results = real_data_test('aminoacids' | 'sugar' | 'dorrit' | 'dreem_tutorial')
%
% For each number of components in cfg.R and each constraint setting, cfg.nstart random
% starts are shared by the methods ALS, ALS + DMD and ALS + Anderson (window 6).  The DMD
% monitor runs in every fit.  Saved to results/real_<name>.mat.
P = local_paths();                                     % copy local_paths_example.m to local_paths.m
addpath(P.nway, '-end');                               % fastnnls
if nargin < 2, cfg = struct; end
def = struct('R', [], 'nonneg', [], 'nstart', 6, 'max_iters', 3000, 'tol', 1e-8, ...
             'methods', {{'none', 'dmd', 'aa'}}, 'checkpoints', [20 50 100 200 500 1000 2000], 'seed0', 100, 'suffix', '');
switch lower(name)
    case 'aminoacids',     def.R = [3 4]; def.nonneg = {true, false}; def.nstart = 10;
    case 'sugar',          def.R = [4 5]; def.nonneg = {true};        def.nstart = 6;
    case 'dorrit',         def.R = [4 5]; def.nonneg = {true, false}; def.nstart = 6;
    case 'dreem_tutorial', def.R = [5 6]; def.nonneg = {true};        def.nstart = 5;
end
fn = fieldnames(def);
for i = 1:numel(fn), if ~isfield(cfg, fn{i}) || isempty(cfg.(fn{i})), cfg.(fn{i}) = def.(fn{i}); end, end
maxNumCompThreads(1);
[X, info] = load_eem(name);
obs = ~isnan(X); half_obs = 0.5 * sum(X(obs).^2);
fprintf('%s: %s\n', name, info);
results = struct('dataset', {}, 'R', {}, 'nonneg', {}, 'method', {}, 'start', {}, 'iters', {}, 'time', {}, ...
                 'f_rel', {}, 'stop', {}, 'n_accept', {}, 'n_try', {}, 'n_mttkrp', {}, 'min_congr', {}, 'max_abs_congr', {}, ...
                 'wratio', {}, 'flag_slow', {}, 'flag_degen', {}, 'final_pair_congr', {}, 'cp', {}, 'f_hist', {}, 'rho_hist', {}, 'congr_hist', {});
t_all = tic;
for R = cfg.R
    for ic = 1:numel(cfg.nonneg)
        nn = cfg.nonneg{ic};
        if ~nn && R == min(cfg.R) && ~strcmpi(name, 'aminoacids'), continue; end   % unconstrained only for over-factored (and all amino-acid) fits
        for st = 1:cfg.nstart
            rng(cfg.seed0 + 10*R + st);
            A0 = arrayfun(@(n) rand(size(X, n), R), 1:3, 'uni', 0);
            line = sprintf('%s R=%d nonneg=%d start=%2d |', name, R, nn, st);
            for im = 1:numel(cfg.methods)
                m = cfg.methods{im};
                o = struct('accel', strrep(m, 'dmdels', 'dmd'), 'nonneg', nn, 'init', {A0}, 'max_iters', cfg.max_iters, 'tol', cfg.tol, ...
                           'monitor', true, 'monitor_forecast', false, 'win', 6);
                if strcmp(m, 'dmd') && nn, o.linesearch = 'backtrack'; end
                if strcmp(m, 'dmdels'), o.linesearch = 'els'; end
                [A, out] = cp_als_dmd(X, R, o);
                r = struct('dataset', name, 'R', R, 'nonneg', nn, 'method', m, 'start', st, 'iters', out.niters, ...
                           'time', out.time, 'f_rel', out.fval / half_obs, 'stop', out.stop, 'n_accept', out.n_accept, ...
                           'n_try', out.n_try, 'n_mttkrp', out.n_mttkrp);
                [r.min_congr, r.max_abs_congr, r.wratio] = congruences(A);
                d = out.diag; K = out.niters;
                slow = d.rho > 0.99; r.flag_slow = first_run(slow, 3);
                degen = slow & d.pair_congr < -0.3; r.flag_degen = first_run(degen, 3);
                r.final_pair_congr = d.pair_congr(K);
                % forecast checkpoints (only meaningful when the run stopped by the tolerance rule)
                cp = struct('k', {}, 'rho', {}, 'fc_iters', {}, 'act_iters', {}, 'fc_dec', {}, 'act_dec', {});
                if strcmp(out.stop, 'tol')
                    for k = cfg.checkpoints(cfg.checkpoints < K)
                        cp(end+1) = struct('k', k, 'rho', d.rho(k), 'fc_iters', d.iters_to_tol(k), 'act_iters', K - k, ...
                                           'fc_dec', d.rem_dec_1mode(k), 'act_dec', out.f_hist(k) - out.f_hist(K)); %#ok<AGROW>
                    end
                end
                r.cp = cp; r.f_hist = out.f_hist; r.rho_hist = d.rho; r.congr_hist = d.pair_congr;
                results(end+1) = r; %#ok<AGROW>
                line = [line sprintf(' %-4s %4d it %6.1fs f=%.4e c=%+.2f |', m, out.niters, out.time, r.f_rel, r.min_congr)]; %#ok<AGROW>
            end
            fprintf('%s [%.0fs]\n', line, toc(t_all));
            if ~exist('results', 'dir'), mkdir('results'); end
            save(fullfile('results', sprintf('real_%s%s.mat', name, cfg.suffix)), 'results', 'cfg', 'info');
        end
    end
end
fprintf('done %s: %d runs in %.0f s\n', name, numel(results), toc(t_all));
end

function [mn, mx, wratio] = congruences(A)
% signed product over modes of the column cosines; mn = most negative pair, mx = largest |.|
N = numel(A); R = size(A{1}, 2);
T = ones(R);
for n = 1:N
    U = A{n} ./ max(sqrt(sum(A{n}.^2, 1)), eps);
    T = T .* (U' * U);
end
off = T(~eye(R));
if isempty(off), mn = NaN; mx = NaN; else, mn = min(off); mx = max(abs(off)); end
w = sqrt(sum(A{1}.^2, 1)); wratio = max(w) / max(min(w), eps);
end

function k = first_run(flag, len)
% first index at which 'flag' has been true for 'len' consecutive iterations (NaN if never)
flag = flag(:)'; flag(isnan(flag)) = false; k = NaN;
c = 0;
for i = 1:numel(flag)
    if flag(i), c = c + 1; if c >= len, k = i; return; end, else, c = 0; end
end
end
