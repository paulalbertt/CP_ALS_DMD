function results = benchmark_cp_als_accel(cfg)
%BENCHMARK_CP_ALS_ACCEL  Simulation study: plain ALS vs. line searches, Anderson and DMD.
%
%   results = benchmark_cp_als_accel()        run with the default configuration
%   results = benchmark_cp_als_accel(cfg)     override fields of the default cfg below
%
% For every (congruence, noise level, replicate) a synthetic rank-R tensor and ONE random
% initialisation are generated; every method is started from that same initialisation.
% Results are appended to cfg.outfile after every instance, so a run can be summarised
% while it is still going (see summarize_benchmark.m).
%
% Recorded per run: iterations, wall time, final objective, relative objective
% f/(1/2||X||^2), factor match score vs. the true factors, accepted/tried acceleration
% steps, cost in MTTKRP equivalents, stop reason, and the full objective/time histories.
if nargin < 1, cfg = struct; end
def = struct('I', [50 50 50], 'R', 5, 'congr', [0.5 0.9 0.95], 'noise', [0 0.1], 'nrep', 20, ...
             'methods', {{'none', 'bro', 'els', 'aa', 'dmd', 'dmdbt'}}, 'max_iters', 5000, 'tol', 1e-8, ...
             'win', 6, 'seed0', 1000, 'outfile', fullfile('results', 'benchmark_results.mat'), ...
             'threads', 1, 'verbose', 1);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(cfg, fn{i}) || isempty(cfg.(fn{i})), cfg.(fn{i}) = def.(fn{i}); end
end
if ~isempty(cfg.threads), maxNumCompThreads(cfg.threads); end
outdir = fileparts(cfg.outfile);
if ~isempty(outdir) && ~exist(outdir, 'dir'), mkdir(outdir); end

% warm-up (JIT) so that the first timed run is not penalised
[Xw, ~] = make_cp_tensor([10 10 10], 2, 0.3, 0.1, 1);
for m = cfg.methods, o = method_opts(m{1}); o.max_iters = 20; cp_als_dmd(Xw, 2, o); end

results = struct('congr', {}, 'noise', {}, 'rep', {}, 'method', {}, 'iters', {}, 'time', {}, ...
                 'fval', {}, 'frel', {}, 'fms', {}, 'n_accept', {}, 'n_try', {}, 'n_mttkrp', {}, ...
                 'stop', {}, 'f_hist', {}, 't_hist', {});
t_all = tic; n_inst = 0;
for ic = 1:numel(cfg.congr)
    for in = 1:numel(cfg.noise)
        for rep = 1:cfg.nrep
            seed = cfg.seed0 + 1000*ic + 100*in + rep;
            [X, Atrue] = make_cp_tensor(cfg.I, cfg.R, cfg.congr(ic), cfg.noise(in), seed);
            rng(seed + 500000);
            A0 = arrayfun(@(n) randn(cfg.I(n), cfg.R), 1:numel(cfg.I), 'uni', 0);
            half_norm2 = 0.5 * sum(X(:).^2);
            line = sprintf('congr=%.2f noise=%.2f rep=%2d |', cfg.congr(ic), cfg.noise(in), rep);
            for im = 1:numel(cfg.methods)
                opts = method_opts(cfg.methods{im});
                opts.max_iters = cfg.max_iters; opts.tol = cfg.tol; opts.init = A0; opts.win = cfg.win;
                [A, out] = cp_als_dmd(X, cfg.R, opts);
                r = struct('congr', cfg.congr(ic), 'noise', cfg.noise(in), 'rep', rep, ...
                           'method', cfg.methods{im}, 'iters', out.niters, 'time', out.time, ...
                           'fval', out.fval, 'frel', out.fval / half_norm2, 'fms', fms_score(A, Atrue), ...
                           'n_accept', out.n_accept, 'n_try', out.n_try, 'n_mttkrp', out.n_mttkrp, ...
                           'stop', out.stop, 'f_hist', out.f_hist, 't_hist', out.t_hist);
                results(end+1) = r; %#ok<AGROW>
                line = [line, sprintf(' %s %5d it %6.2fs fms=%.3f |', cfg.methods{im}, out.niters, out.time, r.fms)]; %#ok<AGROW>
            end
            n_inst = n_inst + 1;
            if cfg.verbose, fprintf('%s  [%.0fs elapsed]\n', line, toc(t_all)); end
            save(cfg.outfile, 'results', 'cfg');
        end
    end
end
if cfg.verbose, fprintf('done: %d instances, %d runs, %.0f s\n', n_inst, numel(results), toc(t_all)); end
end

function opts = method_opts(name)
% method names: 'none' | 'bro' | 'els' | 'aa' | 'dmd' (exact line search) | 'dmdbt' (backtracking)
switch lower(name)
    case 'dmdbt', opts = struct('accel', 'dmd', 'linesearch', 'backtrack');
    otherwise,    opts = struct('accel', lower(name));
end
end
