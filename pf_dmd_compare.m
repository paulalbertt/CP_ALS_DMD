function res = pf_dmd_compare(cfg)
%PF_DMD_COMPARE  N-way toolbox parafac.m with nonnegativity: 'none' vs 'shifted' vs 'dmd'.
%
%   res = pf_dmd_compare()        % defaults mirror PF_DMD_demo.m (100x150x180, R=5, rho=0.8,
%                                 % half-normal nonnegative factors, SNR 10 dB, const=[2 2 2])
%   res = pf_dmd_compare(cfg)     % override fields
%
% Needs the N-way toolbox (nmodel, krb, pfls, ini, ...) and addTensorNoise on the path.
if nargin < 1, cfg = struct; end
def = struct('I', [100 150 180], 'R', 5, 'rho', [0.8 0.95], 'snr_dB', [10 20], 'nrep', 5, ...
             'methods', {{'none', 'shifted', 'dmd'}}, 'maxit', 10000, 'crit', 1e-8, ...
             'outfile', fullfile('results', 'parafac_compare.mat'), 'threads', 1, 'seed0', 100);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(cfg, fn{i}) || isempty(cfg.(fn{i})), cfg.(fn{i}) = def.(fn{i}); end
end
if ~isempty(cfg.threads), maxNumCompThreads(cfg.threads); end
outdir = fileparts(cfg.outfile); if ~isempty(outdir) && ~exist(outdir, 'dir'), mkdir(outdir); end
N = numel(cfg.I);
res = struct('rho', {}, 'snr_dB', {}, 'rep', {}, 'method', {}, 'iters', {}, 'time', {}, 'err', {}, 'fms', {});
t_all = tic;
for ir = 1:numel(cfg.rho)
    for is = 1:numel(cfg.snr_dB)
        for rep = 1:cfg.nrep
            seed = cfg.seed0 + 100*ir + 10*is + rep; rng(seed);
            Sigma = (1 - cfg.rho(ir)) * eye(cfg.R) + cfg.rho(ir) * ones(cfg.R); L = chol(Sigma, 'lower');
            Atrue = cell(1, N);
            for n = 1:N
                A = randn(cfg.I(n), cfg.R) * L';          % rows ~ N(0, Sigma): column correlation rho
                A = max(A, 0) + 1e-9;                     % half-normal, nonnegative (as in PF_DMD_demo)
                Atrue{n} = A ./ sqrt(sum(A.^2, 1));
            end
            Atrue{1} = Atrue{1} .* (abs(randn(1, cfg.R)) + 1e-9);
            X = nmodel(Atrue); X = reshape(X, cfg.I);
            X = addTensorNoise(X, cfg.snr_dB(is));
            X = max(X, 1e-9);
            Old = arrayfun(@(n) rand(cfg.I(n), cfg.R), 1:N, 'uni', 0);
            Options = [cfg.crit 2 0 1 NaN cfg.maxit];
            line = sprintf('rho=%.2f snr=%2d dB rep=%d |', cfg.rho(ir), cfg.snr_dB(is), rep);
            for im = 1:numel(cfg.methods)
                t = tic;
                evalc('[F, it, err] = parafac(X, cfg.R, Options, [2 2 2], Old, [], [], cfg.methods{im});');
                tt = toc(t);
                r = struct('rho', cfg.rho(ir), 'snr_dB', cfg.snr_dB(is), 'rep', rep, 'method', cfg.methods{im}, ...
                           'iters', it, 'time', tt, 'err', err, 'fms', fms_score(F, Atrue));
                res(end+1) = r; %#ok<AGROW>
                line = [line, sprintf(' %-7s %5d it %7.2fs err=%.6e fms=%.3f |', cfg.methods{im}, it, tt, err, r.fms)]; %#ok<AGROW>
            end
            fprintf('%s [%.0fs]\n', line, toc(t_all));
            save(cfg.outfile, 'res', 'cfg');
        end
    end
end
end
