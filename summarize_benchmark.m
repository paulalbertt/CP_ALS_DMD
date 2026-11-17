function S = summarize_benchmark(matfile, tag)
%SUMMARIZE_BENCHMARK  Tables and figures for a benchmark_cp_als_accel results file.
%
%   S = summarize_benchmark('results/benchmark_main.mat', 'main')
%
% Prints, per (congruence, noise) condition and method: median iterations, median cost in
% MTTKRP equivalents, median wall time, median paired cost ratio vs plain ALS, the share
% of runs that recovered the true factors (FMS >= 0.95), the share that hit the iteration
% cap, and the mean acceptance rate.  Also prints an "iterations/MTTKRPs to reach the best
% objective of the instance within 1e-6 of the initial excess" table (time-to-accuracy).
% Figures are written next to the results file as PNG.
if nargin < 2, tag = 'main'; end
L = load(matfile, 'results', 'cfg');
R = L.results; cfg = L.cfg;
methods = cfg.methods; nm = numel(methods);
congr = cfg.congr; noise = cfg.noise;
outdir = fileparts(matfile); if isempty(outdir), outdir = '.'; end

% ---------------------------------------------------------------- palette & names
names = containers.Map({'none','bro','els','aa','dmd','dmdbt'}, ...
    {'ALS', 'ALS + Bro line search', 'ALS + ELS (Rajih)', 'ALS + Anderson', 'ALS + DMD (ELS)', 'ALS + DMD (backtracking)'});
cols  = containers.Map({'none','bro','els','aa','dmd','dmdbt'}, ...
    {[42 120 214]/255, [235 104 52]/255, [27 175 122]/255, [237 161 0]/255, [232 123 164]/255, [0 131 0]/255});

% ---------------------------------------------------------------- time-to-accuracy per instance
% target: excess objective (f - f_best) reduced to 1e-6 of the initial excess, f_best = best final
% objective among the methods on that instance; a run that never reaches it gets Inf.
keys = arrayfun(@(r) sprintf('%g|%g|%d', r.congr, r.noise, r.rep), R, 'uni', 0);
[ukeys, ~, kid] = unique(keys);
tta_it = inf(numel(R), 1); tta_mt = inf(numel(R), 1); tta_t = inf(numel(R), 1); same_sol = true(numel(R), 1);
fexcess = zeros(numel(R), 1);   % (f_final - f_best of the instance) / f_best
for u = 1:numel(ukeys)
    idx = find(kid == u);
    fb = min([R(idx).fval]);
    fmsb = max([R(idx).fms]);
    for i = idx(:)'
        fh = R(i).f_hist; f0 = fh(1);
        target = fb + 1e-6 * (f0 - fb);
        k = find(fh <= target, 1);
        if ~isempty(k)
            tta_it(i) = k;
            tta_mt(i) = R(i).n_mttkrp * k / R(i).iters;      % average cost per iteration x k
            tta_t(i)  = R(i).t_hist(k);
        end
        same_sol(i) = abs(R(i).fms - fmsb) < 0.02;            % landed in the same solution as the best run
        fexcess(i) = (R(i).fval - fb) / max(fb, eps);
    end
end

% ---------------------------------------------------------------- tables
S = struct('congr', {}, 'noise', {}, 'method', {}, 'n', {}, 'med_iters', {}, 'med_mttkrp', {}, 'med_time', {}, ...
           'med_ratio_mttkrp', {}, 'med_ratio_iters', {}, 'pct_fms95', {}, 'pct_maxit', {}, 'accept_rate', {}, ...
           'med_tta_iters', {}, 'med_tta_mttkrp', {}, 'pct_tta_reached', {}, 'med_fexcess', {});
fprintf('\n==== %s: size %s, R=%d, %d replicates, tol=%.0e, max_iters=%d ====\n', tag, mat2str(cfg.I), cfg.R, cfg.nrep, cfg.tol, cfg.max_iters);
for ic = 1:numel(congr)
    for in = 1:numel(noise)
        fprintf('\n-- congruence %.2f, noise %.2f --\n', congr(ic), noise(in));
        fprintf('%-26s %7s %8s %8s %9s %9s %7s %7s %7s | %8s %8s %8s %9s\n', 'method', 'iters', 'MTTKRP', 'time[s]', 'cost/ALS', 'iter/ALS', 'FMS>.95', 'maxit', 'accept', 'tta_it', 'tta_mt', 'reached', 'df/fbest');
        sel0 = [R.congr] == congr(ic) & [R.noise] == noise(in);
        base = R(sel0 & strcmp({R.method}, 'none'));
        for im = 1:nm
            sel = sel0 & strcmp({R.method}, methods{im});
            r = R(sel); if isempty(r), continue; end
            % paired ratios vs ALS (same rep)
            ratio_mt = nan(1, numel(r)); ratio_it = nan(1, numel(r));
            for j = 1:numel(r)
                b = base([base.rep] == r(j).rep);
                if ~isempty(b), ratio_mt(j) = r(j).n_mttkrp / b.n_mttkrp; ratio_it(j) = r(j).iters / b.iters; end
            end
            acc = [r.n_accept] ./ max([r.n_try], 1); acc = acc([r.n_try] > 0);
            s = struct('congr', congr(ic), 'noise', noise(in), 'method', methods{im}, 'n', numel(r), ...
                'med_iters', median([r.iters]), 'med_mttkrp', median([r.n_mttkrp]), 'med_time', median([r.time]), ...
                'med_ratio_mttkrp', median(ratio_mt, 'omitnan'), 'med_ratio_iters', median(ratio_it, 'omitnan'), ...
                'pct_fms95', 100*mean([r.fms] >= 0.95), 'pct_maxit', 100*mean(strcmp({r.stop}, 'max_iters')), ...
                'accept_rate', mean(acc), 'med_tta_iters', median(tta_it(sel)), 'med_tta_mttkrp', median(tta_mt(sel)), ...
                'pct_tta_reached', 100*mean(isfinite(tta_it(sel))), 'med_fexcess', median(fexcess(sel)));
            S(end+1) = s; %#ok<AGROW>
            fprintf('%-26s %7.0f %8.0f %8.3f %9.2f %9.2f %6.0f%% %6.0f%% %6.0f%% | %8.0f %8.0f %7.0f%% %9.1e\n', names(methods{im}), ...
                s.med_iters, s.med_mttkrp, s.med_time, s.med_ratio_mttkrp, s.med_ratio_iters, s.pct_fms95, s.pct_maxit, 100*s.accept_rate, ...
                s.med_tta_iters, s.med_tta_mttkrp, s.pct_tta_reached, s.med_fexcess);
        end
    end
end
fprintf('\n(cost/ALS, iter/ALS: median over replicates of the paired ratio method/ALS; tta_*: median iterations / MTTKRPs\n');
fprintf(' needed to bring f within 1e-6 of the initial excess above the best final f of the instance; reached: share of runs that got there;\n');
fprintf(' df/fbest: median relative excess of the final objective over the best final objective of the instance, i.e. how far\n');
fprintf(' above the best solution a method stopped when its stopping rule fired)\n');

% ---------------------------------------------------------------- figure 1: cost vs congruence
fig = figure('Color', 'w', 'Position', [100 100 320*numel(noise) 320], 'Visible', 'off');
tl = tiledlayout(1, numel(noise), 'TileSpacing', 'compact', 'Padding', 'compact');
for in = 1:numel(noise)
    ax = nexttile; hold(ax, 'on');
    for im = 1:nm
        y = arrayfun(@(c) median([R([R.congr]==c & [R.noise]==noise(in) & strcmp({R.method}, methods{im})).n_mttkrp]), congr);
        plot(ax, congr, y, '-o', 'Color', cols(methods{im}), 'LineWidth', 2, 'MarkerSize', 6, ...
             'MarkerFaceColor', cols(methods{im}), 'MarkerEdgeColor', 'w', 'DisplayName', names(methods{im}));
    end
    set(ax, 'YScale', 'log', 'XTick', congr, 'Box', 'off', 'FontSize', 10, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);
    xlabel(ax, 'factor congruence'); if in == 1, ylabel(ax, 'median cost (MTTKRP equivalents)'); end
    title(ax, sprintf('noise %g%%', 100*noise(in)), 'FontWeight', 'normal');
end
lg = legend(ax, 'Location', 'eastoutside', 'Box', 'off'); lg.Layout.Tile = 'east';
title(tl, sprintf('Cost to convergence, %s, R=%d (median of %d replicates)', mat2str(cfg.I), cfg.R, cfg.nrep), 'FontSize', 11);
exportgraphics(fig, fullfile(outdir, sprintf('fig_cost_%s.png', tag)), 'Resolution', 150); close(fig);

% ---------------------------------------------------------------- figure 2: convergence curves on one hard instance
cI = congr(min(numel(congr), find(congr >= 0.9, 1))); nI = noise(min(numel(noise), 2)); repI = 1;
fig = figure('Color', 'w', 'Position', [100 100 760 330], 'Visible', 'off');
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
sel = find([R.congr] == cI & [R.noise] == nI & [R.rep] == repI);
fb = min([R(sel).fval]);
for panel = 1:2
    ax = nexttile; hold(ax, 'on');
    for i = sel
        r = R(i); ex = max(r.f_hist - fb, eps); 
        if panel == 1, x = 1:r.iters; else, x = r.n_mttkrp * (1:r.iters) / r.iters; end
        plot(ax, x, ex, '-', 'Color', cols(r.method), 'LineWidth', 2, 'DisplayName', names(r.method));
    end
    set(ax, 'YScale', 'log', 'XScale', 'log', 'Box', 'off', 'FontSize', 10, 'YGrid', 'on', 'GridAlpha', 0.15);
    if panel == 1, xlabel(ax, 'iteration (ALS sweep)'); ylabel(ax, 'f_k - f_{best}'); else, xlabel(ax, 'cost (MTTKRP equivalents)'); end
end
lg = legend(ax, 'Location', 'eastoutside', 'Box', 'off'); lg.Layout.Tile = 'east';
title(tl, sprintf('Excess objective on one instance (congruence %.2f, noise %g%%, replicate %d)', cI, 100*nI, repI), 'FontSize', 11);
exportgraphics(fig, fullfile(outdir, sprintf('fig_convergence_%s.png', tag)), 'Resolution', 150); close(fig);

% ---------------------------------------------------------------- figure 3: time vs iterations, all runs
fig = figure('Color', 'w', 'Position', [100 100 560 360], 'Visible', 'off'); ax = axes(fig); hold(ax, 'on');
for im = 1:nm
    r = R(strcmp({R.method}, methods{im}));
    scatter(ax, [r.iters], [r.time], 18, cols(methods{im}), 'filled', 'MarkerFaceAlpha', 0.45, 'DisplayName', names(methods{im}));
end
set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'FontSize', 10, 'YGrid', 'on', 'GridAlpha', 0.15);
xlabel(ax, 'iterations'); ylabel(ax, 'wall time [s] (single thread)');
legend(ax, 'Location', 'eastoutside', 'Box', 'off');
title(ax, sprintf('All runs, %s, R=%d', mat2str(cfg.I), cfg.R), 'FontWeight', 'normal');
exportgraphics(fig, fullfile(outdir, sprintf('fig_time_vs_iters_%s.png', tag)), 'Resolution', 150); close(fig);
fprintf('figures written to %s (fig_cost_%s.png, fig_convergence_%s.png, fig_time_vs_iters_%s.png)\n', outdir, tag, tag, tag);
end
