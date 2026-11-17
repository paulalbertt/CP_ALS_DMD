function S = summarize_real_data()
%SUMMARIZE_REAL_DATA  Tables and figures for the real-data runs (results/real_*.mat).
files = dir(fullfile('results', 'real_*.mat'));
R = []; infos = {};
for i = 1:numel(files)
    L = load(fullfile('results', files(i).name));
    if ~isfield(L, 'results'), continue; end
    if isempty(R), R = L.results; else, R = [R, L.results]; end %#ok<AGROW>
    if ~any(strcmp(infos, L.info)), infos{end+1} = L.info; end %#ok<AGROW>
end
names = containers.Map({'none','dmd','aa','bro','dmdels'}, {'ALS', 'ALS + DMD (backtr.)', 'ALS + Anderson', 'ALS + Bro line search', 'ALS + DMD (ELS)'});
cols  = containers.Map({'none','dmd','aa','bro','dmdels'}, {[42 120 214]/255, [232 123 164]/255, [237 161 0]/255, [235 104 52]/255, [0 131 0]/255});
S = struct('dataset', {}, 'R', {}, 'nonneg', {}, 'method', {}, 'n', {}, 'med_iters', {}, 'med_time', {}, ...
           'med_frel', {}, 'frel_spread', {}, 'pct_degen', {}, 'pct_capped', {}, 'med_iter_ratio', {}, 'accept', {});
fprintf('\n%s\n', strjoin(infos, newline));
datasets = unique({R.dataset}, 'stable');
for d = datasets
    sel_d = strcmp({R.dataset}, d{1});
    for Rv = unique([R(sel_d).R])
        for nn = unique([R(sel_d & [R.R] == Rv).nonneg])
            sel = sel_d & [R.R] == Rv & [R.nonneg] == nn;
            fprintf('\n-- %s, R = %d, %s --\n', d{1}, Rv, tern(nn, 'nonnegative', 'unconstrained'));
            fprintf('%-22s %6s %8s %11s %21s %7s %7s %9s %7s\n', 'method', 'iters', 'time[s]', 'f_rel', 'f_rel min..max', 'degen', 'capped', 'iter/ALS', 'accept');
            base = R(sel & strcmp({R.method}, 'none'));
            for m = {'none', 'bro', 'dmd', 'dmdels', 'aa'}
                r = R(sel & strcmp({R.method}, m{1})); if isempty(r), continue; end
                ratio = nan(1, numel(r));
                for j = 1:numel(r), b = base([base.start] == r(j).start); if ~isempty(b), ratio(j) = r(j).iters / b.iters; end, end
                acc = [r.n_accept] ./ max([r.n_try], 1); acc = acc([r.n_try] > 0);
                s = struct('dataset', d{1}, 'R', Rv, 'nonneg', nn, 'method', m{1}, 'n', numel(r), ...
                    'med_iters', median([r.iters]), 'med_time', median([r.time]), 'med_frel', median([r.f_rel]), ...
                    'frel_spread', [min([r.f_rel]) max([r.f_rel])], 'pct_degen', 100*mean([r.min_congr] < -0.85), ...
                    'pct_capped', 100*mean(strcmp({r.stop}, 'max_iters')), 'med_iter_ratio', median(ratio, 'omitnan'), 'accept', mean(acc));
                S(end+1) = s; %#ok<AGROW>
                fprintf('%-22s %6.0f %8.1f %11.4e %10.3e..%9.3e %6.0f%% %6.0f%% %9.2f %6.0f%%\n', names(m{1}), s.med_iters, s.med_time, ...
                    s.med_frel, s.frel_spread(1), s.frel_spread(2), s.pct_degen, s.pct_capped, s.med_iter_ratio, 100*s.accept);
            end
        end
    end
end
% ---- forecast accuracy on real data
fprintf('\n== DMD forecast accuracy on real data (runs that stopped by the tolerance rule) ==\n');
fprintf('%-16s %6s %8s %12s %9s %9s\n', 'dataset', 'n_cp', 'usable', 'med ratio', 'within x2', 'within x3');
allr = []; alld = {};
for d = datasets
    cp = [R(strcmp({R.dataset}, d{1})).cp]; if isempty(cp), continue; end
    ok = isfinite([cp.fc_iters]) & [cp.rho] < 1;
    rr = log10([cp(ok).fc_iters] ./ max([cp(ok).act_iters], 1));
    fprintf('%-16s %6d %7.0f%% %12.2f %8.0f%% %8.0f%%\n', d{1}, numel(cp), 100*mean(ok), 10^median(rr), 100*mean(abs(rr) < log10(2)), 100*mean(abs(rr) < log10(3)));
    allr = [allr rr]; alld = [alld repmat(d, 1, numel(rr))]; %#ok<AGROW>
end
if ~isempty(allr), fprintf('%-16s %6s %7s %12.2f %8.0f%% %8.0f%%\n', 'all', '', '', 10^median(allr), 100*mean(abs(allr) < log10(2)), 100*mean(abs(allr) < log10(3))); end
% ---- degeneracy early warning
fprintf('\n== degeneracy warning (rate > 0.99 and negative slow-mode pair congruence for 3 iterations) ==\n');
fprintf('%-16s %3s %5s %8s %10s %14s %16s\n', 'dataset', 'R', 'nn', 'n runs', 'degen end', 'flag/degenerate', 'flag/non-degen.');
for d = datasets
    for Rv = unique([R(strcmp({R.dataset}, d{1})).R])
        for nn = [true false]
            sel = strcmp({R.dataset}, d{1}) & [R.R] == Rv & [R.nonneg] == nn; if ~any(sel), continue; end
            r = R(sel); dg = [r.min_congr] < -0.85;
            fl = isfinite([r.flag_degen]);
            if any(dg), med_frac = median([r(dg).flag_degen] ./ [r(dg).iters], 'omitnan'); else, med_frac = NaN; end
            fprintf('%-16s %3d %5d %8d %9.0f%% %7.0f%% @%.2f %15.0f%%\n', d{1}, Rv, nn, numel(r), 100*mean(dg), 100*mean(fl(dg)), med_frac, 100*mean(fl(~dg)));
        end
    end
end
fprintf('(flag/degenerate: share of runs ending degenerate (min pair congruence < -0.85) in which the flag fired, and the median\n fraction of the run at which it fired; flag/non-degen.: share of non-degenerate runs with a flag)\n');
% ---- figures
cmap = [42 120 214; 235 104 52; 27 175 122; 237 161 0] / 255;
fig = figure('Color', 'w', 'Position', [60 60 980 360], 'Visible', 'off');
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile; hold(ax, 'on');
for i = 1:numel(datasets)
    cp = [R(strcmp({R.dataset}, datasets{i})).cp]; if isempty(cp), continue; end
    ok = isfinite([cp.fc_iters]) & [cp.rho] < 1;
    scatter(ax, max([cp(ok).act_iters], 1), [cp(ok).fc_iters], 20, cmap(i, :), 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', strrep(datasets{i}, '_', ' '));
end
lim = [1 1e4]; plot(ax, lim, lim, 'k-', 'HandleVisibility', 'off'); plot(ax, lim, 2*lim, 'k:', 'HandleVisibility', 'off'); plot(ax, lim, lim/2, 'k:', 'HandleVisibility', 'off');
set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15, 'XLim', lim, 'YLim', lim);
xlabel(ax, 'actual remaining iterations'); ylabel(ax, 'DMD forecast'); title(ax, 'forecast of iterations to the stopping rule, real EEM data', 'FontWeight', 'normal', 'FontSize', 9);
legend(ax, 'Location', 'northwest', 'Box', 'off');
% convergence curves: dorrit R=4 nonneg start 1 if present, else first dataset
ax = nexttile; hold(ax, 'on');
sel = find(strcmp({R.dataset}, 'dorrit') & [R.R] == 4 & [R.nonneg] & [R.start] == 1);
if isempty(sel), sel = find(strcmp({R.dataset}, datasets{1}) & [R.start] == 1 & [R.R] == min([R.R]) & [R.nonneg]); end
fb = min(arrayfun(@(i) min(R(i).f_hist), sel));
for i = sel
    plot(ax, 1:R(i).iters, max(R(i).f_hist - fb, eps), '-', 'Color', cols(R(i).method), 'LineWidth', 1.8, 'DisplayName', names(R(i).method));
end
set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);
xlabel(ax, 'iteration'); ylabel(ax, 'f_k - f_{best}'); title(ax, sprintf('%s, R = %d, nonnegative, start 1', strrep(R(sel(1)).dataset, '_', ' '), R(sel(1)).R), 'FontWeight', 'normal', 'FontSize', 9);
legend(ax, 'Location', 'southwest', 'Box', 'off');
exportgraphics(fig, fullfile('results', 'fig_real_data.png'), 'Resolution', 150); close(fig);
% degeneracy trace: an unconstrained over-factored run that ended degenerate
dsel = find(~[R.nonneg] & strcmp({R.method}, 'none') & [R.min_congr] < -0.85);
if ~isempty(dsel)
    i = dsel(1);
    fig = figure('Color', 'w', 'Position', [60 60 980 300], 'Visible', 'off');
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax = nexttile; plot(ax, 1:R(i).iters, R(i).f_hist / R(i).f_hist(1), '-', 'Color', cmap(1,:), 'LineWidth', 1.8); set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on'); xlabel(ax, 'iteration'); ylabel(ax, 'f_k / f_1');
    ax = nexttile; plot(ax, 1:R(i).iters, min(R(i).rho_hist, 1.05), '-', 'Color', cmap(1,:), 'LineWidth', 1.8); yline(ax, 1, ':'); set(ax, 'XScale', 'log', 'Box', 'off', 'YGrid', 'on', 'YLim', [0 1.1]); xlabel(ax, 'iteration'); ylabel(ax, 'DMD rate max |\lambda|');
    ax = nexttile; plot(ax, 1:R(i).iters, R(i).congr_hist, '-', 'Color', cmap(1,:), 'LineWidth', 1.8); yline(ax, -1, ':'); set(ax, 'XScale', 'log', 'Box', 'off', 'YGrid', 'on', 'YLim', [-1.05 1.05]); xlabel(ax, 'iteration'); ylabel(ax, 'slow-mode pair congruence');
    if isfinite(R(i).flag_degen), for a = tl.Children', xline(a, R(i).flag_degen, '--', 'Color', [0.4 0.4 0.4]); end, end
    title(tl, sprintf('%s, R = %d, unconstrained, start %d: a run that ends in a two-factor degeneracy (dashed: warning fired)', strrep(R(i).dataset, '_', ' '), R(i).R, R(i).start), 'FontSize', 10);
    exportgraphics(fig, fullfile('results', 'fig_real_degeneracy.png'), 'Resolution', 150); close(fig);
end
fprintf('\nfigures: results/fig_real_data.png, results/fig_real_degeneracy.png (if a degenerate run exists)\n');
end
function o = tern(c, a, b)
if c, o = a; else, o = b; end
end
