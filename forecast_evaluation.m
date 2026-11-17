function E = forecast_evaluation(cfg)
%FORECAST_EVALUATION  How accurate is the DMD convergence forecast for plain ALS?
%
% For every instance plain ALS is run with opts.monitor until the relative-change rule
% (tol) fires.  At checkpoints k the forecast made from the last 6 iterates is compared
% with what actually happened afterwards:
%   iterations:  diag.iters_to_tol(k)   vs  K - k         (K = iteration at which ALS stopped)
%   decrease:    diag.rem_dec_1mode(k)  vs  f_k - f_K
% In addition the iteration k6 at which the N-way toolbox default rule (relative change
% < 1e-6) would have stopped is located, and the forecast of the remaining relative
% decrease at k6 is compared with the actual one (f_k6 - f_K) / f_K.
% Writes results/forecast_eval.mat and results/fig_forecast.png.
if nargin < 1, cfg = struct; end
def = struct('I', [40 40 40], 'R', 4, 'congr', [0.5 0.8 0.9 0.95], 'noise', 0.01, 'nrep', 8, ...
             'tol', 1e-8, 'max_iters', 6000, 'checkpoints', [10 20 50 100 200 500 1000 2000], 'seed0', 3000);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(cfg, fn{i}) || isempty(cfg.(fn{i})), cfg.(fn{i}) = def.(fn{i}); end
end
maxNumCompThreads(1);
E = struct('congr', {}, 'rep', {}, 'k', {}, 'K', {}, 'rho', {}, 'fc_iters', {}, 'act_iters', {}, ...
           'fc_dec', {}, 'act_dec', {}, 'kind', {});
n_capped = 0;
for ic = 1:numel(cfg.congr)
    for rep = 1:cfg.nrep
        seed = cfg.seed0 + 100*ic + rep;
        [X, ~] = make_cp_tensor(cfg.I, cfg.R, cfg.congr(ic), cfg.noise, seed);
        o = struct('accel', 'none', 'monitor', true, 'monitor_forecast', false, 'max_iters', cfg.max_iters, 'tol', cfg.tol, 'seed', seed + 7);
        [~, out] = cp_als_dmd(X, cfg.R, o);
        if strcmp(out.stop, 'max_iters'), n_capped = n_capped + 1; continue; end
        d = out.diag; K = out.niters; f = out.f_hist;
        ks = cfg.checkpoints(cfg.checkpoints < K);
        rel = [NaN; abs(diff(f)) ./ f(1:end-1)];
        k6 = find(rel < 1e-6, 1);
        if ~isempty(k6) && ~any(ks == k6), ks = [ks k6]; end %#ok<AGROW>
        for k = ks
            E(end+1) = struct('congr', cfg.congr(ic), 'rep', rep, 'k', k, 'K', K, 'rho', d.rho(k), ...
                'fc_iters', d.iters_to_tol(k), 'act_iters', K - k, 'fc_dec', d.rem_dec_1mode(k), ...
                'act_dec', f(k) - f(K), 'kind', tern(~isempty(k6) && k == k6, 'tol6', 'checkpoint')); %#ok<AGROW>
        end
        fprintf('congr %.2f rep %d: K = %d, 1e-6 rule at k = %s\n', cfg.congr(ic), rep, K, mat2str(k6));
    end
end
if ~exist('results', 'dir'), mkdir('results'); end
save(fullfile('results', 'forecast_eval.mat'), 'E', 'cfg');

% ---- summary
fprintf('\n%d runs hit the iteration cap and were excluded\n', n_capped);
ok = isfinite([E.fc_iters]) & [E.rho] < 1;
fprintf('checkpoints: %d, of which %d (%.0f%%) had a usable forecast (max|lambda| < 1)\n', numel(E), sum(ok), 100*mean(ok));
r_it = log10([E(ok).fc_iters] ./ max([E(ok).act_iters], 1));
r_dc = log10(max([E(ok).fc_dec], realmin) ./ max([E(ok).act_dec], realmin));
fprintf('\nforecast / actual (all usable checkpoints):\n');
fprintf('  iterations: median ratio %.2f, within x2: %.0f%%, within x3: %.0f%%\n', 10^median(r_it), 100*mean(abs(r_it) < log10(2)), 100*mean(abs(r_it) < log10(3)));
fprintf('  remaining decrease: median ratio %.2f, within x2: %.0f%%, within x3: %.0f%%\n', 10^median(r_dc), 100*mean(abs(r_dc) < log10(2)), 100*mean(abs(r_dc) < log10(3)));
fprintf('\nby congruence (iterations forecast, usable checkpoints only):\n');
for c = cfg.congr
    sel = ok & [E.congr] == c; if ~any(sel), continue; end
    rr = log10([E(sel).fc_iters] ./ max([E(sel).act_iters], 1));
    fprintf('  congruence %.2f: n=%3d, median ratio %.2f, within x2 %.0f%%, within x3 %.0f%%, usable %.0f%%\n', c, sum(sel), 10^median(rr), 100*mean(abs(rr) < log10(2)), 100*mean(abs(rr) < log10(3)), 100*mean(ok([E.congr] == c)));
end
fprintf('\nby checkpoint (iterations forecast):\n');
for k = cfg.checkpoints
    sel = ok & [E.k] == k & strcmp({E.kind}, 'checkpoint'); if ~any(sel), continue; end
    rr = log10([E(sel).fc_iters] ./ max([E(sel).act_iters], 1));
    fprintf('  k = %4d: n=%3d, median ratio %.2f, within x2 %.0f%%, within x3 %.0f%%\n', k, sum(sel), 10^median(rr), 100*mean(abs(rr) < log10(2)), 100*mean(abs(rr) < log10(3)));
end
% ---- the toolbox default rule (1e-6): how much fit was still to come, and did the forecast say so?
t6 = strcmp({E.kind}, 'tol6');
if any(t6)
    act_rel = [E(t6).act_dec] ./ ([E(t6).K] * 0 + 1);   % absolute remaining decrease
    fprintf('\nAt the iteration where the 1e-6 relative-change rule fires (%d runs):\n', sum(t6));
    fprintf('  median actual remaining iterations to the 1e-8 rule: %.0f; median forecast: %.0f\n', median([E(t6).act_iters]), median([E(t6).fc_iters], 'omitnan'));
    big = [E(t6).act_dec] > 1e-4 * 0.5;  % placeholder, replaced below
    fc = [E(t6).fc_dec]; ac = [E(t6).act_dec];
    fprintf('  remaining decrease of f: median actual %.2e, median forecast %.2e, forecast within x3 of actual in %.0f%% of runs\n', median(ac), median(fc, 'omitnan'), 100*mean(abs(log10(max(fc, realmin) ./ max(ac, realmin))) < log10(3)));
end
% ---- figure
cols = [42 120 214; 235 104 52; 27 175 122; 237 161 0] / 255;
fig = figure('Color', 'w', 'Position', [60 60 820 360], 'Visible', 'off');
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile; hold(ax, 'on');
for i = 1:numel(cfg.congr)
    sel = ok & [E.congr] == cfg.congr(i);
    scatter(ax, max([E(sel).act_iters], 1), [E(sel).fc_iters], 22, cols(i, :), 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', sprintf('congruence %.2f', cfg.congr(i)));
end
lim = [1 1e4]; plot(ax, lim, lim, 'k-', 'LineWidth', 0.8, 'HandleVisibility', 'off'); plot(ax, lim, 2*lim, 'k:', 'HandleVisibility', 'off'); plot(ax, lim, lim/2, 'k:', 'HandleVisibility', 'off');
set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15, 'XLim', lim, 'YLim', lim);
xlabel(ax, 'actual remaining iterations'); ylabel(ax, 'DMD forecast'); title(ax, 'iterations until the stopping rule fires', 'FontWeight', 'normal');
legend(ax, 'Location', 'northwest', 'Box', 'off');
ax = nexttile; hold(ax, 'on');
for i = 1:numel(cfg.congr)
    sel = ok & [E.congr] == cfg.congr(i);
    scatter(ax, max([E(sel).act_dec], 1e-16), max([E(sel).fc_dec], 1e-16), 22, cols(i, :), 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', sprintf('congruence %.2f', cfg.congr(i)));
end
lim = [1e-12 1]; plot(ax, lim, lim, 'k-', 'LineWidth', 0.8, 'HandleVisibility', 'off'); plot(ax, lim, 2*lim, 'k:', 'HandleVisibility', 'off'); plot(ax, lim, lim/2, 'k:', 'HandleVisibility', 'off');
set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15, 'XLim', lim, 'YLim', lim);
xlabel(ax, 'actual remaining decrease of f'); ylabel(ax, 'DMD forecast'); title(ax, 'remaining decrease of the objective', 'FontWeight', 'normal');
title(tl, sprintf('DMD convergence forecast vs outcome, plain ALS, %s, R=%d, %d instances (dotted: factor 2)', mat2str(cfg.I), cfg.R, numel(cfg.congr)*cfg.nrep), 'FontSize', 11);
exportgraphics(fig, fullfile('results', 'fig_forecast.png'), 'Resolution', 150); close(fig);
fprintf('figure written to results/fig_forecast.png\n');
end

function o = tern(c, a, b)
if c, o = a; else, o = b; end
end
