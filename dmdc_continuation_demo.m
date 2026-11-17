function out = dmdc_continuation_demo()
%DMDC_CONTINUATION_DEMO  Following the regularisation path out of a two-factor degeneracy.
% Same degenerate tensor and start as dmdc_damping_demo.  Two continuations with identical
% stop rules (pair congruence > -0.9, stage gain > 2 %, at most 8 stages):
%   fixed-mu path : mu = 1e-2, halve mu after every converged stage (classic continuation)
%   DMDc path     : stabilise at target rate 0.9, then raise the target rate in stages
%                   (0.9, 0.95, 0.975, ...) with the base damping removed and the model
%                   re-identified at every stage
% Prints the stage tables and writes results/fig_continuation.png.
maxNumCompThreads(1);
I = [30 30 30];
rng(3); nv = @(v) v / norm(v);
a = nv(randn(I(1), 1)); a2 = nv(randn(I(1), 1)); b = nv(randn(I(2), 1)); b2 = nv(randn(I(2), 1));
c = nv(randn(I(3), 1)); c2 = nv(randn(I(3), 1));
X = reshape(a * kron(c2, b)' + a * kron(c, b2)' + a2 * kron(c, b)', I);
X = X + 0.01 * norm(X(:)) / sqrt(numel(X)) * randn(I);
R = 2; rng(11); A0 = arrayfun(@(n) randn(I(n), R), 1:3, 'uni', 0);
half = 0.5 * sum(X(:).^2);
M = {struct('control','none'), struct('control','fixedpath','mu0',1e-2,'cont_mu_factor',0.5), ...
     struct('control','dmdc','mu0',1e-3,'lam_target',0.9,'continuation',true,'cont_mu_factor',0.5,'cont_lam_factor',1), ...
     struct('control','dmdc','mu0',1e-3,'lam_target',0.9,'continuation',true,'cont_mu_factor',0.5,'cont_lam_factor',0.5), ...
     struct('control','dmdc','mu0',1e-3,'lam_target',0.9,'continuation',true,'cont_mu_factor',0.1,'cont_lam_factor',1), ...
     struct('control','dmdc','mu0',1e-4,'lam_target',0.9,'congr_crit',-0.85)};
names = {'plain ALS', 'fixed-\mu path (halving from 10^{-2})', 'DMDc path, base halved, rate 0.9', ...
         'DMDc path, base halved, rate 0.9 \rightarrow 1', 'DMDc path, base /10, rate 0.9', 'plain ALS to congruence -0.85, then DMDc hold'};
cols = [42 120 214; 235 104 52; 98 80 214; 27 175 122; 227 73 72; 237 161 0] / 255;
out = cell(1, numel(M));
for i = 1:numel(M)
    o = M{i}; o.max_iters = 3000; o.init = A0;
    [~, out{i}] = cp_als_dmdc(X, R, o); r = out{i};
    fprintf('\n=== %s: %d iterations, stop = %s, final f = %.4e, max weight %.2f, congruence %.3f\n', ...
        strrep(strrep(names{i}, '\mu', 'mu'), '\rightarrow', '->'), r.niters, r.stop, r.fval / half, r.wmax_hist(end), r.congr_hist(end));
    if isfinite(r.t_on), fprintf('  switched on at iteration %d, final mu %.2e\n', r.t_on, r.mu_hist(end)); end
    if ~isempty(r.stages)
        fprintf('  %5s %9s %7s %7s %7s %10s %7s %7s  %s\n', 'stage', 'mu', 'lam_t', 'iters', 'cum', 'f', 'max w', 'congr', 'stop');
        for j = 1:numel(r.stages)
            st = r.stages(j);
            fprintf('  %5d %9.2e %7.4f %7d %7d %10.4e %7.2f %7.3f  %s\n', st.stage, st.mu, st.lam_t, st.iters, st.k_end, st.f / half, st.wmax, st.congr, st.stop_reason);
        end
    end
    if isfield(r, 'ident') && ~isempty(r.ident)
        fprintf('  identifications: %d attempts, %d accepted\n', numel(r.ident), sum([r.ident.ok]));
    end
end
% ---- iterations needed to reach a given point on the path (first iteration with congruence <= c)
levels = [-0.70 -0.75 -0.80 -0.85 -0.90];
fprintf('\niterations to first reach a pair congruence of ... (NaN = never):\n%-44s', 'method');
fprintf(' %7.2f', levels); fprintf('\n');
for i = 1:numel(M)
    r = out{i}; fprintf('%-44s', strrep(strrep(names{i}, '\mu', 'mu'), '\rightarrow', '->'));
    for c = levels
        k = find(r.congr_hist <= c, 1); if isempty(k), k = NaN; end
        fprintf(' %7.0f', k);
    end
    fprintf('\n');
end
fprintf('and the relative objective at those points:\n%-44s', 'method'); fprintf(' %7.2f', levels); fprintf('\n');
for i = 1:numel(M)
    r = out{i}; fprintf('%-44s', strrep(strrep(names{i}, '\mu', 'mu'), '\rightarrow', '->'));
    for c = levels
        k = find(r.congr_hist <= c, 1); if isempty(k), fprintf(' %7s', '-'); else, fprintf(' %7.1e', r.f_hist(k) / half); end
    end
    fprintf('\n');
end
% ---- figure: path in the (congruence, f) plane and f / mu vs iteration
fig = figure('Color', 'w', 'Position', [60 60 1050 360], 'Visible', 'off');
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile; hold(ax, 'on');
for i = 1:numel(M)
    r = out{i};
    plot(ax, r.congr_hist, r.f_hist / half, '-', 'Color', cols(i, :), 'LineWidth', 1.8, 'DisplayName', names{i});
    if ~isempty(r.stages)
        scatter(ax, [r.stages.congr], [r.stages.f] / half, 28, cols(i, :), 'filled', 'MarkerEdgeColor', 'w', 'HandleVisibility', 'off');
        for j = 1:numel(r.stages), text(ax, r.stages(j).congr, r.stages(j).f / half * 1.08, sprintf('%d', r.stages(j).k_end), 'FontSize', 7, 'Color', cols(i, :), 'HorizontalAlignment', 'center'); end
    end
end
set(ax, 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15, 'XLim', [-1 -0.4]);
xline(ax, -0.9, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
xlabel(ax, 'signed congruence of the pair'); ylabel(ax, 'relative objective'); title(ax, 'path (dots: converged stages, labels: cumulative iterations)', 'FontWeight', 'normal', 'FontSize', 9);
legend(ax, 'Location', 'northeast', 'Box', 'off', 'FontSize', 8);
ax = nexttile; hold(ax, 'on');
for i = 1:numel(M), r = out{i}; plot(ax, 1:r.niters, r.f_hist / half, '-', 'Color', cols(i, :), 'LineWidth', 1.8); end
set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);
xlabel(ax, 'iteration'); ylabel(ax, 'relative objective');
ax = nexttile; hold(ax, 'on');
for i = 2:numel(M), r = out{i}; plot(ax, 1:r.niters, max(r.mu_hist, 1e-6), '-', 'Color', cols(i, :), 'LineWidth', 1.8); end
set(ax, 'YLim', [1e-5 2e-2]);
set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);
xlabel(ax, 'iteration'); ylabel(ax, 'damping \mu_k');
title(tl, 'Continuation along the regularisation path, degenerate tensor, R = 2', 'FontSize', 11);
if ~exist('results', 'dir'), mkdir('results'); end
exportgraphics(fig, fullfile('results', 'fig_continuation.png'), 'Resolution', 150); close(fig);
fprintf('\nfigure written to results/fig_continuation.png\n');
end
