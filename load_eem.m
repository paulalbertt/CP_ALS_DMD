function [X, info] = load_eem(name)
%LOAD_EEM  Public EEM fluorescence datasets (NaN = missing/scatter).
%   The N-way toolbox and drEEM folders are read from local_paths.m; copy
%   local_paths_example.m to local_paths.m and set the two paths for your machine.
if exist('local_paths', 'file') ~= 2
    error('load_eem:paths', 'Copy local_paths_example.m to local_paths.m and set the toolbox paths.');
end
P = local_paths(); NW = P.nway; DR = P.dreem;
ws = warning('off', 'all');
switch lower(name)
    case 'aminoacids'
        S = load(fullfile(NW, 'claus.mat')); X = S.X;
        info = 'amino acids (Bro), 5 samples x 201 em x 61 ex, 3 fluorophores, no missing values';
    case 'sugar'
        S = load(fullfile(NW, 'sugar.mat')); X = S.X;
        info = 'sugar process (Bro), 268 samples x 53 em x 7 ex, 4 % missing';
    case 'dorrit'
        S = load(fullfile(NW, 'dorrit.mat')); X = double(S.EEM.data);
        info = 'Dorrit (Baunsgaard), 27 samples x 551 em x 24 ex, scatter regions missing (1 %)';
    case 'dreem_tutorial'
        addpath(genpath(DR));
        S = load(fullfile(DR, 'examples', 'parafac_tutorial_01_start.mat')); X = double(S.samples.X);
        info = 'drEEM tutorial (Murphy), 89 samples x 125 em x 73 ex, scatter removed (14 % missing)';
    otherwise
        error('unknown dataset %s', name);
end
warning(ws);
end
