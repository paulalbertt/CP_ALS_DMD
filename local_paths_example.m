function P = local_paths()
%LOCAL_PATHS  Where the external toolboxes and datasets live on this machine.
%   Copy this file to local_paths.m (ignored by git) and edit the two paths.
%     P.nway  : N-way toolbox folder (fastnnls, nmodel, ... and the example datasets
%               claus.mat, sugar.mat, dorrit.mat used by load_eem)
%     P.dreem : drEEM toolbox folder (examples/parafac_tutorial_01_start.mat)
P.nway  = '/path/to/Nway Version 3.6';
P.dreem = '/path/to/drEEM-2';
end
