

%% Synthetic CP tensor with controllable column-correlation and nonnegativity
%rng('default'); % reproducibility
clear all
for k = 1:1000
I           = [100, 150, 180];     % tensor size (3-way example)
R_true      = 5;                % true rank used to synthesize data
R_est       = 5;                % rank used in factorization (can mismatch)
noise_sigma = 1;             % std of additive Gaussian noise (0 = noiseless)

% ---- NEW: correlation / positivity controls ----
% corr_rho can be a scalar (same for all modes) or a 1xN vector per-mode.
corr_rho    = 0.8;              % pairwise column correlation target (in logs if using 'lognormal' method)
corr_Sigma  = [];               % alternatively: provide a full R_true x R_true covariance (or one per mode as a cell)
pos_method  = 'abs';      % 'none' | 'abs' | 'lognormal'
clamp_after_noise = true;       % set true to clip negatives after adding noise
eps_pos     = 1e-9;             % tiny floor for strict positivity
% ----------------------------------------------

fprintf('Generating synthetic CP tensor of size %s with rank %d...\n', mat2str(I), R_true);

N     = numel(I);
Atrue = cell(1, N);

% ensure corr_rho is per-mode
if isscalar(corr_rho), corr_rho = repmat(corr_rho, 1, N); end

for n = 1:N
    % Build a column covariance/cholesky for this mode.
    if ~isempty(corr_Sigma)
        % Accept either a single Sigma or a cell array of Sigmas per mode
        if iscell(corr_Sigma), Sigma = corr_Sigma{n}; else, Sigma = corr_Sigma; end
    else
        % Equicorrelated covariance: Sigma = (1-rho)I + rho*11^T
        rho   = max(min(corr_rho(n), 0.999), -0.999);   % keep PD
        Sigma = (1 - rho) * eye(R_true) + rho * ones(R_true);
    end
    L = chol(Sigma, 'lower');   % R_true x R_true

    % Base Gaussian draws with identity column-covariance
    Z = randn(I(n), R_true);

    % Impose the desired column correlation across components
    A = Z * L;                  % columns now have covariance ~ Sigma

    % Enforce nonnegativity if requested
    switch lower(pos_method)
        case 'abs'
            A = max(A, 0) + eps_pos;                  % half-normal; simple and fast
        case 'lognormal'
            % Use a log-normal draw: preserves correlation structure in log-domain
            A = exp(A);                                % strictly positive
        case 'none'
            % leave as is (can be sign-indefinite)
        otherwise
            error('Unknown pos_method: %s', pos_method);
    end

    % Normalize columns for a well-scaled model
    for r = 1:R_true
        nr = norm(A(:, r));
        if nr > 0, A(:, r) = A(:, r) / nr; else, A(:, r) = eps_pos * ones(I(n), 1); end
    end

    Atrue{n} = A;
end

% Positive weights (absorbed into mode-1)
lambda   = abs(randn(1, R_true)) + eps_pos;
Atrue{1} = Atrue{1} .* lambda;

% Build dense tensor
X = full_from_cp(Atrue);

% Add noise (Gaussian by default)
if noise_sigma > 0
    X = addTensorNoise(X,10);
end

% If you want the resulting tensor strictly nonnegative, clamp after noise
if clamp_after_noise
    X = max(X, eps_pos);
end

fprintf('Done. Tensor created with pos_method=%s and target rho=%s\n', pos_method, mat2str(corr_rho));


%% fit model
%Initialize loading matrices
OldLoad{1} = rand(size(Atrue{1}));
OldLoad{2} = rand(size(Atrue{2}));
OldLoad{3} = rand(size(Atrue{3}));

%%
Options(1) = 1e-08;
Options(2) = 2;
Options(3) = 0;
Options(4) = 1;
Options(5) = NaN;
Options(6) = 10000;
accel_meth = {'none';'shifted';'dmd'};
for n = 1:length(accel_meth)
%PF contraints
const = [2 2 2];
%PF model
tic()
[Factors,it,err,corcondia]=parafac(X,R_est,Options,const,OldLoad,[],[],accel_meth{n});
all_times(k,n) = toc();
all_iter(k,n) =  it;
    for nn=1:numel(I)
        C = corr(Factors{nn},Atrue{nn}); %#ok<CORR>
        sim_temp(nn) = mean(max(C));
        %fprintf('  Mode %d: max |corr| = %.3f\n', n, s);
    end
all_sim{k,n} = sim_temp;
all_errs(k,n) = err;

end
end
%% ---------------- local helpers ----------------
function X = full_from_cp(A)
%FULL_FROM_CP Build a dense tensor from CP factors (weights absorbed in mode-1)
    N = numel(A);
    Is = cellfun(@(Ai) size(Ai,1), A);
    R = size(A{1},2);
    X = zeros(Is);
    for r=1:R
        outer = A{1}(:,r);
        for n=2:N
            outer = ktimes(outer, A{n}(:,r));
        end
        X = X + outer;
    end
end

function Y = ktimes(a, b)
%KTIMES n-D outer product accumulator
    if isvector(a) && isvector(b)
        Y = reshape(a, [], 1) * reshape(b, 1, []);
    else
        % expand a by outer product with vector b on a new trailing mode
        %Y = bsxfun(@times, a, reshape(b, 1, [])); %#ok<BSXFUN>
        Y = bsxfun(@times, reshape(b, [], 1),reshape(a,[1 size(a)])); %needs fix
        Y = permute(Y,[2 3 1]);
    end
end
