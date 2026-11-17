function [score, perm] = fms_score(Ahat, Atrue)
%FMS_SCORE  Factor match score between an estimated and a true set of CP factors.
%   score = mean over components (after optimal matching) of prod_n |cos(ahat_n, a_n)|.
%   1 = perfect recovery up to permutation, sign and scaling.
N = numel(Atrue); Rh = size(Ahat{1}, 2); Rt = size(Atrue{1}, 2);
S = ones(Rh, Rt);                            % works for Rh ~= Rt (over/under-factoring)
for n = 1:N
    H = Ahat{n} ./ max(sqrt(sum(Ahat{n}.^2, 1)), eps);
    T = Atrue{n} ./ max(sqrt(sum(Atrue{n}.^2, 1)), eps);
    S = S .* abs(H' * T);
end
M = matchpairs(-S, 1e6);                     % maximise total congruence
perm = zeros(1, Rt); perm(M(:, 2)) = M(:, 1);
score = mean(S(sub2ind([Rh Rt], M(:, 1), M(:, 2))));
end
