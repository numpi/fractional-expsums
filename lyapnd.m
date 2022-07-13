function X = lyapnd(A, C, a)
%LYAPND Solve the equation \sum X \times_i A{i} + C = 0 by diagonalization.
%
% If C is a tensor with d + 1 indices, one more than the number of 
% coefficients A{i}, then the equation is solved for all slices of C. 
%
% If the parameter a is specified, then the fractional power of the
% operator is considered instead. 

d = length(A);
% n = arrayfun(@(i) size(A{i}, 1), 1 : d);

% Diagonalize the coefficients
Q = cell(1, d);
D = cell(1, d+1);
for k = d : -1 : 1
    [Q{k}, D{k}] = eig(A{k}, 'vector');
    if issymmetric(A{k})
        C = ttimes_dense(Q{k}', C, k);
        % C = tensorprod(Q{k}', C, 2, d);
    else
        C = ttimes_dense(Q{k}, C, k, true);
    end
end

D{d+1} = zeros(size(C, d+1), 1);

M = D{1};
for k = 2 : d + 1
    M = M + D{k}.';
    M=M(:);
end
M = M.^a;

X = -C ./ reshape(M, size(C));

for k = d : -1 : 1
    X = ttimes_dense(Q{k}, X, k);
    % X = tensorprod(Q{k}, X, 2, d);
end

end