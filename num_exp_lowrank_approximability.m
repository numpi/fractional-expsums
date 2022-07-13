%
% Testing low-rank approximability properties
% 

if ~exist('cp_als')
	fprintf('Please install the Tensor Toolbox\n');
	return;
end

if ~exist('cpd')
	fprintf('Please install TensorLab 3.0\n');
	return;
end

if ~exist('tt_tensor')
	fprintf('Please install the TT-Toolbox\n');
	return;
end

n = 128;
a = 0.5;
A = 2*eye(n) - diag(ones(n-1,1), -1) - diag(ones(n-1,1), 1);
A = A / min(eig(A));

c = kron(randn(n,1), kron(randn(n,1), randn(n,1)));
C = reshape(c, n, n, n);

X = lyapnd({ A, A, A }, C, a);

% Plot the norms of the best low-rank approximant
XX = tensor(X);
maxn = 40;
res = zeros(1, maxn);
for j = 1 : maxn
    M = cp_als(XX, j, 'init', 'nvecs', 'tol', 1e-12, 'maxiter', 100);
    M2 = cpdgen(cpd(double(XX), j));
    res(j) = min(norm(full(M) - full(XX)), frob(M2 - double(XX)));
end
res = res / norm(XX);

% Normalize the decrease
for j = 2 : length(res)
    res(j) = min(res(j), res(j-1));
end

% Do the same for the HOSvd
resml = zeros(1, maxn);
for j = 1 : maxn
    M = hosvd(XX, eps, 'ranks', [j j j]);
    resml(j) = norm(full(XX) - full(M));
end
resml = resml / norm(XX);

% And for TT as well
restt = zeros(1, maxn);
TX = tt_tensor(X);
for j = 1 : maxn
    M = round(TX, eps, j);
    restt(j) = norm(X(:) - full(M), 'fro') / norm(X, 'fro');
end

% Compute the bound?
d = pi * a / 8;
nn = 1 : maxn;
semilogy(1 : maxn, res);
hold on;
f1 = exp(-sqrt(2*pi*d*(1:maxn)));
plot(1 : maxn, f1, 'r--');
%f2 = exp(-(2*pi*d*(1:maxn)));
%plot(1 : maxn, f2, 'g--');
semilogy(1:maxn, resml, 'g');
semilogy(1:maxn, restt, 'k');
dlmwrite('lowrank.dat', [ nn(:), res(:), f1(:), resml(:) restt(:) ], '\t');

