%
% Solution of a fractional differential equation over [0, 1]^d, with d = 3.
%
% The equation is Lu = f, with L the 3D-Laplace operator, and 
%
%   f(x,y,z) = 1 / (x+y+z);
%
% The solution is computed by a direct method diagonalizing the Laplace
% operator exploiting the Kronecker structure, and by exponential sums. 

alpha = 0.4;
d = 3;
n = [ 128, 128, 128 ];

f = @(x) 1 ./ (1 + x{1} + x{2} + x{3});

h = 1 ./ (n - 1);
A = cell(1, d);

for j = 1 : d
    A{j} = spdiags(ones(n(j)-2,1) * [-1 2 -1], -1:1, n(j)-2, n(j)-2) ./ h(j)^2;
end

x = cell(1, d);
for j = 1 : d
    x{j} = linspace(0, 1, n(j));
    x{j} = x{j}(2:end-1);
end

xx = cell(1, d);
[xx{:}] = ndgrid(x{:});

F = f(xx);

% Diagonalization of the A_i
V = cell(1, d); D = cell(1, d);
for j = 1 : d
    [V{j}, D{j}] = eig(full(A{j}));
end

% Apply the transformations to F
FT = F;
for j = 1 : d
    FT = ttimes_dense(V{j}, FT, j, true);
end

ll = cell(1, d);
for j = 1 : d
    ll{j} = diag(D{j});
end
l = cell(1, d);
[l{:}] = ndgrid(ll{:});
DD = zeros(n - 2);
for j = 1 : d
    DD = DD + l{j};
end

FT = (DD.^(-alpha)) .* FT;

for j = 1 : d
    FT = ttimes_dense(V{j}, FT, j);
end

% Exponential sums
res = [];
Ns = [];
bound = [];

lambdamin = min(abs(eig(A{1}))) * d;

for N = 10 : 10 : 250
    Ns = [Ns, N];
    [a, b] = expsum_coeffs(alpha, N, lambdamin);
    
    FT2 = F * 0;

    for i = 1 : length(a)
        E = F;
        for j = 1 : d
            E = ttimes_dense(expm(b(i) * A{j}), E, j);
        end

        FT2 = FT2 + a(i) * E;
    end

    res = [ res, norm(FT2(:) - FT(:)) / norm(FT(:)) ];
    bound = [ bound, exp(-sqrt(2*pi^2*N*alpha/8)) ];
    
    fprintf('N = %d, res = %e\n', N, res(end));
end

bound = 3 * res(1) / bound(1) * bound;

semilogy(Ns, res);
hold on;
semilogy(Ns, bound, 'r--');

dlmwrite(sprintf('exp_tensor_d=%d_n=%d.dat', d, n(1)), [ Ns(:), res(:), bound(:) ], '\t');
