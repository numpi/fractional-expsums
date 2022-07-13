%
% Solution of a fractional differential equation over [0, 1]^d, with d = 3.
%
% The equation is Lu = f, with L the 3D-Laplace operator, and 
%
%   f(x,y,z) = sin(x)cos(y)exp(z);
%
% The solution is computed exploiting the CP structure of the right hand
% side, directly in CP format using exponential sums. 
%
% The discretization is chosen by uniformly subdividing [0, 1] with n
% points, and taking the tensor product; for small n, the accuracy is
% computed by comparing with the solution computed by a direct method. 

alpha = .5;
d = 3;

nmax = 512;

% [N, tdense, tn30, resn30, tn100, resn100, tn200, resn200, tn350, resn350]
data = zeros(0, 10);

for nn = [ 128, 256, 512, 1024, 2048, 4096 ]

    newdata = zeros(1, 10);

    newdata(1) = nn;

    n = nn * ones(1, 3);
    
    ff = { @(x) sin(x), @(y) cos(y), @(z) exp(z) };
    f = @(x) ff{1}(x{1}) .* ff{2}(x{2}) .* ff{3}(x{3});
    
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

    if max(n) <= nmax

        [xx{:}] = ndgrid(x{:});
        F = f(xx);
        
        tdense = tic;
        
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
        
        tdense = toc(tdense);
    else
        tdense = inf;
    end

    newdata(2) = tdense;    
    fprintf('N = %d, tdense = %f\n', n(1), tdense);
    
    % Exponential sums low-rank
    FU = cell(1, d);
    for j = 1 : d
        FU{j} = ff{j}(x{j});
        FU{j} = FU{j}(:);
    end
    
    res = [];
    Ns = [];
    lrtimes = [];
    
    lambdamin = min(abs(eig(A{1}))) * d;

    column = 3;
    
    for N = [ 30, 100, 200, 350 ]
        Ns = [Ns, N];
    
        tlr = tic;
        [a, b] = expsum_coeffs(alpha, N, lambdamin);
        
        FT2U = cell(1, d);
        for j = 1 : d
            FT2U{j} = zeros(n(j)-2, 0);
        end
    
        for i = 1 : length(a)
            for j = 1 : d
                E = expm(b(i) * A{j}) * FU{j};
                if j == 1 
                    E = a(i) * E;
                end
                
                FT2U{j} = [ FT2U{j}, E ];
            end
        end
    
        tlr = toc(tlr);
        
        % Reconstruct the solution
        if max(n) <= nmax
            FT2 = 0 * F;
            for i = 1 : length(a)
                Fj = kron(FT2U{3}(:,i), kron(FT2U{2}(:,i), FT2U{1}(:,i)));
                Fj = reshape(Fj, n-2);
                FT2 = FT2 + Fj;
            end
    
            reslr = norm(FT2(:) - FT(:)) / norm(FT(:));
        else
            reslr = inf;
        end

        res = [ res, reslr ];

        newdata(column) = tlr; 
        column = column + 1;
        newdata(column) = reslr;
        column = column + 1;
        
        fprintf('N = %d, res = %e, time = %f\n', N, reslr, tlr);
    end

    data = [ data ; newdata ];
    
    
end

dlmwrite(sprintf('exp_tensor_lr_d=%d.dat', d), data, '\t');
