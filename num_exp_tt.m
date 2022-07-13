%
% Experiment with the TT format
%

if ~exist('tt_tensor')
	fprintf('Please install the TT-Toolbox\n');
	return;
end

dd = [ 2, 3, 4, 6, 10, 15, 20 ];

% [ d, time, err, rk ]
data = zeros(length(dd), 4);

for l = 1 : length(dd)
    d = dd(l);

    alpha = .5;
    n = 128;
    ttol = 1e-8;
    
    h = 1 ./ (n - 1);
    A = cell(1, d);
    
    for j = 1 : d
        A{j} = spdiags(ones(n-2,1) * [-1 2 -1], -1:1, n-2, n-2) ./ h^2;
    end
    
    % Compute lambdamin
    lambdamin = 0;
    for j = 1 : d
        lambdamin = lambdamin + min(eig(full(A{j})));
    end
    
    x = cell(1, d);
    for j = 1 : d
        x{j} = linspace(0, 1, n)';
        x{j} = x{j}(2:end-1);
    end
    
    % For simplicity we consider 1 ./ (1 + x{1} + ... + x{d});
    f = @(ind) 1 ./ (1 + sum(ind ./ (n-1)));

    RHS = amen_cross((n-2) * ones(1, d), f, ttol);
    
    N = 200;
    [a, b] = expsum_coeffs(alpha, N, lambdamin);
    
    X = tt_zeros(size(RHS));
    
    % Compute the solution
    t = tic;
    for k = 1 : length(a)
        EE = RHS;
        for j = 1 : d
            % Multiply EE in the mode j by expm(A{j} * b(k));
            EE = ttm(EE, j, expm(b(k) * full(A{j})));
        end
    
        X = round(X + a(k) * EE, ttol);
    end
    t = toc(t);
    
    fprintf('n = %d, d = %d, time = %f\n', n, d, t);

    data(l, 1) = d;
    data(l, 2) = t;
    data(l, 3) = nan;
    data(l, 4) = max(rank(X));
    
    if d <= 4
        FA = cell(1, d);
        for j = 1 : d
            FA{j} = full(A{j}); 
        end
        xx = cell(1, d);
        [xx{:}] = ndgrid(x{:});

        FRHS = ones(ones(1, d) * (n-2));
        for j = 1 : d
            FRHS = FRHS + xx{j};
        end
        clear xx;
        FRHS = 1 ./ FRHS;

        % norm(FRHS(:) - full(RHS)) / norm(FRHS(:))

        % XD = lyapnd(FA, -reshape(full(RHS), (n-2)*ones(1,d)), alpha);
        XD = lyapnd(FA, -FRHS, alpha);
        res = norm(XD(:) - full(X)) / norm(XD(:));
        fprintf('    N_exps = %d, res = %e\n', N, res);

        data(l, 3) = res;
    end
end

dlmwrite('exp_tt.dat', data, '\t');

