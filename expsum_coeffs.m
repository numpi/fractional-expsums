function [a, b] = expsum_coeffs(alpha, N, lambdamin) %Theorem 2.7

if ~exist('lambdamin', 'var')
    lambdamin = 1;
end

d = alpha * pi / 8;
beta=cos(2*d/alpha);

% Phase 1: Find Nm, Np that correspond to the given N
fNp = @(Nm) (2*pi*d).^((alpha - 1)/2) / (beta.^alpha) * ...
    Nm.^((alpha+1)/2);

Nm = fsolve(@(Nm) Nm + fNp(Nm) + 1 - N, N, optimoptions('fsolve','Display','off'));
Np = ceil(fNp(Nm));
Nm = N - Np - 1;

h = sqrt(2*pi*d / Nm);

% Phase 2: Compute the coefficients of the exponential sums
a = []; b = [];

% T=h*(exp(-(log(2)^(1/alpha))*xi)/(2));
a = [ a, h/2 ];
b = [ b, -(log(2)^(1/alpha)) ];

for k=1:Np
    p=(1+exp(k*h));
    m=(1+exp(-k*h));
    %T=T+h*(exp((-log(p)^(1/alpha))*xi)/m);
    a = [a, h/m ];
    b = [b, -log(p)^(1/alpha) ];
    
    %T=T+h*(exp((-log(m)^(1/alpha))*xi)/p);
    a = [a, h/p];
    b = [b, -log(m)^(1/alpha)];
end

for k=Np+1:Nm
    p=(1+exp(k*h));
    m=(1+exp(-k*h));
    % T=T+h*(exp((-log(m)^(1/alpha))*xi)/p);
    a = [a, h/p];
    b = [b, -log(m)^(1/alpha)];
end

% T=T./(alpha*gamma(alpha));
a = a ./ (alpha*gamma(alpha));

% Adjust for spectrum in [lambdamin, +infty] instead of [1, +infty]
a = a .* (lambdamin^(-alpha));
b = b ./ lambdamin;
