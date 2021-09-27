function [ADP] = CSI2ADP_theta_N(H,N,Ncc,Nt,Nc)
%%
V = zeros(Nt,N);
theta = linspace(0,pi,N);
for i = 1 : Nt
    for j = 1 : N
        V(i,j) = exp(-1i * pi * (i-1) * cos(theta(j)));
    end 
end 

% Constructing F
F = zeros(Nc,Ncc);

for i = 1 : Nc
    for j = 1 : Ncc
        F(i,j) = exp(1i * 2 * pi * i * j / Ncc );
    end 
end 

ADP = V' * (H) * F ;
end