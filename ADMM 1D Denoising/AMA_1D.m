function out = AMA_1D(f,lam,rho, Nit)
% Created by Tarmizi Adam on 24/5/2017
%This is a simple program for 1D Signal denoising using total variation
% denoising. The iterative algorithm used in this program is the 
% Alternating Minimization Algorithm (AMA) proposed by Paul Tseng in the
% paper titled:
%             "Application of a Splitting Algorithm to Decomposition in 
%             Convex Programming and Variational Inequalities"

% This solver solves the following minimization problem:

%    min_x  lambda/2*||u - f||^2_2 + ||Du||_1

%   Input:   
%         f = The noisy observed signal (1D-signal)
%       lam = Regularization parameter. Controls the smoothness of the 
%             denoised signal.
%       rho = Regularization parameter related to the constraint (ADMM)
%       Nit = Number of iterations to run the ADMM iterative method.
%
%   Output:
%        out.sol         = The denoised signal
%        out.funVal      = Plot of the convergence of our objective
%                          function
%
% Remarks: This code is based on the derivation in the authors ADMM notes
%          in the subsection AMA TV denoising.
%%

f         = f(:);
u         = f;          %Initialize
N         = length(f);
mu        = zeros(N,1); % Lagrange multiplier
v         = mu; %Initialize the v sub-problem
funcVal   = zeros(Nit,1);

[D,DT,DTD] = defDDt(N);

    for k = 1:Nit
        u_old = u;
        
        %% u sub-problem %%
       
        u = f + (1/lam)*DT*mu; %Main difference with ADMM. No need to solve linear system.
        
        %% v sub-problem %%
        x    = D*u - mu/rho;
        v    = shrink(x, 1/rho);
        
       %% Update Lagrange multiplier
        mu   = mu + (v - D*u);
        
        r1   = u - f;
        funcVal(k) = (lam/2)*norm(r1,'fro')^2 + sum(v(:));
    
    end
    
    out.sol = u;
    out.funVal = funcVal(1:k);

end

function [D,DT,DTD] = defDDt(N)
%Create a first order difference matrix D
e = ones(N,1);
B = spdiags([e -e], [1 0], N, N);
B(N,1) =  1;

D = B; 
clear B;
% Create the transpose of D
DT = D'; %Remember that DT = -D, also called the backward difference.
DTD = D'*D;
end

function z = shrink(x,r)
z = sign(x).*max(abs(x)- r,0);
end

