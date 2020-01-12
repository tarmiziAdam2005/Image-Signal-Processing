function out = lrmcADMM(Y,I,P,opts)


%Created on 31/12/2019 by Tarmizi Adam (Email: tarmizi_adam2005@yahoo.com).
% This code solves the noisy matrix completion problem with ADMM. The original
% problem is the following:
%
%       minimize_X  F(X) = 0.5*||Y_omega - X_omega||^2_2 + lambda||X||_*
%
% We re-write the problem as the follows(used in this code)
%
%    minimize_X F(X) = 0.5*||M_omega||^2_2 + lambda||X||_*
%               subject to,  M_omega + X_omega = Y_omega,
%
% and solve using ADMM.
%
%          Input:   Y: The noisy and corrupted matrix (observation)
%                   I: Original matrix
%                   P: The mask, location of zero pixels/entries.
%                opts: Options, see "lrmcADMM_Demo" file.
%
%          Output:  Xk: Estimated low rank matrix.


lam = opts.lam; % Regularization parameter
Nit = opts.Nit; % Number of iteration for algorithm termination
tol = opts.tol; % Tolerance for algorithm stopping criteria.

relError = zeros(Nit,1);

Xk = zeros(size(I)); % Initialize Xk
mu = zeros(size(I)); % Lagrange multiplier
M = Xk;              % Initialize M (see objective function being solved)

beta = 0.5;  %Parameter related to the augmented Lagrange term.


%% ***** Main Loop *****

for k =1:Nit
    
    X_old = Xk;
    
    %***** X Subproblem ****
    
    V = Xk - (P.*M + P.*Xk - Y - mu/beta);
    
    Xk = svt(V,lam/beta);
    
    %***** M subproblem *****
    RHS = beta*Y - beta*(P.*Xk) + mu;
    
    M = RHS./(beta + 1);
    
    %***** Update Lagrange multipliers *****
    
    mu = mu - beta*(P.*M - Y + P.*Xk);
    
     Err = Xk -  X_old;
    relError(k) = norm(Err,'fro')/norm(Xk,'fro');
    
    if relError(k) < tol
        break;
    end 
         
end

%% ***** End Main Loop *****

out.sol = Xk;
out.err = relError(1:k);


end

function Z = shrink(X,r)  %Shrinkage operator
    Z = sign(X).*max(abs(X)- r,0);
end

function Z = svt(X, r) %Singular value thresholding

    [U, S, V] = svd(X);
    s = shrink(S,r);
    
    Z = U*s*V';
end