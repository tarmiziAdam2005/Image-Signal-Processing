function out = RPCA_ADMM(M,opts)

% Created by Tarmizi Adam on 2/06/2017
%This is a simple program for Robust Principal Component Analysis (RPCA)
%for Image inpainting (restoring missing pixels values). The iterative algorithm used in this program is the 
% Alternating Direction Methods of Multiplier (ADMM) which is a splitting
% version of the classical Augmented Lagrangian Method (ALM).

% This solver solves the following minimization problem:

%    min_L&M  ||L||_* + lambda*||S||_1
%    s.t  L + S = M

%   Input:   
%         M = Noisy observation
%       opts = related parameters for ADMM algorithm etc. 
%  
%   Output:
%        out.solL        = Low rank matrix (in our context, inpainted image)
%        out.solS        = Sparse matrix ( in our context, separated missing pixels)

% Any suggestion,comments, errors regarding this code, do not hesitate to email me at: 
%
%           tarmizi_adam2005@yahoo.com
%
%%

L = zeros(size(M)); % Low rank matrix
S = zeros(size(M));  % Sparse matrix
Mu = zeros(size(M));  % Lagrange multipliers

rho = opts.rho; % contraint regularization parameter
lam = opts.lam; %Regularization parameter, > 0

relError        = zeros(opts.Nit,1);

%====================== Main ADMM algorithm loop =========================
for i = 1:opts.Nit
    
    % L subproblem
    X = M + Mu./rho;
    L = svt(X-S , 1/rho); %Solve for L, by singular value thresholding
    
    % S subproblem
    S = shrink(X-L, lam/rho); %Solve for S by soft thresholding
    
    Mu = Mu - rho*(L+S-M); %Update our Lagrange multipliers
    
    E_r = L+S-M;
    
    relError(i)    = norm(E_r,'fro')/norm(M, 'fro');
    
     if relError(i) < opts.tol
            break;
     end

end

out.solL                = L;
out.solS                = S;
out.relativeError       = relError(1:i);

end

function Z = shrink(X,r)  %Shrinkage operator
    Z = sign(X).*max(abs(X)- r,0);
end

function Z = svt(X, r) %Singular value thresholding

    [U, S, V] = svd(X,'econ');
    Z = U*shrink(S,r)*V';
end