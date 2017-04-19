function [f, relchg] = ALMTV(g,H,lam,rho,Nit,tol)
%%
%==========Anisotropic Total variation using Augmented Lagrangian====
% ALMTV: Augmented Lagrangian Method Total Variation
% The original paper [1] Takes into account image and video restorations.
% This code is a little modification for just gray scale image restoration.
% This code does not support video restoration like [1]. Only the
% mathematical theory used in [1] to code this program.

%inputs:
%       g       : Observed blurred(and possibly noisy) image
%       H       : Point spread function/Blurring kernel (A linear operator)
%       lam     : regularization parameter
%       rho     : regularization parameter of the Augmented Lagrangian form
%                 of the main problem.
%       Nit     : Number of iterations
%       tol     : Error tolerance for stopping criteria

%============== References====================
% See the following papers 

% [1] Chan, Stanley H., Ramsin Khoshabeh, Kristofor B. Gibson, Philip E. Gill, 
%     and Truong Q. Nguyen. "An augmented Lagrangian method for total variation 
%     video restoration." Image Processing, IEEE Transactions on 20, no. 11 
%     (2011): 3097-3111.

%  [2] Chan, Stanley H., Ramsin Khoshabeh, Kristofor B. Gibson, Philip E. Gill, and 
%      Truong Q. Nguyen. "An augmented Lagrangian method for video restoration." In 
%      Acoustics, Speech and Signal Processing (ICASSP), 2011 IEEE International 
%      Conference on, pp. 941-944. IEEE, 2011.

% The program solves the following core objective function

%   min_f   lam/2*||Hf - g||^2 + ||Df||_1

%%

[row,col] = size(g);
f       = g;

u1      = zeros(row,col); %Initialize intermediat variables for u subproblem
u2      = zeros(row,col); %     "       "           "       for u subproblem

y1      = zeros(row,col); %Initialize Lagrange Multipliers
y2      = zeros(row,col); %   "       Lagrange Multipliers

eigHtH  = abs(fft2(H,row,col)).^2; %eigen value for HtH
eigDtD  = abs(fft2([1 -1], row, col)).^2 + abs(fft2([1 -1]', row, col)).^2 % eigen value ofDtD
Htg     = imfilter(g, H, 'circular');

[D,Dt]      = defDDt(); %Declare forward finite difference operators
[Df1, Df2] = D(f);

rnorm = sqrt(norm(Df1(:))^2 + norm(Df2(:))^2);

    for k=1:Nit
    
        %Solving the f subproblem
        f_old   = f;
        rhs     = lam*Htg + Dt(u1 - (1/rho)*y1, u2 - (1/rho)*y2);
        eigA    = lam*eigHtH + rho*eigDtD;
        f       = fft2(rhs)./eigA;
        f       = real(ifft2(f));
        
    
        %Solving the u subproblem
        [Df1, Df2]  = D(f);
        v1          = Df1 + (1/rho)*y1;
        v2          = Df2 + (1/rho)*y2;
    
        u1          = shrink(v1,1/lam);
        u2          = shrink(v2,1/lam);
    
        %Update y, the Lagrange multipliers
        y1          = y1 - rho*(u1 - Df1);
        y2          = y2 - rho*(u2 - Df2);
        
        rnorm_old  = rnorm;
        rnorm      = sqrt(norm(Df1(:)-u1(:), 'fro')^2 + norm(Df2(:)-u2(:), 'fro')^2);
        
        relchg = norm(f(:)-f_old(:))/norm(f_old(:));
        relchg(Nit) = relchg;
        
        if relchg < tol
            break
        end
    
    end
end


function [D,Dt] = defDDt()
D  = @(U) ForwardDiff(U);
Dt = @(X,Y) Dive(X,Y);
end

function [Dux,Duy] = ForwardDiff(U)
 Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
 Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
end

function DtXY = Dive(X,Y)
  % Transpose of the forward finite difference operator
  % Divergence
  DtXY = [X(:,end) - X(:, 1), -diff(X,1,2)];
  DtXY = DtXY + [Y(end,:) - Y(1, :); -diff(Y,1,1)];   
end

function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
end