function out = lpALM(y, Img, H, mu, rho, Nit,p, tol)

%Created by Tarmizi on 11/02/ 2016
% Non-convex lp total variation deblurring....
% the total variation deblurring in this code uses non-convex penalty
% function which is the lp norm where 0 < p < 1.
% This code minimizes the objective function:
%       
%       1/2 ||y - Hx||_2^2 + mu*||Dx||_p^p
%
% In this code, we apply the Augmented Lagrangian Method (ALM) with 
% ADMM to handel the subproblems and use the
% IRL1 algorithm to approximately solve the lp penalty.

% If you happen to use this code for any research work, feel free to email
% me just to let me know, at tarmizi_adam2005@yahoo.com
% OR if you find any bugs also feel free to email me.

%If you use this code, use it at your own risk. No guarantees are provided
%for its usage.

[row,col] = size(y);
x         = y;

v1        = zeros(row,col); %Initialize intermediate variables for v subproblem
v2        = zeros(row,col); %     "       "           "       for v subproblem

y1        = zeros(row,col); %Initialize Lagrange Multipliers
y2        = zeros(row,col); %   "       Lagrange Multipliers

relError     = zeros(Nit,1); % Compute error relative to the previous iteration.
relErrorImg  = relError;     %Compute error relative to the original image
psnrGain     = relError;     % PSNR improvement every iteration
funcVal      = relError;     %Function value at each iteration

eigHtH  = abs(fft2(H,row,col)).^2; %eigen value for HtH
eigDtD  = abs(fft2([1 -1], row, col)).^2 + abs(fft2([1 -1]', row, col)).^2;

Hty     = imfilter(y, H, 'circular');

[D,Dt]      = defDDt(); %Declare forward finite difference operators
[Dx1, Dx2] = D(x);

 for k=1:Nit
     
     x_old = x;
     rhs = Hty + rho*Dt(v1 - (1/rho)*y1, v2 - (1/rho)*y2);
     eigA    = eigHtH + rho*eigDtD;
     x       = fft2(rhs)./eigA;
     x       = real(ifft2(x));
     
     [Dx1, Dx2]  = D(x);
     
     w1 = mu*p./(abs(Dx1)+ 0.0001).^(1-(p)); %IRL1 Weight update
     w2 = mu*p./(abs(Dx2) + 0.0001).^(1-(p));% IRL1 Weight update
     
     u1          = Dx1 + y1;
     u2          = Dx2 + y2;
     
     v1          = shrink(u1,0.5*w1.*mu/rho)- y1; 
     v2          = shrink(u2,0.5*w2.*mu/rho) - y2;
     
     
     y1          =  Dx1 - v1;
     y2          =  Dx2 -v2;
     
     relError(k)    = norm(x - x_old,'fro')/norm(x, 'fro');
      r1          = imfilter(x, H, 'circular')-y;
        funcVal(k)  = (1/2)*norm(r1,'fro')^2 + mu*sum(Dx1(:)+Dx2(:));
     
      if relError(k) < tol
            break
      end 
    
 end

out.psnrf                = psnr_fun(x, Img);
out.ssimf                = ssim_index(x,Img);
out.sol                 = x;                %Deblurred image
out.functionValue       = funcVal(1:k);
out.relativeError       = relError(1:k);
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
  % is the divergence fo the forward finite difference operator
  DtXY = [X(:,end) - X(:, 1), -diff(X,1,2)];
  DtXY = DtXY + [Y(end,:) - Y(1, :); -diff(Y,1,1)];   
end

function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
end