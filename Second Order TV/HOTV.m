function out = HOTV(f, Img ,K ,opts)
%Copyright, created by Tarmizi Adam on 31/07/2016
%This codes solves the second order (2nd) Total variation model problem
%    
%     min_u lam/2*||Ku - f||_2^2 + omega*TV(DDu)_iso,
%
%using ADMM where, DDu is the 2nd order difference operator and TV(.)_iso is the 
%isotropic total variation functional. This model is reffered as the Lysaker–Lundervold–Tai (LLT) model for image
%restoration. Please refer to the paper:

%    M. Lysaker, A. Lundervold, and X.-C. Tai, Noise removal using fourth-order partial differential
%    equations with applications to medical magnetic resonance images in space and time, IEEE Trans.
%    Image Process., 12 (2003), pp. 1579–1590.

% To follow exactly the LLT model, set 'lam = 1' and play around with 'omega'.
%Run this code through the demo .m 
%code HOTV_demo.m

% Any suggestions, errors or bugs feel free to contact me at:
%       tarmizi_adam2005@yahoo.com

[row, col] = size(f);
u               = f;

Nit             = opts.Nit;
tol             = opts.tol;
lam             = opts.lam; % The regularization parameter.
beta            = opts.beta;
omega           = opts.omega;
alpha           = 0.07;

%v1        = zeros(row,col); %Initialize intermediate variables for v subproblem
%v2        = zeros(row,col); %     "       "           "       for v subproblem

mu1        = zeros(row,col); %Initialize Lagrange Multipliers
mu2        = zeros(row,col); %   "       Lagrange Multipliers
mu3        = zeros(row,col);
mu4        = zeros(row,col);

relError   = zeros(Nit,1); % Compute error relative to the previous iteration.
funcVal    = relError;
psnrGain   = relError; 
v1         = zeros(row, col);
v2         = v1;
v3         = v1;
v4         = v1;


eigK            = psf2otf(K,[row col]); %In the fourier domain
eigKtK          = abs(eigK).^2;
eigDDtDD = abs(psf2otf([1 -2 1],[row col])).^2 + abs(psf2otf([1 -1;-1 1],[row col])).^2 ...
            + abs(psf2otf([1 -1;-1 1],[row col])).^2 + abs(psf2otf([1;-2;1],[row col])).^2;


[DD,DDt] = defDDt2;

[Duxx,Duxy,Duyx,Duyy] = DD(u);

curNorm = sqrt(norm(Duxx(:) - v1(:),'fro'))^2 + sqrt(norm(Duxy(:) - v2(:),'fro'))^2 + sqrt(norm(Duyx(:) - v3(:),'fro'))^2 + sqrt(norm(Duyy(:) - v4(:),'fro'))^2;

Ktf = imfilter(f,K,'circular');

tg = tic;

    for k = 1:Nit
        
        x1      = Duxx + mu1/beta;
        x2      = Duxy + mu2/beta;
        x3      = Duyx + mu3/beta;
        x4      = Duyy + mu4/beta;
        
        [v1, v2, v3, v4] = isoShrink(x1,x2,x3,x4,omega/beta);
    
        u_old   = u;
        rhs     = lam*Ktf + beta*DDt(v1 - mu1/beta, v2 - mu2/beta, v3 - mu3/beta, v4 - mu4/beta);
        lhs     = lam*eigKtK + beta*eigDDtDD;
        
        u       = fft2(rhs)./lhs;
        u       = real(ifft2(u));
        
        [Duxx,Duxy,Duyx,Duyy] = DD(u);
          
        mu1     = mu1 - beta*(v1 - Duxx);
        mu2     = mu2 - beta*(v2 - Duxy);
        mu3     = mu3 - beta*(v3 - Duyx);
        mu4     = mu4 - beta*(v4 - Duyy);
         
        relError(k)    = norm(u - u_old,'fro')/norm(u, 'fro');
        psnrGain(k)    = psnr_fun(u,Img);
        r1          = imfilter(u, K, 'circular')-f;
        funcVal(k)  = (lam/2)*norm(r1,'fro')^2 + omega*sqrt(sum(Duxx(:).^2 + Duxy(:).^2 + Duyx(:).^2 + Duyy(:).^2));
        
        if relError(k) < tol
            break;
        end
        
        normOld = curNorm;
        curNorm = sqrt(norm(Duxx(:) - v1(:),'fro'))^2 + sqrt(norm(Duxy(:) - v2(:),'fro'))^2 + sqrt(norm(Duyx(:) - v3(:),'fro'))^2 + sqrt(norm(Duyy(:) - v4(:),'fro'))^2;
        
        if curNorm > alpha*normOld
            beta = 0.95*beta;
        end  
    end
    
tg = toc(tg);
    
    out.sol                 = u;
    out.relativeError       = relError(1:k);
    out.functionValue       = funcVal(1:k);
    out.cpuTime             = tg;
    out.psnrGain            = psnrGain(1:k);
    out.psnrRes             = psnr_fun(u, Img);
    out.ssimRes             = ssim_index(u, Img);
    out.OverallItration     = size(out.functionValue,1); %No of itr to converge

end


 function [DD,DDt] = defDDt2
        % defines finite difference operator D
        % and its transpose operator
        DD  = @(U) ForwardD2(U);
        DDt = @(Duxx,Duxy,Duyx,Duyy) Dive2(Duxx,Duxy,Duyx,Duyy);
 end
    
  function [Duxx Duxy Duyx Duyy] = ForwardD2(U)
        %
        Duxx = [U(:,end) - 2*U(:,1) + U(:,2), diff(U,2,2), U(:,end-1) - 2*U(:,end) + U(:,1)];
        Duyy = [U(end,:) - 2*U(1,:) + U(2,:); diff(U,2,1); U(end-1,:) - 2*U(end,:) + U(1,:)];
        %
        Aforward = U(1:end-1, 1:end-1) - U(  2:end,1:end-1) - U(1:end-1,2:end) + U(2:end,2:end);
        Bforward = U(    end, 1:end-1) - U(      1,1:end-1) - U(    end,2:end) + U(    1,2:end);
        Cforward = U(1:end-1,     end) - U(1:end-1,      1) - U(  2:end,  end) + U(2:end,    1);
        Dforward = U(    end,     end) - U(      1,    end) - U(    end,    1) + U(    1,    1);
        % 
        Eforward = [Aforward ; Bforward]; Fforward = [Cforward ; Dforward];
        Duxy = [Eforward, Fforward]; Duyx = Duxy;
        %
  end
    
   function Dt2XY = Dive2(Duxx,Duxy,Duyx,Duyy)
        %
        Dt2XY =         [Duxx(:,end) - 2*Duxx(:,1) + Duxx(:,2), diff(Duxx,2,2), Duxx(:,end-1) - 2*Duxx(:,end) + Duxx(:,1)]; % xx
        Dt2XY = Dt2XY + [Duyy(end,:) - 2*Duyy(1,:) + Duyy(2,:); diff(Duyy,2,1); Duyy(end-1,:) - 2*Duyy(end,:) + Duyy(1,:)]; % yy
        %
        Axy = Duxy(1    ,    1) - Duxy(      1,    end) - Duxy(    end,    1) + Duxy(    end,    end);
        Bxy = Duxy(1    ,2:end) - Duxy(      1,1:end-1) - Duxy(    end,2:end) + Duxy(    end,1:end-1);
        Cxy = Duxy(2:end,    1) - Duxy(1:end-1,      1) - Duxy(  2:end,  end) + Duxy(1:end-1,    end);
        Dxy = Duxy(2:end,2:end) - Duxy(  2:end,1:end-1) - Duxy(1:end-1,2:end) + Duxy(1:end-1,1:end-1);
        Exy = [Axy, Bxy]; Fxy = [Cxy, Dxy];
        %
        Dt2XY = Dt2XY + [Exy; Fxy];
        Dt2XY = Dt2XY + [Exy; Fxy];
   end
    
   function [v1, v2, v3, v4] = isoShrink(u1,u2,u3,u4,r)
    u = sqrt(u1.^2 + u2.^2 + u3.^2 + u4.^2);
    u(u==0) = 1;
    u = max(u - r,0)./u;
    
    v1 = u1.*u;
    v2 = u2.*u;
    v3 = u3.*u;
    v4 = u4.*u;
   end


