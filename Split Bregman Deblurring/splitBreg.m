function out = splitBreg(g,Img,H,lam,mu,Nit,tol)
%%
% The program solves the following core objective function
% using Split Bregman method
%
%   min_f   mu/2*||Hf - g||^2 + ||Df||_1
%
%   where ||DF||_1 is |D_h*f| + |D_v*f|
%  References:
%  [1] Goldstein, Tom, and Stanley Osher. "The split Bregman method for L1-regularized problems." 
% SIAM Journal on Imaging Sciences 2.2 (2009): 323-343.
%%

[row,col] = size(g);
f         = g;
x         = zeros(row,col);
y         = zeros(row,col);
bx        = zeros(row,col);
by        = zeros(row,col);


relError     = zeros(Nit,1); % Compute error relative to the previous iteration.
relErrorImg  = relError;     %Compute error relative to the original image
psnrGain     = relError;     % PSNR improvement every iteration
funcVal      = relError;     %Function value at each iteration


eigHtH   = psf2otf(H,[row,col]);
Htg     = imfilter(g, H, 'circular');

[D,Dt]      = defDDt(); %Declare forward finite difference operators

%=================== Main algorithm starts here ======================
nabla   = fspecial('laplacian',0);  
lhs     = mu*conj(eigHtH).*eigHtH - lam*psf2otf(nabla, [row col]);

    for k=1:Nit
        
        f_old = f;
        
        rhs = mu*Htg + lam*Dt(x - bx, y - by); 
        
        f = fft2(rhs)./lhs;
        f = real(ifft2(f));
        
        [Dfx, Dfy]  = D(f);
        %Dfx = Dx(f);
        %Dfy = Dy(f);
        
        x = shrink(Dfx + bx,1/lam);
        y = shrink(Dfy + by,1/lam);
        
        % update bregman parameters
        bx = bx + Dfx - x;
        by = by + Dfy - y;
        
        relError(k)    = norm(f - f_old,'fro')/norm(f, 'fro');
        relErrorImg(k) = norm(Img - f,'fro')/norm(Img,'fro');
        psnrGain(k)    = psnr_fun(f,Img);
        
        r1          = imfilter(f, H, 'circular')-g;
        funcVal(k)  = (lam/2)*norm(r1,'fro')^2 + sum(Dfx(:)+Dfy(:));
          
       % if relError(k) < tol
            %break
        %end 
    end

%======================= Results ==============================
out.psnrf                = psnr_fun(f, Img);
out.ssimf                = ssim_index(f,Img);
out.sol                 = f;                %Deblurred image
out.relativeError       = relError(1:k);
out.relativeErrorImg    = relErrorImg(1:k);
out.psnrGain            = psnrGain(1:k)
out.functionValue       = funcVal(1:k);
end

function [D,Dt] = defDDt()
D  = @(U) ForwardDiff(U);
Dt = @(X,Y) Dive(X,Y);
end

function [Dux,Duy] = ForwardDiff(U)
 Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
 Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
end

function DxyT = Dive(X,Y)

[rows, cols] = size(X);
dxt = zeros(rows, cols);
dxt(:,1:cols-1) = X(:,1:cols-1)-X(:,2:cols);
dxt(:,cols) = X(:,cols)-X(:,1);

dyt = zeros(rows, cols);
dyt(1:rows-1,:) = Y(1:rows-1,:)-Y(2:rows,:);
dyt(rows,:) = Y(rows,:)-Y(1,:);

DxyT = dxt + dyt;

end

function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
end



