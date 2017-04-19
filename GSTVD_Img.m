function [x, err] = GSTVD_Img(y, K, lam, Nit)
% Created by Tarmizi Adam 2/09/2015. A program to do Total variation
% Denoising (TVD)

%   Output:
%           x : Denoised image (Display this)
%          err: Error at each iteration (Plot this to see convergence)
%   Input:
%           y  : Noisy Image (Observed Image)
%           K  : Group size
%          lam : regularization parameter (lambda)
%          Nit : Number of iteration to stop the Algorithm

% The codes here follows closely several papers as 
% references:

%   1) M. Figueiredo, J. B. Dias, J. P. Oliveira, R. D. Nowak et al., 
%      “On total variation denoising: A new majorization-minimization 
%      algorithm and an experimental comparisonwith wavalet denoising,” 
%      in IEEE International Conference on Image Processing. 
%      IEEE, 2006, pp. 2633–2636.

%   2) Micchelli, C. A., Shen, L., and Xu, Yuesheng. 
%      "Proximity algorithms for image models: Denoising"               
%      Inverse Problems (27).1-29 (2011)

%   3) Tutorial and codes from: I. Selesnick, 
%      “Total variation denoising (an mm algorithm)

%   4) Liu, Jun, Ting-Zhu Huang, Ivan W. Selesnick, Xiao-Guang Lv, 
%      and Po-Yu Chen. "Image restoration using total variation with 
%      overlapping group sparsity." Information Sciences 295 (2015): 232-246.

%   5) Selesnick, Ivan W., and Po-Yu Chen. "Total variation denoising 
%      with overlapping group sparsity." 
%      In Acoustics, Speech and Signal Processing (ICASSP), 2013 IEEE 
%      International Conference on, pp. 5696-5700. IEEE, 2013.

y = double(y);
y = y(:); 

n = length(y);

[D, DT, DDT ] = DiffOper(sqrt(n)); %pre-compute some matrices, including 
                                 % our differential operator D (hor and
                                 % ver)
h = ones(K,K);
%h= h(:);
x = y;
Dx = D*x;
Dy = D*y;

dim = length(Dx);
err = zeros(Nit,1);

for k = 1:Nit
    
    xu = x;
    
    r = sqrt(conv2(abs(Dx).^2, h,'same'));
    v = conv2(1./r, h, 'same');  
    
    F = 1/lam * spdiags(1./v,0,dim,dim) + DDT;  %1/lam*diag(Dx) + DDT
    
    %******Can use backlash*****************
    z = F\Dy;                             %*  
    x = y - DT*z; %update x               %*
    %***************************************
    
    %*******Can use also congugate gradinet to solve the linear system****
    %z = cgs(F,Dy,[],100); %solve linear system for z, F*z = Dy         %*
    %x = y - DT*z; %update x                                            %*
    %*********************************************************************
    
    e = norm(xu-x)/norm(x); %convergence error
    err(k) = e;
    Dx = D*x; 
    
end

x = reshape(x,256,256);
end

function [D,DT,DDT] = DiffOper(N)
e = ones(N,1);
B = spdiags([e -e], [1 0], N, N);
B(N,1) =  1;

Dv = kron(speye(N),B); 
Dh = kron(B,speye(N));

D = [ Dv ; Dh ]; %combine vertical and horizontal
                 % difference matrix in one big matrix D. refer 2)
DT = D';
DDT = D*D';
end