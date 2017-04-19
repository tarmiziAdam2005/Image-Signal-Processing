function [x, err] = TVD_Img(y, lam, Nit)
% Created by Tarmizi Adam 2/09/2015. A program to do Total variation
% Denoising (TVD)

%   Output:
%           x : Denoised image (Display this)
%          err: Error at each iteration (Plot this to see convergence)
%   Input:
%           y  : Noisy Image (Observed Image)
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

y = double(y);
y = y(:); 

n = length(y);

[D DT DDT ] = DiffOper(sqrt(n)); %pre-compute some matrices, including 
                                 % our differential operator D (hor and
                                 % ver)

x = y;

% For Images, we need vertical and Horizontal differnece matrix
Dx = D*x; %N-Point Vertical difference
Dy = D*y; %N-Point Horizontal difference

dim = length(Dx);
err = zeros(Nit,1);

% Algorithm iteration start here...
for k = 1:Nit
    
    xu = x;
    F = 1/lam * spdiags(abs(Dx),0,dim,dim) + DDT;  %1/lambda*Dx + DDT
    z = cgs(F,Dy,[],40); %solve linear system for z, F*z = Dy
    x = y - DT*z; %update x, see reference 1)
    e = norm(xu-x)/norm(x); %convergence error
    err(k) = e;
    Dx = D*x; 
    
end

x = reshape(x,256,256);

end

% Function to create vertical and Horizontal difference matrix
function [D DT DDT] = DiffOper(N)
B = spdiags([-ones(N,1) ones(N,1)], [0 1], N,N+1);
B(:,1) = [];
B(1,1) = 0;
D = [ kron(speye(N),B) ; kron(B,speye(N)) ]; %combine vertical and horizontal
                                             % difference matrix in one big
                                             % matrix D. refer 2)
DT = D';
DDT = D*D';
end