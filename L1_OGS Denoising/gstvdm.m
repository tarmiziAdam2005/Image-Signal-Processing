function x  = gstvdm(y, K, lam, Nit)
% [x, cost] = gstvdm(y, K, lam, Nit)
% Group-Sparse Total Variation modification Denoising.
%
% INPUT
%   y - noisy signal
%   K - group size (small positive integer)
%   lam - regularization parameter (lam > 0)
%   Nit - number of iterations
%
% OUTPUT
%   x - denoised signal


% Ivan Selesnick, selesi@poly.edu, 2012
% Modified by LJ, UESTC, 2013
% history
h = ones(K,K);                                      % For convolution 
x = y;                                              % Initialization

if K ~=1
    for k = 1:Nit
        r = sqrt(conv2(abs(x).^2, h,'same')); % zero outside the bounds of  x
     %   r =  sqrt(imfilter(abs(x).^2,h));  % slower than conv2
        v = conv2(1./r, h, 'same');     
    %     F = 1./(lam*v) + 1;
    %     x = y - y./F;      
        x = y./(1+lam*v);
    end
else
    for k = 1:Nit
        r = sqrt(abs(x).^2); % zero outside the bounds of  x
     %   r =  sqrt(imfilter(abs(x).^2,h));  % slower than conv2
        v = 1./r;  
    %     F = 1./(lam*v) + 1;
    %     x = y - y./F;      
        x = y./(1+lam*v);
     end
end

