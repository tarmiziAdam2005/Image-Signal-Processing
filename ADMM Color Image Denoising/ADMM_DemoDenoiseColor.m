clc;
clear all;
close all;

Img = double(imread('butterfly.png')); %Your Image goes here

[M,N,O] = size(Img);

sigma = 30; % standard variation of noise

%% Adding noise to each chanel %%
g1 = Img(:,:,1) + sigma * randn(size(Img(:,:,1))); 
g2 = Img(:,:,2) + sigma * randn(size(Img(:,:,2)));
g3 = Img(:,:,3) + sigma * randn(size(Img(:,:,3))); 
%%

imgNoise = cat(3,g1,g2,g3); %The noisy image

%% Some initializations %%

cleanImg = zeros(M,N,O); % pre-allocate space for our denoised image

lam    = 25; % Regularization parameter. Play around with this !
             % Larger values smoothens the image from noise.
             
rho     = 2; %initial regularization param related to the lagrange constraints
             % You can try to play around with this. However, in this code
             % this value will be updated in each iteration of the
             % algorithm to speed up the convergence.
             
Nit     = 400;  % Total iteration of the algorithm
tol     = 1e-5; % Error tolerance before the algorithms stops. (Stopping criteria)

%Regularization function. Isotropic TV ('iso') or Anisotropic TV ('ani')
regType = 'ani';
%%

%=============Denoising algorithm==========
overallTime = tic;
for i = 1:3
    
    % Here, we denoise each rgb chanel separately.
    out = ADMM(imgNoise(:,:,i),Img,lam,rho,Nit,tol,regType);
    cleanImg(:,:,i) = out.sol;
    
end
%========================================
overallTime = toc(overallTime);

figure;
imshow(uint8(Img));
title(sprintf('Original Clean Image'));


figure;
imshow(uint8(imgNoise));
title(sprintf('Noisy Image (Additive Gaussian)'));

figure;
imshow(uint8(cleanImg));
title(sprintf('ADMM Denoised Image (CPU time = %3.3f)',overallTime));
