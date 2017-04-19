% ================ Demo for grayscale image denoising==================
% This code demostrates the use of the ADMM Image denoising. ADMM is a
% splitting version of the more classical ALM (Augmented Lagrangian Method)
% This code solves the problem of
%       min_x  1/2*||y - x||^2_2 + lamda||Dx||_1

clc
clear all;
close all;

Img = double(imread('Lighthouse256.bmp')); %Your Image goes here

sigma = 25; % standard variation

g = Img +  sigma * randn(size(Img)); %Add a little noise

lam    = 29.7;


res     = cell([1 size(lam,2)]);
resSSIM = cell([1 size(lam,2)]); %Store SSIM result of each iteration
resPSNR = cell([1 size(lam,2)]); %Store PSNR result of each iteration

rho     = 2; %regularization param related to the lagrange constraints

Nit     = 400;
tol     = 1e-5;

%Regularization function. Isotropic TV ('iso') or Anisotropic TV ('ani')
regType = 'iso';

%=============Denoising algorithm==========
 
    out = ADMM(g,Img,lam,rho,Nit,tol,regType);

%========================================


figure;
imshow(uint8(out.sol));
title(sprintf('ADMM Denoised (PSNR = %3.3f dB,SSIM = %3.3f, cputime %.3f s) ',...
                       out.psnrf, out.ssimf, out.cpuTime));

figure;
imshow(uint8(g));
title(sprintf('Noisy (PSNR = %3.3f dB, SSIM = %3.3f)',  psnr_fun(g, Img),ssim_index(g,Img)));

figure;
imshow(uint8(Img));
title(sprintf('Original (PSNR = %3.3f dB)',  psnr(Img,Img)));

