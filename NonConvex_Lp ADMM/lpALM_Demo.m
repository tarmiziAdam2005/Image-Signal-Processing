% ================ Demo for grayscale image deblurring==================

%Created by Tarmizi Adam. Use this code at your own risk
%Demo code that is used for the lp-norm ALM deblurring code.
% Feel free to email me at tarmizi_adam2005@yahoo.com if you find any 
% bugs or just drop by and inform me that you used the code somewhere in 
% your research.

clc
clear all;
close all;

Img = double(imread('cameraman.bmp')); %Your Image goes here

%H = fspecial('gaussian', [7 7], 5);
H = ones(9,9)/81;
g = imfilter(Img,H,'circular');

BSNR = 30;
sigma = BSNR2WGNsigma(g, BSNR);
g = g +  sigma * randn(size(Img)); %Add a little noise

mu     = 0.45;

%res     = cell([1 size(l,2)]);
rho     = 0.1; %default 2
Nit     = 400;
tol     = 1e-5;
p       = 0.3;

%=============Deblurr algorithm==========
%for k=1:length(lam)
    tg = tic;
    % out = lpALM_rhoUpdate(g,Img,H,mu,rho,Nit,p,tol); %with rho  										 %update
    out = lpALM(g,Img,H,mu,rho,Nit,p,tol); % lp-Norm TV function
    tg = toc(tg);
    %res{1,k} = out;
%end
%========================================


figure;
imshow(uint8(out.sol));
%title('Deblurred');
title(sprintf('TV_p (IRL1) Deblurred (PSNR = %3.3f dB, cputime %.3f s) ',...
                       out.psnrf, tg));

figure;
imshow(uint8(g));
title(sprintf('Blurred (PSNR = %3.3f dB',  psnr_fun(g, Img)));

figure;
imshow(uint8(Img));
title('Original');

