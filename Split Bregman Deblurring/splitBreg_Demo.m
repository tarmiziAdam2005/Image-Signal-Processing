% ================ Demo for grayscale image deblurring==================
% This code demostrates the use of the Split Bregman method Total  
% Variation (SBMTV) for image deblurring.
% Make sure the main SBMTV file is in the same folder, "splitBreg.m"

clc
clear all;
close all;

Img = double(imread('peppers.bmp')); %Your Image goes here
%I = double(Img);

%H = ones(9,9)/81;
H = fspecial('gaussian', [7 7], 5);
%H = fspecial('motion',20,45); %Try a motion blur
g = imfilter(Img,H,'circular');

BSNR = 40;
sigma = BSNR2WGNsigma(g, BSNR);
g = g +  sigma * randn(size(Img)); %Add a little noise

                    
lam     = 10; %regularization parameter associated with the constraint
mu      = 10000; %Regularization parameter
res     = cell([1 size(lam,2)]);
Nit     = 10;
tol     = 1e-5;
pm.mu = 30; 
%=============Deblurr algorithm==========
%for k=1:length(lam)
    tg = tic;
    %
    out = splitBreg(g,Img,H,lam,mu,Nit,tol);
    %[uTV,outputTV] = deblurTV(g,H,pm);
    tg = toc(tg);
%end
%========================================

figure;
imshow(uint8(out.sol));
title('Deblurred');

figure;
imshow(uint8(g));
title('Blurred');

figure;
imshow(uint8(Img));
title('Original');


