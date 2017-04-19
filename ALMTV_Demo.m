% ================ Demo for grayscale image deblurring==================
% This code demostrates the use of the ALMTV (Augmented Lagrangian Method
% Total Variation) code.

clc
clear all;
close all;

Img = im2double(imread('Lighthouse256.bmp')); %Your Image goes here
%I = double(Img);

%H = ones(9,9)/81;
H = fspecial('gaussian', [7 7], 5);
%H = fspecial('motion',20,45); %Try a motion blur
g = imfilter(Img,H,'circular');

%[g,I] = BlurCrop(Img,H);
%g = imnoise(g, 'gaussian', 0, 0.00001); % A little bit of noise

lam     = 10000; %Regularization parameter
rho     = 2;
Nit     = 20;
tol     = 1e-5;

%=============Deblurr algorithm==========
tg = tic;
[f, relchg] = ALMTV(g,H,lam,rho,Nit,tol);
tg = toc(tg);
%========================================

figure;
imshow(f);
title('Deblurred');

figure;
imshow(g);
title('Blurred');

figure;
imshow(Img);
title('Original');