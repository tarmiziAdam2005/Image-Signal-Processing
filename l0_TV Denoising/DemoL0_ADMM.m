
% Created by Tarmizi Adam,27/8/2019

% This demo runs the L0-TV impulse noise restoration. Works for pure
% denoising and deblurring under influence of impulse noise.
% For further information on L0-TV please refer to the function "L0_ADMM.m"
% function called within this script.

clc;
clear all;
close all;

imageName = 'peppers.bmp';    

Img = imread(imageName);

if size(Img,3) > 1
    Img = rgb2gray(Img);
end

[row, col] = size(Img);

row = int2str(row);
col = int2str(col);

imageSize = [row 'x' col];

K = fspecial('gaussian', [7 7], 5); % Gaussian Blur
%K     =   fspecial('average',1); % For denoising
f = imfilter(Img,K,'circular');


f  = impulsenoise(f,0.5,0);
f = double(f);

O = ones(size(Img));
O(f == 255) = 0;
O(f == 0) = 0;

Img = double(Img)/255;
f = f/255;


opts.lam = 2.0; % Regularization parameter, play with this !
opts.tol = 1e-4;
opts.Nit = 300;
opts.O   = O;

%******** Main denoising function call********

out =  L0_ADMM(f,Img,K,opts);


figure;
imshow(Img,[]);

figure;
imshow(f,[]);

figure;
imshow(out.sol,[]);
title(sprintf('l0-TV_ADMM Denoising (PSNR = %3.3f dB,SSIM = %3.3f) ',...
                       psnr_fun(out.sol*255,Img*255),ssim_index(out.sol*255,Img*255)));

