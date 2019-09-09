
% 28/8/2019
% This is the demo file to run image denoising and deblurring with impulse
% noise. The algorithm is the L1-OGSTV (re-implementation from the original paper).

% L1 --> Data fidelity term (for impulse noise)
% OGSTV --> Regularization term (supress staircase artifacts)

% See the function "gstv2d_imp.m" for further details on L1-OGSTV

clc;
clear all;
close all;

imageName = 'boat256.bmp';    

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
f1 = imfilter(Img,K,'circular');
f1 = double(f1);


f  = impulsenoise(f1,0.7,0);
f = double(f);
Img = double(Img);


opts.lam       = 18; % for denoising try starting with these  
opts.grpSz     = 3; % OGS group size
opts.Nit       = 300;
opts.Nit_inner = 5;
opts.tol       = 1e-4;

% main function

out = gstv2d_imp(f,Img,K,opts);
 
 
    
figure;
imshow(out.sol,[]),
title(sprintf('ogs2d\\_tv (PSNR = %3.2f dB,SSIM = %3.3f, cputime %.2f s) ',...
                      psnr_fun(out.sol,Img),ssim_index(out.sol,Img)))
 
figure;                   
imshow(uint8(Img))
title('Original Image');

figure;
imshow(uint8(f))
title('Blur + Noisy');



