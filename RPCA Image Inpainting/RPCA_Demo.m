clc;
clear all;
close all;

% First Created on 3/07/2017 by Tarmizi Adam
% Update History : 8/6/2019,
% Demo for robust principal component analysis (RPCA) using ADMM
% Uses the ADMM/ADM algorithm to solve the the pricipal component pursuit:
%
%    min_L&M  ||L||_* + lambda*||S||_1
%    s.t  L + S = M
%
% This code demostrates the RPCA for image inpainting (Filling missing pixel values).
% The main solver is "RPCA_ADMM()". For furter information, check file
% "RPCA_ADMM.m"


Img = imread('peppers.bmp');

sigma = 0.2; %density of the noise
Img_n = imnoise(Img,'salt & pepper',sigma);
Img_n = double(Img_n);

%% ==================== Parameter options =====================

opts.lam       = 0.053; % Regularization parameter
opts.Nit       = 1000;  % Algorithm iteration
opts.tol       = 1.0e-5; %Stopping criterion value
opts.rho      = 0.05;    % Regularization param of ADMM constraint

%% ==================== Run our ADMM algorithm for RPCA ! ==================
out = RPCA_ADMM(Img_n, opts);


%% ===================== Show some results =====================
figure;
imshow(uint8(Img_n));
title(sprintf('Corrupted (Missing pixels).(PSNR = %3.3f dB)', psnr_fun(Img_n, double(Img))));

figure;
imshow(Img);
title('Original');

figure;
imshow(uint8(out.solL));
title(sprintf('Low rank restored (PSNR = %3.3f dB,SSIM = %3.3f) ',...
                       psnr_fun(out.solL,double(Img)),ssim_index(out.solL,double(Img))));
                  
figure;
imshow(uint8(out.solS));
title(sprintf('Sparse component'));

figure;
plot(out.relativeError);
title('Relative Error');

