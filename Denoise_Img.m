%Created by Tarmizi Adam 1/9/2015. TVD and GSTVD denoising for random Noise 
%Run this script with functions:

%[x1,err1] = TVD_Img(y,lam,Nit);
%           or
%[x1,err1] = GSTVD_Img(y,K,lam,Nit);

%Which must be in the same folder as this script.




clc; clear all;
close all;

s = double(imread('lenaN','bmp'));

Nit = 20; %Number of iterations to run the algorithm
sigma = 30; %Noise level added to the image
lam = 10 %sqrt(3*sigma); %Regularization parameter lambda
K = 6;

%% Add noise to image
noise = sigma * randn(size(s)); 
y = s + noise;  

[x1,err1] = TVD_Img(y,lam,Nit);
%[x1,err1] = GSTVD_Img(y,K,lam,Nit);

score = psnr(s,x1); %PSNR score of the denoised Image
rmse = sqrt(sum((s(:)-x1(:)).^2)/numel(s)); %RMSE score of the denoise image
x1 = uint8(x1);


%%                      Start Plotting Results
figure;

subplot 431
imshow(uint8(s));
title('Original');

subplot 432
imshow(uint8(y));
title(['Noisy, \sigma = ' int2str(sigma)]);

subplot 433
imshow(x1);
title(['Denoised with \lambda = ',int2str(lam),', PSNR = ',num2str(score)])

subplot 434
plot(err1)
title('Error')


