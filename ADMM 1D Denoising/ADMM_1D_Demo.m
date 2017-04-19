
% Created on 5/3/2017 by Tarmizi Adam.
% Small demo for 1D-signal Total Variation (TV) denoising using the
% Alternating Direction Methods of Multiplier (ADMM).
% This demo file calls the function " ADMM_1D() " solver.

clc;
clear all;
close all;


%Generate sine wave;
%{
x = -10*pi:.1:10*pi; 
y = sin(x); 
plot(x,y)
%}

%Load a piecewice defined function instead of a sine wave
% You can use any  signal that you have
load testSig3.mat;
y = testSig3;

%add some noise to it
sigma = 10;
noisy_y = y + sigma * randn(1, length(y));

%figure;
%plot(x,noisy_y) %To plot sine wave;

lam = 0.0078;
rho = 1.0;
Nit = 200;

%% ********** Run the TV-solver ***************

out = ADMM_1D(noisy_y, lam, rho, Nit); %Run the Algorithm !!!

%% ********************************************

%%
figure;
subplot(3,1,1)
plot(y);
axis tight;
title('Original Signal');

subplot(3,1,2);
plot(noisy_y)
axis tight;
title('Noisy Signal');

subplot(3,1,3);
plot(out.sol);
axis tight;
title('TV Denoised');

figure;
plot(out.funVal);
title('Function Value');
%%