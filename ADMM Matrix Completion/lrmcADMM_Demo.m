% This demo files solve the low rank matrix completion problem applied to
% image inpainting. In this demo, we apply the Alternating Direction Method
% of Multipliers (ADMM) to do so.

% Created by Tarmizi Adam on 31/12/2019. For furter assistance on the ADMM
% algorithm used for the matrix completion problem, refer to the file
% "lrmcADMM.m" file.
%

clc;
clear all;
close all;
                        
  %% For image inpainting %%
  
  X = imread('peppers.bmp');
  X = double(X);
  
  [n1,n2] =size(X);

%% Create projection matrix %%
MisLvl = 0.2; % percentage of missing pixels/entries, change here.

J = randperm(n1*n2);
J = J(1:round(MisLvl*n1*n2)); 
P = ones(n1*n2,1);
P(J) = 0;
P = reshape(P,[n1,n2]); % our projection matrix

%% Simulate our corrupted original matrix %%
Y = X(:);
sigma = 30; %noise level
noise = sigma*randn(n1*n2,1);

Y = Y + noise;
Y = reshape(Y,[n1,n2]);
Y = P.*Y; % Our final noisy + missing entry matrix (Observation)

opts.lam = 800; % Regularization parameter. Play with this and see effects.
opts.Nit = 500; % Number of iteration for algorithm termination
opts.tol = 1e-3;

out = lrmcADMM(Y,X,P,opts); % Call ADMM solver for matrix completion.

figure;
imshow(out.sol,[]);

figure; 
imshow(X,[]);

figure;
imshow(Y,[]);








