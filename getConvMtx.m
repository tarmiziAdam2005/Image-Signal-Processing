function T = getConvMtx(H,m,n)

%This program was posted by stackoverflow user Stav and can be found
% here: 
%     http://stackoverflow.com/questions/26151265/matlab-2d-convolution-
%     matrix-with-replication
% Copyright thus goes to the user Stav

% Input:
%         H: blur kernel e.g: 5 x 5 gaussian kernel etc.
%         m: row of resulting convolution matrix T.
%         n: columns of the resulting convolution matrix T

% Output
%         T: the convlution matrix

%This code is used to create the convolution matrix T like the predefined
%MATLAB function convmtx2. However, this code replicates the convolution 
%matrix to get an M x M size matrix.

%This code may be used for linear inverse problems i.e. lnear models like
%       y = Tx + w
% where y is the observed blury image T is the convolution matrix (this
% code), x the original clean signal (Image) to be estimated and w the
% noise.

vHalfKerSz = floor(size(H) / 2);

mInds = reshape(1:m*n, m, n);
mInds = padarray(mInds, vHalfKerSz, 'replicate');

Tcols = zeros(m*n*numel(H), 1);
Trows = zeros(m*n*numel(H), 1);
Tvals = zeros(m*n*numel(H), 1);

i = 0; p = 0;
for c = 1:n
    for r = 1:m
        p = p + 1;

        mKerInds = mInds(r:r+size(H,1)-1, c:c+size(H,2)-1);

        [U, ~, ic] = unique(mKerInds(:));

        for k = 1:length(U)
            i = i + 1;
            Tcols(i) = U(k);
            Trows(i) = p;
            Tvals(i) = sum(H(mKerInds == U(k)));
        end
    end
end

T = sparse(Trows(1:i), Tcols(1:i), Tvals(1:i), m*n, m*n);

end