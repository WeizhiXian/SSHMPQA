% SSHMPQA
% Perceptual Quality Analysis in Deep Domain Using Structure Separation and High-Order Moment

clear
clc

load("model_parameters.mat"); % the predtrained SSHMPQA model
params = load('vgg16net_param.mat'); % the pretrained vgg16 parameters
resize_img = 1; % resize the input image  so that the short side of the image is 256

%========================================================================

ref = imread(".\images\1.png"); % reference image

for i = 2:5
    str = num2str(i);
    dist = imread(strcat(".\images\",str,".png"));% distorted image
    sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
    fprintf("%d SSHMPQA = %6.4f\n",i,sshmvqa);
    ssimval  = SSIM(ref, dist);
    fprintf("%d SSIM = %6.4f\n",i,ssimval);
end

%========================================================================

ref = imread(".\images\texture1.jpg"); % reference image

dist = imread(".\images\texture2.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("SSHMPQA = %6.4f\n",sshmvqa);% 0.11
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);% 0.47

dist = imread(".\images\texture3.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("SSHMPQA = %6.4f\n",sshmvqa);% 0.40
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);% 0.03

%========================================================================

ref = imread(".\images\init.jpg"); % reference image

dist = imread(".\images\moveright.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("moveright\nSSHMPQA = %6.4f\n",sshmvqa);% 0.52
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);% 0.33

dist = imread(".\images\scale.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("scale\nSSHMPQA = %6.4f\n",sshmvqa);%  0.46
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);%  0.38

dist = imread(".\images\rotate.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("rotate\nSSHMPQA = %6.4f\n",sshmvqa);%  0.46
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);%  0.37

dist = imread(".\images\timepass.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("time\nSSHMPQA = %6.4f\n",sshmvqa);% SSHMPQA = 0.48
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);% SSIM = 0.43

dist = imread(".\images\gaussnoise.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("noise\nSSHMPQA = %6.4f\n",sshmvqa);% 0.36
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);% SSIM = 0.53

dist = imread(".\images\blur.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("blur\nSSHMPQA = %6.4f\n",sshmvqa);% SSHMPQA = 0.19
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);% SSIM = 0.55

dist = imread(".\images\jpeg.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("jpeg\nSSHMPQA = %6.4f\n",sshmvqa);%  0.30
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);%  0.68

dist = imread(".\images\jpeg2000.jpg");% distorted image
sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m);
fprintf("jpeg2k\nSSHMPQA = %6.4f\n",sshmvqa);%  0.22
ssimval  = SSIM(ref, dist);
fprintf("SSIM = %6.4f\n",ssimval);% 0.57

%========================================================================

function sshmvqa =SSHMPQA(ref,dist,params,resize_img,w,M,m)

tic
[ref_Stru,ref_M,ref_K,ref_S] = deep_features(ref,params,resize_img);
[dist_Stru,dist_M,dist_K,dist_S] = deep_features(dist,params,resize_img);

% distance features between reference and distorted images
dist_ref_features  = final_features(ref_Stru,ref_M,ref_K,ref_S,dist_Stru,dist_M,dist_K,dist_S);

dist_ref_features = (dist_ref_features - m)./(M-m);
X = [ones(size(dist_ref_features,1),1),dist_ref_features];
sshmvqa = X*w;

toc
end

function dist_ref_features = final_features(ref_Stru,ref_M,ref_K,ref_S,dist_Stru,dist_M,dist_K,dist_S)

dist_ref_features = [];
chns = [3,64,128,256,512,512];

% deep structure maps distance
for i = 1:6
    for k = 1:chns(i)
        temp = norm(dist_Stru{i}(:,:,k)-ref_Stru{i}(:,:,k),'fro');
        dist_ref_features = [dist_ref_features temp];
    end
end

% deep texture maps distance
dist_ref_features = [dist_ref_features abs(dist_M-ref_M) abs(dist_K-ref_K) abs(dist_S-ref_S)];
end

function [Stru,M,K,S] = deep_features(I,params,resize_img)

features = extract_features(I,params,resize_img);

% extract structure map
Stru = cell(6,1);

% extract statistical features to describe texture
M = [];
K = [];
S = [];

texturesize = [6 6 4 4 2 2];
maxIter = [3 3 2 2 1 1];

for i = 1:6
    data = double(extractdata(features{i}));
    St = zeros(size(data));
    G = zeros(size(data,3),size(data,1)*size(data,2));

    for k = 1:floor(size(data,3)/3)
        St(:,:,(3*k-2):3*k) = tsmooth(data(:,:,(3*k-2):3*k),0.05,texturesize(i),0.03,maxIter(i));
    end
    St(:,:,(size(data,3)-2):size(data,3)) = tsmooth(data(:,:,(size(data,3)-2):size(data,3)),0.05,texturesize(i),0.03,maxIter(i));

    for k = 1:size(data,3)
        G(k,:)=reshape(data(:,:,k),1,size(data,1)*size(data,2));
        [m,~,skew,kurt] = calc_statistics(G(k,:));
        M = [M,m];
        K = [K,kurt];
        S = [S,skew];
    end

    Stru{i} = St;

end
end

function features = extract_features(I,params,resize_img)
if resize_img && min(size(I,1),size(I,2))>256
    I = imresize(I,256/min(size(I,1),size(I,2)));
end
I = dlarray(double(I)/255,'SSC');

features = cell(6,1);
% stage 0
features{1} = I;
dlX = (I - params.vgg_mean)./params.vgg_std;

% stage 1
weights = dlarray(params.conv1_1_weight);
bias = dlarray(params.conv1_1_bias');
dlY = relu(dlconv(dlX,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv1_2_weight);
bias = dlarray(params.conv1_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
features{2} = dlY;

% stage 2
weights = dlarray(params.L2pool_1);
dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
dlY = sqrt(dlY);

weights = dlarray(params.conv2_1_weight);
bias = dlarray(params.conv2_1_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv2_2_weight);
bias = dlarray(params.conv2_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
features{3} = dlY;

% stage 3
weights = dlarray(params.L2pool_2);
dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
dlY = sqrt(dlY);

weights = dlarray(params.conv3_1_weight);
bias = dlarray(params.conv3_1_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv3_2_weight);
bias = dlarray(params.conv3_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv3_3_weight);
bias = dlarray(params.conv3_3_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

features{4} = dlY;

% stage 4
weights = dlarray(params.L2pool_3);
dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
dlY = sqrt(dlY);

weights = dlarray(params.conv4_1_weight);
bias = dlarray(params.conv4_1_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv4_2_weight);
bias = dlarray(params.conv4_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv4_3_weight);
bias = dlarray(params.conv4_3_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

features{5} = dlY;

% stage 5
weights = dlarray(params.L2pool_4);
dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
dlY = sqrt(dlY);

weights = dlarray(params.conv5_1_weight);
bias = dlarray(params.conv5_1_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv5_2_weight);
bias = dlarray(params.conv5_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv5_3_weight);
bias = dlarray(params.conv5_3_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

features{6} = dlY;
end

function [m,m2,skew,kurt] = calc_statistics(X)
m = mean(X);
m2 = mean((X-m).^2);
m3 = mean((X-m).^3);
m4 = mean((X-m).^4);

skew = m3/((sqrt(m2))^3+1e-6);
kurt = m4/(m2^2+1e-6)-3;
end

%========================================================================

function S = tsmooth(I,lambda,sigma,sharpness,maxIter)
if (~exist('lambda','var'))
    lambda=0.01;
end
if (~exist('sigma','var'))
    sigma=3.0;
end
if (~exist('sharpness','var'))
    sharpness = 0.02;
end
if (~exist('maxIter','var'))
    maxIter=4;
end
I = im2double(I);
x = I;
sigma_iter = sigma;
lambda = lambda/2.0;
dec=2.0;
for iter = 1:maxIter
    [wx, wy] = computeTextureWeights(x, sigma_iter, sharpness);
    x = solveLinearEquation(I, wx, wy, lambda);
    sigma_iter = sigma_iter/dec;
    if sigma_iter < 0.5
        sigma_iter = 0.5;
    end
end
S = x;
end

function [retx, rety] = computeTextureWeights(fin, sigma,sharpness)

fx = diff(fin,1,2);
fx = padarray(fx, [0 1 0], 'post');
fy = diff(fin,1,1);
fy = padarray(fy, [1 0 0], 'post');

vareps_s = sharpness;
vareps = 0.001;

wto = max(sum(sqrt(fx.^2+fy.^2),3)/size(fin,3),vareps_s).^(-1);
fbin = lpfilter(fin, sigma);
gfx = diff(fbin,1,2);
gfx = padarray(gfx, [0 1], 'post');
gfy = diff(fbin,1,1);
gfy = padarray(gfy, [1 0], 'post');
wtbx = max(sum(abs(gfx),3)/size(fin,3),vareps).^(-1);
wtby = max(sum(abs(gfy),3)/size(fin,3),vareps).^(-1);
retx = wtbx.*wto;
rety = wtby.*wto;

retx(:,end) = 0;
rety(end,:) = 0;

end

function ret = conv2_sep(im, sigma)
ksize = bitor(round(5*sigma),1);
g = fspecial('gaussian', [1,ksize], sigma);
ret = conv2(im,g,'same');
ret = conv2(ret,g','same');
end

function FBImg = lpfilter(FImg, sigma)
FBImg = FImg;
for ic = 1:size(FBImg,3)
    FBImg(:,:,ic) = conv2_sep(FImg(:,:,ic), sigma);
end
end

function OUT = solveLinearEquation(IN, wx, wy, lambda)
%
% The code for constructing inhomogenious Laplacian is adapted from
% the implementaion of the wlsFilter.
%
% For color images, we enforce wx and wy be same for three channels
% and thus the pre-conditionar only need to be computed once.
%
[r,c,ch] = size(IN);
k = r*c;
dx = -lambda*wx(:);
dy = -lambda*wy(:);
B(:,1) = dx;
B(:,2) = dy;
d = [-r,-1];
A = spdiags(B,d,k,k);
e = dx;
w = padarray(dx, r, 'pre'); w = w(1:end-r);
s = dy;
n = padarray(dy, 1, 'pre'); n = n(1:end-1);
D = 1-(e+w+s+n);
A = A + A' + spdiags(D, 0, k, k);
if exist('ichol','builtin')
    L = ichol(A,struct('michol','on'));
    OUT = IN;
    for ii=1:ch
        tin = IN(:,:,ii);
        [tout, ~] = pcg(A, tin(:),0.1,100, L, L');
        OUT(:,:,ii) = reshape(tout, r, c);
    end
else
    OUT = IN;
    for ii=1:ch
        tin = IN(:,:,ii);
        tout = A\tin(:);
        OUT(:,:,ii) = reshape(tout, r, c);
    end
end

end

%========================================================================

function [mssim, ssim_map] = SSIM(img1, img2, K, window, L)
%SSIM Index, Version 1.0
%Copyright(c) 2003 Zhou Wang
%All Rights Reserved.
%
%The author is with Howard Hughes Medical Institute, and Laboratory
%for Computational Vision at Center for Neural Science and Courant
%Institute of Mathematical Sciences, New York University.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names ap pearon all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error visibility to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, no. 4, pp.600-612,
%Apr. 2004.
%
%Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim ssim_map] = ssim_index(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map

if (nargin < 2 || nargin > 5)
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

[M N] = size(img1);

if (nargin == 2)
   if ((M < 11) | (N < 11))
           ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   K(1) = 0.01;                            % default settings
   K(2) = 0.03;                                    
   L = 255;                                  
end

if (nargin == 3)
   if ((M < 11) | (N < 11))
           ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
                   ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
           ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end

if (nargin == 4)
   [H W] = size(window);
   if ((H*W) < 4 | (H > M) | (W > N))
           ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
                   ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
           ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end

if (nargin == 5)
   [H W] = size(window);
   if ((H*W) < 4 | (H > M) | (W > N))
           ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
                   ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
           ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end

if size(img1,3)~=1
   org=rgb2ycbcr(img1);
   test=rgb2ycbcr(img2);
   y1=org(:,:,1);
   y2=test(:,:,1);
   y1=double(y1);
   y2=double(y2);
 else 
     y1=double(img1);
     y2=double(img2);
 end
img1 = double(y1); 
img2 = double(y2);

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));


mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');

mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;

sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 & C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
   denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end
mssim = mean2(ssim_map);

return
end