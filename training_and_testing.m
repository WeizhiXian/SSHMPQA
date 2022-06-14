%{ One can use the code in the "main_demo.m" to extract the features of images in the datasets, or
%  directly download the pre-extracted features (kadid.mat, live.mat, csiq.mat, tid.mat) from the following website.
%  https://pan.baidu.com/s/1uQX3BtRrdemfVd_4ZBn0fA
%  (password:SSHM).
%}

% training:
%
% Kadid10k
% SROCC 0.9478
% KROCC 0.8015
%
% testing:
% LIVE
% SROCC 0.9627
% KROCC 0.8278
%
% CSIQ
% SROCC 0.9451
% KROCC 0.7952
%
% TID2013
% SROCC 0.8793
% KROCC 0.6942

clc
clear
close all

% load data  and MOS (5901-th column) normalization

load("kadid.mat");
Mkadid = 5;
mkadid = 1;
kadid(:,5901) = (kadid(:,5901)-mkadid)/(Mkadid-mkadid);

load("live.mat");
load("csiq.mat");
load("tid.mat");

% data preprocessing (features normalization)

data = [kadid;live;csiq;tid];
temp = data(:,5901);
M = max(data,[],1);
M = repmat(M,size(data,1),1);
m = min(data,[],1);
m = repmat(m,size(data,1),1);
data = (data - m)./(M-m);
data(:,5901) = temp;
kadid = data(1:10125,:);
live =  data(10126:10904,:);
csiq= data(10905:11770,:);
tid = data(11771:14770,:);

% training

train = kadid;
A = [ones(size(train,1),1),train(:,1:5900)];
lambda =500;
I = eye(size(A,2));
I(1,1) = 0;
b = train(:,5901);
H = (A'*A) +lambda*I ;
H = (H+H')/2;
f = A'*b;
w = H\f;

% testing

figure(1)
X = [ones(size(kadid,1),1),kadid(:,1:5900)];
Y = kadid(:,5901);
Y = (Mkadid-mkadid)*Y+mkadid;
YPred = X*w;
plot(Y,YPred,'o');
SROCC1 = corr(Y,YPred,'type','Spearman');
KROCC1 = corr(Y,YPred,'type','Kendall');
fprintf("Kadid10k\nSROCC\t%6.4f\nKROCC\t%6.4f\n\n",SROCC1,KROCC1);

figure(2)
X = [ones(size(live,1),1),live(:,1:5900)];
Y = live(:,5901);
YPred = X*w;
plot(Y,YPred,'o');
SROCC2 = corr(Y,YPred,'type','Spearman');
KROCC2 = corr(Y,YPred,'type','Kendall');
fprintf("LIVE\nSROCC\t%6.4f\nKROCC\t%6.4f\n\n",SROCC2,KROCC2);

figure(3)
X = [ones(size(csiq,1),1),csiq(:,1:5900)];
Y = csiq(:,5901);
YPred = X*w;
plot(Y,YPred,'o');
SROCC3 = corr(Y,YPred,'type','Spearman');
KROCC3 = corr(Y,YPred,'type','Kendall');
fprintf("CSIQ\nSROCC\t%6.4f\nKROCC\t%6.4f\n\n",SROCC3,KROCC3);

figure(4)
X = [ones(size(tid,1),1),tid(:,1:5900)];
Y = tid(:,5901);
YPred = X*w;
plot(Y,YPred,'o');
SROCC4 = corr(Y,YPred,'type','Spearman');
KROCC4 = corr(Y,YPred,'type','Kendall');
fprintf("TID2013\nSROCC\t%6.4f\nKROCC\t%6.4f\n\n",SROCC4,KROCC4);


