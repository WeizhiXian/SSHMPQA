# DTSIQA

This is a matlab implementation of Perceptual Quality Analysis in Deep Domain Using Structure Separation and High-Order Moment (SSHMPQA).

## Environments

* Matlab >= 2019b

## Model

* You can use "main_demo.m" to calculate the perceptual quality of two images.
* example 1
    - ref: texture1.jpg
    - dist1: texture2.jpg, texture3.jpg
* example 2
    - ref: init.jpg
    - dist: timepass.jpg, blur.jpg, gaussnoise.jpg, jpeg.jpg, jpeg2000.jpg, moveright.jpg, rotate.jpg, scale.jpg

## Training & Testing

* Training dataset: Kadid10k
* Testing: datasets: LIVE, CSIQ, TID2013
* You can use "training_and_testing.m" to implement training and testing on the datasets.
* You can use "function [Stru,M,K,S] = deep_features(I,params,resize_img)" and "function dist_ref_features = final_features(ref_Stru,ref_M,ref_K,ref_S,dist_Stru,dist_M,dist_K,dist_S)" in the "main_demo.m" to extract the final features of images in the datasets.
* You can also directly download the pre-extracted final features of images in these datasets ("kadid.mat", "live.mat", "csiq.mat", "tid.mat") from <a href="">https://pan.baidu.com/s/1uQX3BtRrdemfVd_4ZBn0fA</a> (password:SSHM)
