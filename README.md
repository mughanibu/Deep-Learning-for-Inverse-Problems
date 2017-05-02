# Deep Learning for Inverse Problems

## Overview
This project hosts the code and datasets I used for Deep Learning course at Boston University. It aims to post-process the images the low quality images produced as a result of solving inverse problems in imaging (particularly Computed Tomography) and produce high-quality images.

We use two different networks, modified U-Net [1], and VDSR [2], and compare their performance for a Computed Tomography problem.  Various parts of this implementation use codes from [3] and [4]. Please cite the relevant papers if you use a network in your work.

Data used for this work can be downloaded from the following Google drive link:
https://drive.google.com/drive/folders/0Bx_60BaQ0CesMk43UkdReXlxRlU?usp=sharing

* U-Net has been modified to produce single channel output of dimensions same as the dimensions of input image. Additionally, we also add Batch-Norm to each convolutional layers.

## Files
- VDSR.py	: training and testing file for VDSR network performing image learning task using MSE loss function.
- VDSR_mae.py	: training and testing file for VDSR network performing image learning task using MAE loss function.
- VDSR_mae_res.py	: training and testing file for VDSR network performing residual learning task using MAE loss function.
- VDSR_res.py	: training and testing file for VDSR network performing residual learning task using MSE loss function.
- unet.py	: training and testing file for U-Net performing image learning task using MSE loss function.
- unet_mae.py	: training and testing file for U-Net performing image learning task using MAE loss function.
- unet_res.py	: training and testing file for U-Net performing residual learning task using MSE loss function.
- unet_res_mae.py	: training and testing file for U-Net performing residual learning task using MAE loss function.

* MAE stands for minimum absolute error, and MSE stands for minimum squared error.

[1] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015. 
[2] Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks", Proc. of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
[3] https://github.com/jakeret/tf_unet

[4] https://github.com/Jongchan/tensorflow-vdsr
