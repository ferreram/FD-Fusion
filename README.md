# FD-Fusion

Repository for the code related to the paper: Fast Stereo Disparity Maps Refinement By Fusion of Data-Based And Model-Based Estimations - 3DV 2019


## Usage Samples

### Train on KITTI 2015 - Test on KITTI 2012

python train_kitti.py --test_inputs_file data_files/kitti_2015_train_inputs_lr_sgm.txt --test_targets_file data_files/kitti_2015_train_targets.txt --test_inputs_file data_files/kitti_2012_train_inputs_lr_sgm.txt --test_targets_file data_files/kitti_2012_train_targets.txt --datadir /path/kitti_stereo/ --mode 2  --method 2

## Abstract
The estimation of disparity maps from stereo pairs has many applications in robotics and autonomous driving. Stereo matching has first been solved using model-based approaches, with real-time considerations for some, but today's most recent works rely on deep convolutional neural networks and mainly focus on accuracy at the expense of computing time.  In this paper, we present a new method for disparity maps estimation getting the best of both worlds: the accuracy of data-based methods and the speed of fast model-based ones.  The proposed approach fuses prior disparity maps to estimate a refined version.  The core of this fusion pipeline is a convolutional neural network that leverages dilated convolutions for fast context aggregation without spatial resolution loss.  The resulting architecture is both very effective for the task of refining and fusing prior disparity maps and very light, allowing our fusion pipeline to produce disparity maps at rates up to 125 Hz.  We obtain state-of-the-art results in terms of speed and accuracy on the KITTI benchmarks.


## Model-based Methods Settings

### SGM

Algorithm: [CUDA - SGM](https://github.com/dhernandez0/sgm)

##### Settings:

8 path directions: P1=6, P2=96

### SGBM

Algorithm: [OpenCV](https://docs.opencv.org/3.3.1/d2/d85/classcv_1_1StereoSGBM.html) v. 3.3.1.

##### Settings:

pre_filter_cap = 63  
sad_window_size = 3  
p1 = sad_window_size\*sad_window_size\*4  
p2 = sad_window_size\*sad_window_size\*32  
min_disparity = 0  
num_disparities = 128  
uniqueness_ratio = 10  
speckle_window_size = 100  
speckle_range = 32  
disp_max_diff = 1  
full_dp = 1  

sgbm = cv2.StereoSGBM_create(min_disparity, num_disparities, sad_window_size,p1,p2,disp_max_diff, pre_filter_cap,
                uniqueness_ratio, speckle_window_size, speckle_range, full_dp)


### ELAS

Algorithm: [pyelas](https://github.com/jlowenz/pyelas)

##### Settings:

disp_min              = 0;     disp_max              = 255;  support_threshold     = 0.95;  
support_texture       = 10;    candidate_stepsize    = 5;    incon_window_size     = 5;  
incon_threshold       = 5;     incon_min_support     = 5;    add_corners           = 1;  
grid_size             = 20;    beta                  = 0.02; gamma                 = 5;  
sigma                 = 1;     sradius               = 3;    match_texture         = 0;  
lr_threshold          = 2;     speckle_sim_threshold = 1;    speckle_size          = 200;  
ipol_gap_width        = 5000;  filter_median         = 1;    filter_adaptive_mean  = 0;  
postprocess_only_left = 0;     subsampling           = 0;  

## Reference
If you find our work  useful in your research, please consider citing our paper:
```
@inproceedings{FerreraFDFusion2019,
  title     = {Fast Stereo Disparity Maps Refinement By Fusion of Data-Based And Model-Based Estimations},
  author    = {Ferrera, Maxime and Boulch, Alexandre and Moras, Julien},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2019}
}
```
