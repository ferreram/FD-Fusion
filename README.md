# FD-Fusion

Repository for the code related to the paper: Fast Stereo Disparity Maps Refinement By Fusion of Data-Based And Model-Based Estimations - 3DV 2019

The code will be released anytime soon...


## Abstract
The estimation of disparity maps from stereo pairs has many applications in robotics and autonomous driving. Stereo matching has first been solved using model-based approaches, with real-time considerations for some, but today's most recent works rely on deep convolutional neural networks and mainly focus on accuracy at the expense of computing time.  In this paper, we present a new method for disparity maps estimation getting the best of both worlds: the accuracy of data-based methods and the speed of fast model-based ones.  The proposed approach fuses prior disparity maps to estimate a refined version.  The core of this fusion pipeline is a convolutional neural network that leverages dilated convolutions for fast context aggregation without spatial resolution loss.  The resulting architecture is both very effective for the task of refining and fusing prior disparity maps and very light, allowing our fusion pipeline to produce disparity maps at rates up to 125 Hz.  We obtain state-of-the-art results in terms of speed and accuracy on the KITTI benchmarks.

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
