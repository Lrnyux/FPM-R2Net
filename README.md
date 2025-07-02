# FPM-R2Net
This is the official version of paper **"FPM-R2Net: Fused Photoacoustic and Operating Microscopic Imaging with Cross-modality Representation and Registration Network"**
![Introduction](https://github.com/Lrnyux/FPM-TRNet/assets/86871168/64ee234b-3b4d-4fa2-b2c0-06f5ed08a77a)


## Proposed Method
The proposed method takes the paired PAM and RGB images as input and predicts the correspondence which is utilized to obtain the final fused image as output. The proposed method contains two subnetworks, i.e., MOTNet: Modality Transform Network and HIRNet: Hierarchical Iterative Registration Network. The MOTNet takes the input images and extracts the modality maps which contain the unified representation of vessels and remove background noise. The HIRNet estimates the correspondence based on modality maps in a coarse-to-fine manner.
![Network Architecture](https://github.com/Lrnyux/FPM-TRNet/assets/86871168/805437d4-202a-46e7-9e6a-62bbba2b4676)


## Related Datasets
To evaluate the performance of our proposed method, we propose two datasets for quantitative evaluation.
The proposed synthetic and in vivo datasets will be available upon request.


## Results Example
https://github.com/Lrnyux/FPM-TRNet/assets/86871168/d68cb707-8018-4340-bbee-29f9698e9172



## Citation
Please cite our work if you find this work useful for your research.
```latex
@article{Liufpmr2net2025,
author = {Yuxuan Liu, Jiasheng Zhou, Yating Luo, Sung-Liang Chen, Yao Guo and Guang-Zhong Yang},
title = {FPM-R2Net: Fused Photoacoustic and operating Microscopic imaging with cross-modality Representation and Registration Network},
journal = {Medical Image Analysis},
pages = {103698},
year = {2025},
 } 
  
```
