# Autoencoding a Soft Touch to Learn Grasping Underwater
*Ning Guo, Xudong Han, Xiaobo Liu, Shuqiao Zhong, Zhiyuan Zhou, Jian Lin, Jiansheng Dai, Fang Wan**, Chaoyang Song*


## Graphic Abstract
![graphical_abstract](https://github.com/bionicdl-sustech/AmphibiousSoftFinger/assets/42087775/6b8aeedf-3895-4105-977d-6d2db1ae758d)


## Datasets

1. Download [Soft Finger Dataset](https://drive.google.com/file/d/19CmZHYsDnuvNeUjVXZHiOqFZsTBYsM9z/view?usp=sharing). 
2. Extract "dataset" folder to root directory of this git repository : /path/to/SoftFingerSvae


## Training && Evaluation

1. Cnn regression model: SoftFinger_cnn.py
2. Vae model: SoftFinger_Vae.py
3. Svae model: SoftFinger_Svae.py


## Requirements

This code was developed with Python 3.8 on Ubuntu 18.04.  Additional Python packages:

- pytorch
- pytorch_lightning
- torchmetrics
- numpy
- torchvision
- pandas
- skimage
- PILLOW

## Supplementary Videos

### Movie S1: Amphibian Grasping with Visual-Tactile Soft Finger.
https://github.com/bionicdl-sustech/AmphibiousSoftFinger/assets/42087775/6b174823-5098-4b22-b3b1-d1dfa4ef5bca

### Movie S2: Real-time Force/Torque Prediction.
https://github.com/bionicdl-sustech/AmphibiousSoftFinger/assets/42087775/6318d6fb-f823-4b1a-bdbf-4b306a1af491

### Movie S3: Object Grasping Success Rates Experiments with/without Contact Feedback.
https://github.com/bionicdl-sustech/AmphibiousSoftFinger/assets/42087775/f4dd5d5d-09b0-4d93-8122-fa9ddbaa3db7


### Movie S4: Contact Force Following Experiments.
https://github.com/bionicdl-sustech/AmphibiousSoftFinger/assets/42087775/589bc227-b075-4e7e-9789-6bfa5914213b


### Movie S5: Object Shape Adaptation Experiments.
https://github.com/bionicdl-sustech/AmphibiousSoftFinger/assets/42087775/c0851ee2-04e8-4b1b-bd4b-9cf1b75559be


### Movie S6: Robot End-effector Reaction to Soft Finger Twist.

https://github.com/bionicdl-sustech/AmphibiousSoftFinger/assets/42087775/b478be4c-6425-409e-8a5b-9867c33a15ff




## Links

- [Project Page](https://gabriel-ning.github.io/research/softfingerlearning/)
