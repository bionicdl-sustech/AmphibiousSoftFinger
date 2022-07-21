# SVAE for Soft Finger Force Prediction

This repository contains the implementation of the Supervised Variational Autoencoder for Soft Finger Force Prediction from the paper:

**Autoencoding a Soft Touch to Learn Grasping Underwater**

*[Ning Guo](https://gabriel-ning.github.io/), Xudong Han, Xiaobo Liu, Fang Wan**, Chaoyang Song*


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


## Links

- [Project Page](https://gabriel-ning.github.io/research/softfingerlearning/)
