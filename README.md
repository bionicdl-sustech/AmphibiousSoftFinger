# Autoencoding a Soft Touch to Learn Grasping Underwater
*[Ning Guo](https://gabriel-ning.github.io/), Xudong Han, Xiaobo Liu, Shuqiao Zhong, Zhiyuan Zhou, Jian Lin, Jiansheng Dai, Fang Wan**, Chaoyang Song*


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


## Links

- [Project Page](https://gabriel-ning.github.io/research/softfingerlearning/)
