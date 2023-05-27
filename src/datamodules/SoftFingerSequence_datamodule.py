from typing import Optional, Tuple
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from skimage import io

import torchvision
import matplotlib.pyplot as plt


class Soft_Finger_Sequence(Dataset):
    """
    Soft Finger Dataset.
        Serves as input to DataLoader to transform X 
        into sequence data using rolling window. 
        DataLoader using this dataset will output batches 
        of `(batch_size, seq_len, n_features)` shape.
    """
    def __init__(self, image_folder_dir: str, lable_npy_file: str, seq_len: int = 1, transform = None):
        """
        Args:
            image_folder_dir (string): Absolute Path to the images. eg. "/home/.../.../images"
                contents:   "1.png", "2.png" ..., "xx.png", ..., 
                             xx stands for sequence
            lable_npy_file (string): contains all the forces lable correspondes to images.
        """
        
        'Gieven data size is 500'
        '[0:500,:] can be drop out'
        numpy_label = np.load(lable_npy_file)[0:500,:]
        labels = torch.tensor(numpy_label).float()
        self.seq_len = seq_len
        self.transform = transform
        
        images_tensor = []
        for index in range(labels.__len__()):
            img_name = os.path.join(image_folder_dir, str(index) + ".png")
            image_numpy = io.imread(img_name)
            
            if self.transform:
                image_tensor = self.transform(image_numpy)
            else :
                a = transforms.ToTensor()
                image_tensor = a(image_numpy)
            
            images_tensor.append(image_tensor)
            
        images_tensor = torch.stack(images_tensor)
        
        # print(images_tensor.shape)
        # print(labels.shape)
        
        self.images_tensor = []
        self.labels = []
        for index in range(labels.__len__() - (self.seq_len - 1)):
            self.images_tensor.append(images_tensor[index:index+self.seq_len])
            self.labels.append(labels[index + self.seq_len - 1])
        
        self.images_tensor = torch.stack(self.images_tensor)
        self.labels = torch.stack(self.labels)
        
        # print(self.images_tensor.shape)
        # print(self.labels.shape)
        
    def __len__(self):
        return self.labels.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (self.images_tensor[index,:], self.labels[index])


class SoftFingerSequenceDataModule(LightningDataModule):
    """
    Example of LightningDataModule for SoftFinger dataset.
    SoftFinger dataset sample contains :
                 a reseized (224 * 224) image
                 a labels Fx, Fy, Fz, Tx, Ty, Tz
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """
    def __init__(
        self,
        image_folder_dir: str,
        lable_npy_file: str, 
        seq_len: int = 3,
        train_test_split: Tuple[float, float] = (0.7, 0.2),
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):       
        super().__init__()
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # self.data_shape = data_type
        
        self.image_folder = image_folder_dir
        self.label_file = lable_npy_file
        self.seq_len = seq_len
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize([224, 224]),
            ]
        )

        # self.dims is returned when you call datamodule.size()
        # self.dims = (3, 224, 224)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
        # DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
        # print("Data mean", DATA_MEANS)
        # print("Data std", DATA_STD)

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`.
        """
        if not self.data_train or not self.data_val or not self.data_test:
            dataset = Soft_Finger_Sequence(self.image_folder, self.label_file, self.seq_len , self.transform)

            train_length = int(
                self.train_test_split[0] * len(dataset))
            
            test_length = int(
                self.train_test_split[1] * len(dataset))
            
            val_length = int(
                len(dataset) - train_length - test_length)

            self.data_train, self.data_val, self.data_test = random_split(
                dataset, (train_length, val_length,
                          test_length), generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def get_train_images(self, num):  
        return torch.stack([self.data_train[i][0] for i in range(num)], dim=0)

# def show(imgs):
#     grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
#     plt.imshow(grid)
#     # plt.axis('off')
#     plt.show()

if __name__ == "__main__":
    a = SoftFingerSequenceDataModule(image_folder_dir="/home/ghost/Documents/workspace/Thesis-Project/AmphibiousSoftFinger/images",
                                     lable_npy_file="/home/ghost/Documents/workspace/Thesis-Project/AmphibiousSoftFinger/force_vecs.npy")
    a.setup()

    for batch in a.train_dataloader():
        x, label = batch
        print(label.shape)
        print(x.shape)

    b = a.get_train_images(4)
    print(b.shape)
        # show(b)
    
    # print(a.data_train[1].shape)

    
