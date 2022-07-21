import torch
from torch import nn, Tensor
from torchvision.transforms import transforms

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()
        
        self.transforms = nn.Sequential(
            transforms.RandomErasing(p=0.75),
            transforms.ColorJitter(brightness=.5,hue=.3),
            transforms.RandomAutocontrast(p=0.75)
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out