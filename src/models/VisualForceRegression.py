from typing import Any, List
import torch
from torch import nn, Tensor
from torchvision import transforms
from pytorch_lightning import LightningModule
from torchmetrics.functional.regression.r2score import r2score
from torchvision import models

import pytorch_lightning as pl

from .modules.Encoder import deformation_encoder

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

class VisualFingerForceNet(LightningModule) :
    """
    Direct use features for regression task
    Re-implement the regression block
    add latent layer+ relu

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        drop_out: float = 0.5,
        relative_f_t = 1.0,
        data_augmentation : bool = True,
        first_conv: bool = True,
        maxpool1 : bool = True,
        latent_dim = 64,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._create_network()
        # loss function
        self.criterion = torch.nn.MSELoss(reduction ="mean")
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = r2score
        self.val_accuracy = r2score
        self.test_accuracy = r2score
        self.augmentation = DataAugmentation()

    def _create_network(self):
        # target space dimension
        target_space_dimension = 6
        # re-organise the initial CNN 

        feature_encoder = deformation_encoder(self.hparams.first_conv, self.hparams.maxpool1)
        num_ftrs = feature_encoder.ftrs
        self.model = nn.Sequential(feature_encoder,
                                #    nn.Dropout(self.hparams.drop_out),
                                   nn.Linear(num_ftrs, self.hparams.latent_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hparams.latent_dim,target_space_dimension))
  
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any, automentation: bool = False):
        x, y = batch

        if self.hparams.data_augmentation == True:
            if automentation == True:
                x = self.augmentation(x)

        y_hat = self.forward(x)
        y = torch.torch.squeeze(y)
        # different weight about force and torque
        loss_force = self.criterion(y_hat[:,0:3],y[:,0:3])
        loss_torque = self.criterion(y_hat[:,3:6],y[:,3:6])
        den = 1 + self.hparams.relative_f_t
        loss = (1/den) * loss_force + (self.hparams.relative_f_t / den) * loss_torque/1000.0 

        preds = y_hat
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch,True)
  

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        acc_force = self.train_accuracy(preds[:,0:3], targets[:,0:3])
        acc_torque = self.train_accuracy(preds[:,3:6], targets[:,3:6])

        self.log("train/acc_force", acc_force, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_torque", acc_torque, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)


        acc_force = self.train_accuracy(preds[:,0:3], targets[:,0:3])
        acc_torque = self.train_accuracy(preds[:,3:6], targets[:,3:6])

        self.log("val/acc_force", acc_force, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_torque", acc_torque, on_step=False, on_epoch=True, prog_bar=True)



        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        
        acc_force = self.train_accuracy(preds[:,0:3], targets[:,0:3])
        acc_torque = self.train_accuracy(preds[:,3:6], targets[:,3:6])

        self.log("val/acc_force", acc_force, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_torque", acc_torque, on_step=False, on_epoch=True, prog_bar=True)


        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, 
            # weight_decay=self.hparams.weight_decay
        )


if __name__ == "__main__":
    a = VisualFingerForceNet(data_augmentation=True)
    print(a)