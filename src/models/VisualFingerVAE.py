from typing import Any, List
import torch
from torch import nn, Tensor
from torchvision import transforms
from pytorch_lightning import LightningModule
from torchmetrics.functional.regression.r2score import r2score
from torchvision import models

import urllib.parse
from argparse import ArgumentParser

from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F

import pytorch_lightning as pl

from .modules.Encoder import deformation_encoder
from .modules.Decoder import deformation_decoder

class VisualFingerVAE(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    """
    def __init__(
        self,
        input_height: int = 224,
        first_conv: bool = True,
        maxpool1: bool = True,
        kl_coeff: float = 0.1,
        latent_dim: int = 64,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.latent_dim = latent_dim
        self.input_height = input_height

        self.encoder = deformation_encoder(first_conv, maxpool1)
        self.decoder = deformation_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc_mu = nn.Linear(self.encoder.ftrs, self.latent_dim)
        self.fc_var = nn.Linear(self.encoder.ftrs, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        pass

    def from_pretrained(self, checkpoint_name):
        pass

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    a = VisualFingerVAE(first_conv = False, maxpool1 = False, kl_coeff = 0.1, latent_dim = 6)
    print(a)