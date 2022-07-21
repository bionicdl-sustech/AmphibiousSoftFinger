from typing import Any, List

import torch
from torch import nn, Tensor
from torch.autograd import grad, grad_mode
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

class VisualForceVAE(LightningModule):
    """Standard VAE with Gaussian Prior and approx posterior.
    """
    def __init__(
        self,
        VAE_weight: float = 0.1,
        input_height: int = 224,
        first_conv: bool = True,
        maxpool1: bool = True,
        kl_coeff: float = 0.1,
        latent_dim: int = 64,
        lr: float = 1e-4,
        relative_f_t = 1.0,
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
        
        self.vae_weight = VAE_weight
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.target_dim = 6
        self.criterion = torch.nn.MSELoss(reduction ="mean")
        
        
        self.encoder = deformation_encoder(first_conv, maxpool1)
        self.decoder = deformation_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        self.regression_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.target_dim)
        )

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
        '''Reconstruct Lable X and Y'''
        return self.decoder(z), self.regression_block(mu)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        
        '''reconstruct label'''
        y_hat = self.regression_block(mu)
        
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        
        return z, self.decoder(z), y_hat, p, q

    def predict_force(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        
        return self.regression_block(mu)
        
    def predict_latent_code(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        p, q, z = self.sample(mu, log_var)
        return z
        
    def predict_image(self, z):
        return self.decoder(z)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, y_hat, p, q = self._run_step(x)

        '''image_recon_loss'''
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        '''label prediction loss'''
        y = torch.torch.squeeze(y)
        # different weight about force and torque
        force_pred_loss = self.criterion(y_hat[:,0:3],y[:,0:3])
        torque_pred_loss = self.criterion(y_hat[:,3:6],y[:,3:6])
        den = 1 + self.hparams.relative_f_t
        pred_loss = (1/den) * force_pred_loss + (self.hparams.relative_f_t / den) * torque_pred_loss/1000.0 
        
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + self.vae_weight * recon_loss/(1.0 + self.vae_weight) + pred_loss/(1.0 + self.vae_weight)

        logs = {
            "recon_loss": recon_loss,
            "pred_loss": pred_loss,
            "kl": kl,
            "loss": loss,
        }
        
        label = y
        prediction = y_hat

        return loss, logs , label, prediction

    def training_step(self, batch, batch_idx):
        loss, logs , label, prediction  = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)

        ## ADD
        train_accuracy = r2score    
        # log train metrics
        acc = train_accuracy(label.detach(), prediction.detach())
        self.log("train/acc", acc, on_step=False, on_epoch=True)

        acc_force = train_accuracy(label.detach()[:,0:3], prediction.detach()[:,0:3])
        acc_torque = train_accuracy(label.detach()[:,3:6], prediction.detach()[:,3:6])
        self.log("train/acc_force", acc_force, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_torque", acc_torque, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs , label, prediction  = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

        ## ADD
        val_accuracy = r2score    
        # log train metrics
        acc = val_accuracy(label.detach(), prediction.detach())
        self.log("val/acc", acc, on_step=False, on_epoch=True)

        acc_force = val_accuracy(label.detach()[:,0:3], prediction.detach()[:,0:3])
        acc_torque = val_accuracy(label.detach()[:,3:6], prediction.detach()[:,3:6])
        self.log("val/acc_force", acc_force, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_torque", acc_torque, on_step=False, on_epoch=True, prog_bar=True)

        return loss
        
    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        '''label force'''
        y = torch.torch.squeeze(y)
        '''predict force'''
        y_hat = self.predict_force(x)

        test_accuracy = r2score    
        # log test metrics
        acc = test_accuracy(y, y_hat)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        

        return {"acc": acc}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    a = VisualForceVAE(VAE_weight = 1, latent_dim = 32, lr = 0.000001)
    print(a)
