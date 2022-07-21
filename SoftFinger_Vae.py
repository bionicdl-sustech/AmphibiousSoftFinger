import torch
import pytorch_lightning as pl
from torchvision import models
from src.datamodules.SoftFinger_datamodule import SoftFingerDataModule

from src.models import VisualFingerVAE
from src.callbacks.printing_callback import MyPrintingCallback, GenerateCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    # VAE Learning for Soft Finger
    
    ## Training Data
    dm = SoftFingerDataModule(data_type="InAir" , num_workers = 8, batch_size = 64,)
    dm.setup()

    ## Training Pipeline
    trainer = pl.Trainer(max_epochs = 150,gpus = [0],callbacks=[ModelCheckpoint(
        save_weights_only=True,),
        GenerateCallback(dm.get_train_images(4), every_n_epochs=1),
        LearningRateMonitor("epoch")],)

    ## Model Parameter Explanation: 
    #  kl_coeff, coresponds to beta in the paper
    #  latent_dim, defines the dimension of the latent space
    model = VisualFingerVAE(kl_coeff = 0.1, latent_dim = 32)

    ## Model Training
    trainer.fit(model, dm)

    ## Model Evaluation
    model.eval()        
    trainer.test(model,dm)


