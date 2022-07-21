import torch
import pytorch_lightning as pl
from torchvision import models
from src.datamodules.SoftFinger_datamodule import SoftFingerDataModule

from src.models import VisualFingerForceNet
from src.callbacks.printing_callback import MyPrintingCallback, GenerateCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    # CNN regression Learning for Soft Finger
    
    ## Training Data
    dm = SoftFingerDataModule(data_type="InAir" , num_workers = 8, batch_size = 64,)
    dm.setup()
    
    ## Training Pipeline
    trainer = pl.Trainer(max_epochs = 300,gpus = [1],callbacks=[ModelCheckpoint(
        save_weights_only=True,
    ), LearningRateMonitor("epoch")],)

    ## Model Parameter Explanation: 
    #  latent_dim, defines the dimension of the latent space
    #  Add data_augmentation for generalization
    model = VisualFingerForceNet(latent_dim=32, data_augmentation=False)
    
    ## Model Training
    trainer.fit(model, dm)

    ## Model Evaluation
    model.eval()        
    trainer.test(model,dm)
