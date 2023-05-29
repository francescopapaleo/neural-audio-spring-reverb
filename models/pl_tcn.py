# Base class for pytorch lightning

import torch
import auraloss
import pytorch_lightning as pl

from utils.utils import center_crop, causal_crop
from argparse import ArgumentParser

class Base(pl.LightningModule):
    '''Base module for TCN models.
    '''
    def __init__(self,
                 learning_rate=1e-3,                 
                 n_inputs=1, 
                 n_outputs=1, 
                 n_blocks=10, 
                 kernel_size=13, 
                 n_channels=64, 
                 dilation_growth=4, 
                 cond_dim=0,
                 num_examples = 1,
                 **kwargs
                 ):
        super(Base, self).__init__()

        # added to mitigate "can't pickle _thread.lock objects" problem
        # kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, types.ModuleType)}
        print(self.hparams)
        self.save_hyperparameters()

        self.esr    = auraloss.time.ESRLoss()
        self.dc     = auraloss.time.DCLoss()
        self.stft   = auraloss.freq.STFTLoss()
        self.learning_rate = learning_rate
        self.optimizer_name = "Adam"
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.aggregate_loss = 0

    def forward(self, x, p):
       return self.model(x, p)

    def training_step(self, batch, batch_idx):
        input, target = batch

        pred = self(input)

        # crop the input and target signals
        if self.hparams.causal:
            input_crop = causal_crop(input, pred.shape[-1])
            target_crop = causal_crop(target, pred.shape[-1])
        else:
            input_crop = center_crop(input, pred.shape[-1])
            target_crop = center_crop(target, pred.shape[-1])
        
        # compute the loss
        if self.hparams.train_loss == "dc":
            loss = self.dc(pred, target)
        elif self.hparams.train_loss == "stft":
            loss = self.stft(pred, target)
        elif self.hparams.train_loss == "esr":
            loss = self.esr(pred, target)
        elif self.hparams.train_loss == "dc+stft":
            loss = self.dc(pred, target) + self.stft(pred, target)
        else:
            raise NotImplementedError(f"Invalid loss fn: {self.hparams.train_loss}")
        
        self.log("step_train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.training_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("epoch_train_loss", epoch_mean)
        return epoch_mean
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        pred = self(input)

        if self.hparams.causal:
            input_crop = causal_crop(input, pred.shape[-1])
            target_crop = causal_crop(target, pred.shape[-1])
        else:
            input_crop = center_crop(input, pred.shape[-1])
            target_crop = center_crop(target, pred.shape[-1])

        esr_loss = self.esr(pred, target_crop)
        dc_loss = self.dc(pred, target)
        stft_loss = self.stft(pred, target)
        val_loss = (esr_loss)
        
        self.log_dict({"val_loss": val_loss})
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=4, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": 'val_loss'
        }
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- training related ---
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--train_loss', type=str, default="dc+stft")
        # --- vadliation related ---
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--num_examples', type=int, default=4)

        return parser