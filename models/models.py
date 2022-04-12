import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pathlib import Path
import argparse

from .datasets import DatasetFromFolder
from .losses import PSNR, SSIM
import models

class SR_Model(pl.LightningModule):
    def __init__(self, opt):
        super(SR_Model,self).__init__()
        self.save_hyperparameters()  # for checkpoint

        self.dataroot     = Path(opt.dataroot)
        self.scale_factor = opt.scale_factor
        self.batch_size   = opt.batch_size
        self.n_worker     = opt.n_worker
        self.model        = opt.model
        self.in_channel   = opt.in_channel
        self.patch_size   = opt.patch_size
        self.upsample     = opt.upsample

        if opt.loss.upper() == "PSNR":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError

    def forward(self, input):
        return self.net(input)

    def training_step(self, batch, batch_nb):
        img_lr = batch['lr']
        img_hr = batch['hr']
        img_sr = self.forward(img_lr)
        tr_loss= self.criterion(img_sr, img_hr)

        if self.training:
            self.log('tr_loss', tr_loss, prog_bar=True)
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        return tr_loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_tr_loss', avg_loss)

    def validation_step(self, batch, batch_nb):
        val_loss = self.training_step(batch, batch_nb)
        self.log('val_loss', val_loss)
        return val_loss

    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        
    def configure_optimizers(self):
        optimizer    = optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                     patience=5, factor=0.5, min_lr=1e-5, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'avg_val_loss', # Metric for ReduceLROnPlateau to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for LearningRateMonitor to use
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir="%s/Train" %(self.dataroot),
            scale_factor=self.scale_factor,
            patch_size=self.patch_size,
            preupsample=self.upsample
        )
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_worker)

    def val_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir="%s/Validation" %(self.dataroot),
            scale_factor=self.scale_factor,
            mode='eval',
            preupsample=self.upsample
        )
        return DataLoader(dataset, batch_size=1, num_workers=self.n_worker)
       
    def test_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir="%s/Test" %(self.dataroot),
            scale_factor=self.scale_factor,
            mode='eval',
            preupsample=self.upsample
        )
        return DataLoader(dataset, batch_size=1, num_workers=self.n_worker)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataroot', type=str, required=True)
        parser.add_argument('--loss', choices=['psnr'], help='select losses', required=True)
        parser.add_argument('--in_channel', type=int, help='number of input channel', default=3)
        parser.add_argument('--h_dims', type=int, help='hidden dimmensions', default=64)
        parser.add_argument('--upsample', default=False, help='pre-upsample', action='store_true')
        parser.add_argument('--patch_size', type=int, default=96, help='patch size for cropping')
        parser.add_argument('--n_worker', type=int, help='num_workers in dataloader', default=8)
        parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        return parser

class SRCNN_Model(SR_Model):
    def __init__(self, opt):
        super(SRCNN_Model,self).__init__(opt)
        self.h_dims = opt.h_dims

        self._define_model()
        
    def _define_model(self):
        if self.model == 'srcnn':
            self.net = models.SRCNN(self.in_channel, self.h_dims)
        else:
            raise NotImplementedError

class FSRCNN_Model(SR_Model):
    def __init__(self, opt):
        super(FSRCNN_Model,self).__init__(opt)
        self.s_dims = opt.s_dims
        self.n_map  = opt.n_map
        self.h_dims = opt.h_dims

        self._define_model()

    def _define_model(self):
        if self.model == 'fsrcnn':
            self.net = models.FSRCNN(self.scale_factor, self.in_channel, 
                                     self.h_dims, self.s_dims, self.n_map )
        else:
            raise NotImplementedError
     
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataroot', type=str, required=True)
        parser.add_argument('--loss', choices=['psnr'], help='select losses', required=True)
        parser.add_argument('--in_channel', type=int, help='number of input channel', default=3)
        parser.add_argument('--h_dims', type=int, help='hidden dimmensions', default=64)
        parser.add_argument('--upsample', default=False, help='pre-upsample', action='store_true')
        parser.add_argument('--patch_size', type=int, default=96, help='patch size for cropping')
        parser.add_argument('--n_worker', type=int, help='num_workers in dataloader', default=8)
        parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        parser.add_argument('--s_dims', type=int, help='shrink dimmensions', default=12)
        parser.add_argument('--n_map', type=int, help='number of map layers', default=4)
        return parser

