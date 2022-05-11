import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.utils import make_grid, save_image

from pathlib import Path
import argparse
import math

from .datasets import DatasetFromFolder
from .losses import PSNR, VGG19FeatureExtractor
from kornia.losses import SSIMLoss, PSNRLoss
from kornia.color import rgb_to_grayscale
from .networks import SRCNN, FSRCNN, EDSR, Generator, Discriminator

class SR_Model(pl.LightningModule):
    def __init__(self, opt):
        super(SR_Model,self).__init__()
        self.save_hyperparameters()  # for checkpoint

        self.dataroot     = Path(opt.dataroot)
        self.scale_factor = opt.scale_factor
        self.batch_size   = opt.batch_size
        self.n_worker     = opt.n_worker
        self.model        = opt.model
        self.in_channels  = opt.in_channels
        self.patch_size   = opt.patch_size
        self.upsample     = opt.upsample
        self.epochs       = opt.epochs

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
            'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor
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

class SRCNN_Model(SR_Model):
    def __init__(self, opt):
        super(SRCNN_Model,self).__init__(opt)
        self.h_dims = opt.h_dims
        self.net = SRCNN(self.in_channels, self.h_dims)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataroot', type=str, required=True)
        parser.add_argument('--loss', choices=['psnr'], help='select losses', required=True)
        parser.add_argument('--in_channels', type=int, help='number of input channel', default=3)
        parser.add_argument('--h_dims', type=int, help='hidden dimmensions', default=64)
        parser.add_argument('--upsample', default=False, help='pre-upsample', action='store_true')
        parser.add_argument('--patch_size', type=int, default=96, help='patch size for cropping')
        parser.add_argument('--n_worker', type=int, help='num_workers in dataloader', default=8)
        parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        return parser

class FSRCNN_Model(SR_Model):
    def __init__(self, opt):
        super(FSRCNN_Model,self).__init__(opt)
        self.s_dims = opt.s_dims
        self.n_map  = opt.n_map
        self.h_dims = opt.h_dims
        self.net = FSRCNN(self.scale_factor, self.in_channels, 
                          self.h_dims, self.s_dims, self.n_map )
     
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataroot', type=str, required=True)
        parser.add_argument('--loss', choices=['psnr'], help='select losses', required=True)
        parser.add_argument('--in_channels', type=int, help='number of input channel', default=3)
        parser.add_argument('--h_dims', type=int, help='hidden dimmensions', default=64)
        parser.add_argument('--upsample', default=False, help='pre-upsample', action='store_true')
        parser.add_argument('--patch_size', type=int, default=96, help='patch size for cropping')
        parser.add_argument('--n_worker', type=int, help='num_workers in dataloader', default=8)
        parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        parser.add_argument('--s_dims', type=int, help='shrink dimmensions', default=12)
        parser.add_argument('--n_map', type=int, help='number of map layers', default=4)
        return parser

class EDSR_Model(SR_Model):
    def __init__(self, opt):
        super(EDSR_Model,self).__init__(opt)
        self.n_blocks = opt.n_blocks
        self.h_dims   = opt.h_dims
        self.res_scale= opt.res_scale

        self.net = EDSR(self.scale_factor, self.in_channels, 
                        self.h_dims, self.n_blocks, self.res_scale )
     
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataroot', type=str, required=True)
        parser.add_argument('--loss', choices=['psnr'], help='select losses', required=True)
        parser.add_argument('--in_channels', type=int, help='number of input channel', default=3)
        parser.add_argument('--h_dims', type=int, help='hidden dimmensions', default=64)
        parser.add_argument('--upsample', default=False, help='pre-upsample', action='store_true')
        parser.add_argument('--patch_size', type=int, default=96, help='patch size for cropping')
        parser.add_argument('--n_worker', type=int, help='num_workers in dataloader', default=8)
        parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        parser.add_argument('--n_blocks', type=int, help='number of resblocks', default=16)
        parser.add_argument('--res_scale', type=float, help='scale of resblocks', default=1.0)
        return parser

class SRGAN_Model(SR_Model):
    def __init__(self, opt):
        super(SRGAN_Model, self).__init__(opt)
        self.net_G = Generator(self.scale_factor)
        self.net_D = Discriminator()
        # For training
        self.vgg_feature_extractor = VGG19FeatureExtractor(self.in_channels)
        # For validation
        self.criterion = nn.MSELoss()
        self.criterion_PSNR= PSNRLoss(1.)
        self.criterion_SSIM= SSIMLoss(window_size=11, reduction='mean')
    
    def forward(self, x):
        return self.net_G(x)

    def training_step(self, batch, batch_nb, optimizer_idx):
        img_lr = batch['lr']  # [0,1]
        img_hr = batch['hr']  # [0,1]
      
        if optimizer_idx == 0:  # discriminator
            loss = self._disc_loss(img_hr, img_lr)
            self.log('loss/disc', loss, prog_bar=True, on_step=True, on_epoch=True)
        elif optimizer_idx == 1:  # generator
            loss = self._gen_loss(img_hr, img_lr)
            self.log('loss/gen', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _disc_loss(self, img_hr: torch.Tensor, img_lr: torch.Tensor):
        real_pred = self.net_D(img_hr)
        real_loss = self._adv_loss(real_pred, ones=True)
        
        _, fake_pred = self._fake_pred(img_lr)
        fake_loss = self._adv_loss(fake_pred, ones=False)
    
        disc_loss = 0.5 * (real_loss + fake_loss)
        return disc_loss
       
    def _gen_loss(self, img_hr: torch.Tensor, img_lr: torch.Tensor):
        img_sr, fake_pred = self._fake_pred(img_lr)
    
        perceptual_loss = self._perceptual_loss(img_hr, img_sr)
        adv_loss = self._adv_loss(fake_pred, ones=True)
        content_loss = self._content_loss(img_hr, img_sr)
        g_loss = 0.006 * perceptual_loss + 0.001 * adv_loss + content_loss
        return g_loss

    def _fake_pred(self, img_lr: torch.Tensor):
        img_sr = self(img_lr)
        fake_pred = self.net_D(img_sr)
        return img_sr, fake_pred

    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool):
        bce_loss = nn.BCELoss()
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        adv_loss = bce_loss(pred, target)
        return adv_loss

    def _perceptual_loss(self, img_hr: torch.Tensor, img_sr: torch.Tensor):
        real_features = self.vgg_feature_extractor(img_hr)
        fake_features = self.vgg_feature_extractor(img_sr)
        perceptual_loss = self._content_loss(real_features, fake_features)
        return perceptual_loss

    @staticmethod
    def _content_loss(img_hr: torch.Tensor, img_sr: torch.Tensor):
        return nn.functional.mse_loss(img_hr, img_sr)

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            img_lr = batch['lr']
            img_hr = batch['hr']
            img_sr = self.forward(img_lr)

            #if self.current_epoch % 1 == 0:
            #    save_image(img_sr, './temp/sample_%03d_%03d.png' %(batch_nb, self.current_epoch))
            val_loss =  self.criterion(img_sr, img_hr)
            psnr_loss = self.criterion_PSNR(img_sr, img_hr)  # loss (negative)
            ssim_loss = self.criterion_SSIM(img_sr, img_hr) 
            self.log('val_loss',  val_loss, prog_bar=True)
            psnr = -1 * psnr_loss
            ssim = 1 - ssim_loss
            self.log('psnr',psnr, prog_bar=True)
            self.log('ssim',ssim, prog_bar=True)
        return {'val_loss': val_loss, 'psnr' : psnr, 'ssim' : ssim}

    def configure_optimizers(self):
        ds_len = int(self.train_dataloader().dataset.__len__())
        optimizer_D = optim.Adam(self.net_D.parameters(),lr=1e-4)
        optimizer_G = optim.Adam(self.net_G.parameters(),lr=1e-4)
        scheduler_D = optim.lr_scheduler.OneCycleLR(optimizer_D, max_lr=0.001,
                                                    steps_per_epoch=int(ds_len/self.batch_size),
                                                    epochs=self.epochs)
        scheduler_G = optim.lr_scheduler.OneCycleLR(optimizer_G, max_lr=0.001,
                                                    steps_per_epoch=int(ds_len/self.batch_size),
                                                    epochs=self.epochs)
        return [optimizer_D, optimizer_G], [scheduler_D, scheduler_G]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataroot', type=str, required=True)
        parser.add_argument('--loss', choices=['psnr'], help='select losses', default='psnr')
        parser.add_argument('--in_channels', type=int, help='number of input channel', default=3)
        parser.add_argument('--upsample', default=False, help='pre-upsample', action='store_true')
        parser.add_argument('--patch_size', type=int, default=96, help='patch size for cropping')
        parser.add_argument('--n_worker', type=int, help='num_workers in dataloader', default=4)
        parser.add_argument('--scale_factor', type=int, default=2, help='scale factor')
        parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
        return parser
