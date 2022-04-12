import torch
import torch.nn as nn

import cv2
import numpy as np

"""
reference by https://github.com/bonlime/pytorch-tools
"""

class GANLoss(nn.Module):
    """
    Pytorch module for GAN Loss
    from https://github.com/S-aiueo32/sr-pytorch-lightning/blob/master/models/losses.py
    """
    def __init__(self, gan_mode='wgangp', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # not updated by optimizer
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        if gan_mode == 'mse':
            self.loss = nn.MSELoss()
        elif gan_mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' %(gan_mode))

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).detach()

    def forward(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss =  -1 * prediction.mean() 
            else:
                loss = prediction.mean()
        return loss

class PSNR(nn.Module):
    """
    Peak Signal/Noise Ratio
    img1 and img2 have range [0, 255]
    """
    def __init__(self, max_val=1.):
        super(PSNR, self).__init__()
        self.max_val = max_val

    def forward(self, predictions, targets):
        mse  = torch.mean((predictions - targets) ** 2.)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr

class SSIM:
    """
    Structure Similarity
    img1, img2: [0, 255]
    """
    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:    # Greyscale or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions")

    @staticmethod
    def _ssim(img1, img2):
       C1 = (0.01 * 255) ** 2.
       C2 = (0.03 * 255) ** 2.
 
       img1 = img1.astype(np.float64)
       img2 = img2.astype(np.float64)
       kernel = cv2.getGaussianKernel(11, 1.5)
       window = np.outer(kernel, kernel.transpose())

       mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
       mu2 = cv2.fliter2D(img2, -1, window)[5:-5, 5:-5]
       mu1_sq = mu1 ** 2.
       mu2_sq = mu2 ** 2.
       mu1_mu2 = mu1 * mu2
       sigma1_sq = cv2.filter2D(img1 ** 2., -1, window)[5:-5, 5:-5] - mu1_sq
       sigma2_sq = cv2.filter2D(img2 ** 2., -1, window)[5:-5, 5:-5] - mu2_sq
       sigma12   = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

       ssim_map  = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
       return ssim_map.mean()
