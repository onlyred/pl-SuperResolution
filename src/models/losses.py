from collections import namedtuple

import torch
import torch.nn as nn
import torchvision.models.vgg as vggs

import cv2
import numpy as np

"""
reference by https://github.com/bonlime/pytorch-tools
"""
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, image_channels: int = 3) -> None:
        super().__init__()

        assert image_channels in [1, 3]
        self.image_channels = image_channels

        vgg = vggs.vgg19(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.image_channels == 1:
            x = x.repeat(1, 3, 1, 1)

        return self.vgg(x)

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
