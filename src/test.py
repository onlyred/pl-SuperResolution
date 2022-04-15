import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import pandas as pd

import torch
from torchvision.utils import save_image
from kornia.losses import PSNRLoss, SSIMLoss
from kornia.color import rgb_to_grayscale

import models

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['srcnn','fsrcnn','edsr'],required=True)
    parser.add_argument('--ckpt', type=str,required=True)
    opt = parser.parse_args()
    return opt

def main():
    opt = get_arguments()

    # load model classes
    if opt.model == 'srcnn':
        Model = models.SRCNN_Model
    elif opt.model == 'fsrcnn':
        Model = models.FSRCNN_Model
    elif opt.model == 'edsr':
        Model = models.EDSR_Model
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

    model = Model.load_from_checkpoint( opt.ckpt ) # save_hyperparameters()
    model.to(device)
    model.eval()
    model.freeze()

    save_dir = Path(opt.ckpt)
    save_dir = save_dir.with_name(Path(opt.ckpt).stem.replace("_ckpt_",''))
    save_dir.mkdir(exist_ok=True)
 
    criterion_PSNR = PSNRLoss(1.)
    criterion_SSIM = SSIMLoss(window_size=11, reduction='mean')

    csv = defaultdict(list)

    psnr_mean = 0
    ssim_mean = 0
    dataloader = model.test_dataloader()
    tbar = tqdm( dataloader )
    for batch in tbar:
        img_name = batch['path'][0]
        img_lr   = batch['lr'].to(device)
        img_hr   = batch['hr'].to(device)
        img_sr   = model(img_lr)
       
        img_hr_ = rgb_to_grayscale(img_hr)
        img_sr_ = rgb_to_grayscale(img_sr)
 
        psnr     = criterion_PSNR(img_hr_, img_sr_).item()
        ssim     = 1 - criterion_SSIM(img_hr_, img_sr_).item()
        csv['Name'].append(img_name)
        csv['PSNR'].append(-psnr)
        csv['SSIM'].append(ssim)
        
        psnr_mean += psnr
        ssim_mean += ssim

        save_image(img_sr, save_dir / f'{img_name}.png', nrow=1)
        
    psnr_mean /= len(dataloader)
    ssim_mean /= len(dataloader)
    print(f'PSNR: {psnr_mean:.4}, SSIM : {ssim_mean:.4}')
    df = pd.DataFrame(csv).round(3)
    df.to_csv(save_dir / 'stat.csv', index=False)

if __name__ == "__main__":
    main()
