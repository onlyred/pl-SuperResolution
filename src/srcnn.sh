#!/bin/sh

### SRCNN
dataroot="../dataset/Flickr1024"
logdir="./logs"
model='srcnn'
scale_factor=4
bs=16
gpus=1
seed=0
loss='psnr'
patch_size=250
epochs=3000
in_channel=3
h_dims=64
python train.py --model ${model} \
                --dataroot ${dataroot} \
                --loss ${loss} \
                --batch_size ${bs} \
                --gpus ${gpus} \
                --seed ${seed} \
                --patch_size ${patch_size} \
                --epochs ${epochs} \
                --in_channel ${in_channel} \
                --h_dims ${h_dims} \
                --logdir ${logdir} \
                --upsample
