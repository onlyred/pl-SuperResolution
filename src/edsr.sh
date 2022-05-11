#!/bin/sh

### EDSR
dataroot="../dataset/Flickr1024"
logdir="./logs"
scale_factor=4
bs=16
gpus=0
seed=0
loss='psnr'
patch_size=250
epochs=3000
in_channel=3
h_dims=64
model='edsr'
h_dims=64
n_blocks=16
res_scale=1.0

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
                --n_blocks ${n_blocks} \
                --res_scale ${res_scale} \
                --logdir ${logdir}
