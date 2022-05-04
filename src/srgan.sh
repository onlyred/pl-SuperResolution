#!/bin/sh

### SRCNN
dataroot="${HOME}/onlyred/deepSR/dataset/Flickr1024"
logdir="./logs"
model='srgan'
scale_factor=4
bs=16
gpus=1
seed=0
patch_size=250
epochs=3000
in_channel=3
python train.py --model ${model} \
                --dataroot ${dataroot} \
                --batch_size ${bs} \
                --gpus ${gpus} \
                --seed ${seed} \
                --patch_size ${patch_size} \
                --epochs ${epochs} \
                --in_channel ${in_channel} \
                --logdir ${logdir} 
