#!/bin/sh

model='srgan'

ckpt_path="./logs/${model}/2hnq243w"
ckpt=`ls ${ckpt_path}/*.ckpt`


python test.py --model ${model} \
               --ckpt ${ckpt}

                 
                  
