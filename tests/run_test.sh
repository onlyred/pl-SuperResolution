#!/bin/sh

model='fsrcnn'

#ckpt_path="./logs/${model}/1qexsu5k"
ckpt_path="./logs/${model}/795xgiwe"
ckpt=`ls ${ckpt_path}/*.ckpt`


python test.py --model ${model} \
               --ckpt ${ckpt}

                 
                  
