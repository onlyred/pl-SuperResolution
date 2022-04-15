#!/bin/sh

model='edsr'

#ckpt_path="./logs/${model}/1qexsu5k"
#ckpt_path="./logs/${model}/795xgiwe"
ckpt_path="./logs/${model}/1pu5q60r"
ckpt=`ls ${ckpt_path}/*.ckpt`


python test.py --model ${model} \
               --ckpt ${ckpt}

                 
                  
