#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>train_split100_reduced_resnet50_esm2_2560_combine_o256_triplet.log 2>&1

python train-triplet.py \
	--training_data split100_reduced \
	--model_name split100_reduced_resnet50_esm2_2560_combine_o256_triplet \
	--out_dim 256 \
	--epoch 7000
