#!/usr/bin/env bash
set -e

# Experiment 1: spatial-domain baseline
python train.py \
  --train_dir ../data_1um_bicubic/train \
  --val_dir ../data_1um_bicubic/val \
  --out_dir runs/spatial_only \
  --epochs 10 \
  --qualities 10 30 50 \
  --lambda_freq 0.0

# Experiment 2: spatial + DCT frequency loss
python train.py \
  --train_dir ../data_1um_bicubic/train \
  --val_dir ../data_1um_bicubic/val \
  --out_dir runs/spatial_plus_dct \
  --epochs 10 \
  --qualities 10 30 50 \
  --lambda_freq 0.1

# Evaluate generalization across JPEG quality factors
python evaluate.py \
  --checkpoint runs/spatial_only/best.pt \
  --data_dir ../data_1um_bicubic/test \
  --qualities 10 20 30 40 50

python evaluate.py \
  --checkpoint runs/spatial_plus_dct/best.pt \
  --data_dir ../data_1um_bicubic/test \
  --qualities 10 20 30 40 50
