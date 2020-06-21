#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_on_image.py --path $1 --object_id=$2\
  --width 640 --height 480 --cam_fx 1066.778 --cam_fy 1067.487 --cam_cx 312.9869 --cam_cy 241.3109\
  --model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth\
  --refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth
