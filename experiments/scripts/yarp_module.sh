#!/bin/bash

export PYTHONUNBUFFERED="True"
export YARP_CLOCK=/clock

python3 ./tools/yarp_module.py  --object_id=$1 --gpu_id 0 --fps 30\
  --width 640 --height 480 --cam_fx 686.2422145630587 --cam_fy 686.2422145630587 --cam_cx 320.0 --cam_cy 240.0\
  --model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth\
  --refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth
