#!/bin/bash

# set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

if [ ! -d YCB_Video_toolbox ];then
    echo 'Downloading the YCB_Video_toolbox...'
    git clone https://github.com/yuxng/YCB_Video_toolbox.git
    cd YCB_Video_toolbox
    unzip results_PoseCNN_RSS2018.zip
    cd ..
    cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
fi

for class_name in `cat ./YCB_Video_toolbox/classes.txt`
do
    for i in 0048 0049 0050 0051 0052 0053 0054 0055 0056 0057 0058 0059
    do
        mkdir -p experiments/eval_result/ycb/${class_name}/${i}
    done
done

python3 ./tools/eval_ycb_testing_full.py --dataset_root ./datasets/ycb/YCB_Video_Dataset\
  --model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth\
  --refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth
