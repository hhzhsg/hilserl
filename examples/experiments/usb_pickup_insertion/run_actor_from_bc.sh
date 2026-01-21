#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export PYTHONPATH=$PYTHONPATH:/home/user/hzh/hil-serl/serl_robot_infra

# ========== 配置 ==========
checkpoint_path="/mnt/satadisk2/dataset/hil-serl/init"

bc_checkpoint_path="/home/user/hzh/hil-serl/examples/experiments/usb_pickup_insertion/bc_checkpoint"

# python ../../train_rlpd_from_bc.py \
#     --exp_name usb_pickup_insertion \
#     --checkpoint_path $checkpoint_path \
#     --bc_checkpoint_path $bc_checkpoint_path \
#     --actor

# ========== 评估模式 ==========
# 取消下面的注释来评估指定 checkpoint
python ../../train_rlpd_from_bc.py \
    --exp_name usb_pickup_insertion \
    --checkpoint_path $checkpoint_path \
    --actor \
    --eval_checkpoint_step 34000 \
    --eval_n_trajs 10
