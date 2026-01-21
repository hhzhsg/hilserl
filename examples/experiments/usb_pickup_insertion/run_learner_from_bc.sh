#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export PYTHONPATH=$PYTHONPATH:/home/user/hzh/hil-serl/serl_robot_infra

# ========== 配置 ==========
checkpoint_path="/mnt/satadisk2/dataset/hil-serl/init"

bc_checkpoint_path="/mnt/satadisk2/dataset/hil-serl/bc"

demo_path="/home/user/hzh/hil-serl/examples/experiments/usb_pickup_insertion/demo_data/usb_pickup_insertion_20_demos_2026-01-20_20-02-25.pkl"

python ../../train_rlpd_from_bc.py \
    --exp_name usb_pickup_insertion \
    --checkpoint_path $checkpoint_path \
    --bc_checkpoint_path $bc_checkpoint_path \
    --demo_path $demo_path \
    --learner
