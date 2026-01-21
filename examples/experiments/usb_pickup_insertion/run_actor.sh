#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export PYTHONPATH=$PYTHONPATH:/home/user/hzh/hil-serl/serl_robot_infra

# ========== 需要修改的部分 ==========
checkpoint_path="/mnt/satadisk2/dataset/hil-serl/init"

python ../../train_rlpd.py \
    --exp_name usb_pickup_insertion \
    --checkpoint_path $checkpoint_path \
    --actor \
    --eval_checkpoint_step 10000 \
    --eval_n_trajs 10