#!/bin/bash
exp_name="$1"
# run_cmd="python -u train_toy1.py \
run_cmd="python -u train_toy3.py \
    --exp_name=$exp_name \
    --dataset=toy \
    --data_dir=toy3_  \
    --data_seed=1 \
    --pgm_path=../../checkpoints/pgm_toy3/checkpoint.pt \
    --bs=10000 \
    --gpu=5 \
    --seed=2 \
    --logw_s=1 \
    --logw_c=0\
    --logw_n=9 \
    --natural_eps=1e-4 \
    --epoch_num=50001 \
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"
# --logw_n=11 \
# bash train_toy3.sh formal_ours_toy3_seed0_4 nohup
# bash train_toy3.sh formal_ours_toy3_seed1_4 nohup
# bash train_toy3.sh formal_ours_toy3_seed2_4 nohup

# --logw_n=9 \
# bash train_toy3.sh formal_ours_toy3_seed0_4_9 nohup
# bash train_toy3.sh formal_ours_toy3_seed1_4_9 nohup
# bash train_toy3.sh formal_ours_toy3_seed2_4_9 nohup

# 0-4 logw_n=15 --cf_lr=5.0
#0-42 logw_n=13 --cf_lr=5.0
    # --vae simple\
# --logw_t=1 \
#     --logw_i=0\
#     --logw_n=13 \
#     --natural_eps=1e-3 \ best for this
#     --epoch_num=50001 \
#     --cf_lr=5.0 \
#     --lr_step_size 10000 \
#     --lr_gamma 0.1 \





if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi



