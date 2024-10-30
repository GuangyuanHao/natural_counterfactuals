#!/bin/bash
exp_name="$1"
# run_cmd="python -u train_toy5.py \
run_cmd="python train_toy5.py \
    --exp_name=$exp_name \
    --dataset=toy \
    --data_dir=toy5_  \
    --data_seed=1 \
    --pgm_path=../../checkpoints/pgm_toy5/checkpoint.pt \
    --bs=10000 \
    --gpu=0 \
    --seed=2 \
    --logw_s=1 \
    --logw_c=0\
    --logw_n=9 \
    --natural_eps=1e-4 \
    --epoch_num=50001\
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"
# 50001

# -logw_n=9 \
# bash train_toy5.sh formal_ours_toy5_seed2_4_9 nohup
# bash train_toy5.sh formal_ours_toy5_seed1_4_9 nohup
# bash train_toy5.sh formal_ours_toy5_seed0_4_9 nohup


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
# lr=10
# bash train_toy5.sh formal_ours_toy5_2_seed0_3 nohup
# bash train_toy5.sh formal_ours_toy5_2_seed0_4_11 nohup
# --logw_n=11 \
# bash train_toy5.sh formal_ours_toy5_2_seed0_4_13 nohup
# --logw_n=13 \
    # --natural_eps=1e-4 \

# bash train_toy5.sh formal_ours_toy5_2_seed0_4_new_init nohup
    # --logw_n=14 \
    # --natural_eps=1e-4 \

#lr=5
# bash train_toy5.sh formal_ours_toy5_2_seed0_4_13_2 nohup
# --logw_n=13 \
