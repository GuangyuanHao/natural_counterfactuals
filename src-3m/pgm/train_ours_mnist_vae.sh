#!/bin/bash
# vae
exp_name="$1"
run_cmd="python train_ours_mnist.py \
    --exp_name=$exp_name \
    --dataset=morphomnist \
    --data_dir=../../datasets/morphomnist \
    --pgm_path=../../checkpoints/t_i_d/morph_other/checkpoint.pt \
    --predictor_path=../../checkpoints/t_i_d/morph_other_aux/checkpoint.pt \
    --vae_path=../../checkpoints/t_i_d/morph_dscm_prior/checkpoint.pt \
    --vae simple \
    --bs=10000 \
    --gpu=5 \
    --seed=2 \
    --logw_t=2 \
    --logw_i=0\
    --logw_n=13 \
    --natural_eps=1e-3 \
    --epoch_num=50001 \
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"

# bash train_ours_mnist_w.sh mnist/w_seed0_t2 nohup
# bash train_ours_mnist_w.sh mnist/w_seed1_t2 nohup
# bash train_ours_mnist_w.sh mnist/w_seed2_t2 nohup
# bash train_ours_mnist_w.sh mnist/w_seed0_t3 nohup
# bash train_ours_mnist_w.sh mnist/w_seed1_t3 nohup
# bash train_ours_mnist_w.sh mnist/w_seed2_t3 nohup


# hvae

# 
# --logw_n=11 \
#     --natural_eps=1e-2 \
# --logw_n=15 \
#     --natural_eps=1e-4 \

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




