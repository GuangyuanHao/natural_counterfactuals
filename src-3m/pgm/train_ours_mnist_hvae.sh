#!/bin/bash
# vae
# exp_name="$1"
# run_cmd="python train_ours_mnist.py \
#     --exp_name=$exp_name \
#     --dataset=morphomnist \
#     --data_dir=../../datasets/morphomnist \
#     --pgm_path=../../checkpoints/t_i_d/morph_other/checkpoint.pt \
#     --predictor_path=../../checkpoints/t_i_d/morph_other_aux/checkpoint.pt \
#     --vae_path=../../checkpoints/t_i_d/morph_dscm_prior/checkpoint.pt \
#     --vae simple \
#     --bs=10000 \
#     --w_t=2 \
#     --gpu=1 \
#     --seed=2 \
#     --natural_eps=1e-2 \
#     --logw_n=11 \
#     --epoch_num=50001 \
#     --cf_lr=5.0 \
#     --lr_step_size 10000 \
#     --lr_gamma 0.1
# "
    # --natural_eps=1e-2 \
    # --logw_n=11 \

# bash train_ours_mnist.sh mnist/seed0_2_11 nohup
# bash train_ours_mnist.sh mnist/seed1_2_11 nohup
# bash train_ours_mnist.sh mnist/seed2_2_11 nohup

    # --natural_eps=1e-4 \
    # --logw_n=15 \
# bash train_ours_mnist.sh mnist/seed0_4_15 nohup
# bash train_ours_mnist.sh mnist/seed1_4_15 nohup
# bash train_ours_mnist.sh mnist/seed2_4_15 nohup



#50001
# bash train_ours_mnist.sh mnist/seed0_3_13 nohup
# bash train_ours_mnist.sh mnist/seed1_3_13 nohup
# bash train_ours_mnist.sh mnist/seed2_3_13 nohup







# hvae
exp_name="$1"
run_cmd="python train_ours_mnist.py \
    --exp_name=$exp_name \
    --dataset=morphomnist \
    --data_dir=../../datasets/morphomnist \
    --pgm_path=../../checkpoints/t_i_d/morph_other/checkpoint.pt \
    --predictor_path=../../checkpoints/t_i_d/morph_other_aux/checkpoint.pt \
    --vae_path=../../checkpoints/t_i_d/morph_hvae_prior160/checkpoint.pt \
    --bs=10000 \
    --logw_t=1 \
    --logw_i=0\
    --gpu=4 \
    --seed=2 \
    --natural_eps=1e-4 \
    --logw_n=15 \
    --epoch_num=50001 \
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"

    # --natural_eps=1e-3 \
    # --logw_n=13 \
#50001
# bash train_ours_mnist.sh mnist/h_seed0_3_13 nohup 6 
# bash train_ours_mnist.sh mnist/h_seed1_3_13 nohup 2 
# bash train_ours_mnist.sh mnist/h_seed2_3_13 nohup 3

    # --natural_eps=1e-4 \
    # --logw_n=15 \
# bash train_ours_mnist.sh mnist/h_seed0_4_15 nohup 5 
# bash train_ours_mnist.sh mnist/h_seed1_4_15 nohup 6
# bash train_ours_mnist.sh mnist/h_seed2_4_15 nohup 4

    # --natural_eps=1e-2 \
    # --logw_n=11 \

# bash train_ours_mnist.sh mnist/h_seed0_2_11 nohup 2
# bash train_ours_mnist.sh mnist/h_seed1_2_11 nohup 3
# bash train_ours_mnist.sh mnist/h_seed2_2_11 nohup 5













if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
