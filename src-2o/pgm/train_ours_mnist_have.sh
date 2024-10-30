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
#     --logw_t=1 \
#     --logw_i=0\
#     --gpu=0 \
#     --seed=2 \
#     --natural_eps=1e-4 \
#     --logw_n=14 \
#     --epoch_num=50001 \
#     --cf_lr=5.0 \
#     --lr_step_size 10000 \
#     --lr_gamma 0.1
# "
    # --seed=0 \
    # --natural_eps=1e-4 \
    # --logw_n=14 \
# bash train_ours_mnist.sh mnist/seed0_4_14 nohup
# bash train_ours_mnist.sh mnist/seed1_4_14 nohup
# bash train_ours_mnist.sh mnist/seed2_4_14 nohup

    # --seed=0 \
    # --natural_eps=1e-2 \
    # --logw_n=10 \

# bash train_ours_mnist.sh mnist/seed0_2_10 nohup
# bash train_ours_mnist.sh mnist/seed1_2_10 nohup
# bash train_ours_mnist.sh mnist/seed2_2_10 nohup

# bash train_ours_mnist.sh mnist/seed0_3_12 nohup
# bash train_ours_mnist.sh mnist/seed1_3_12 nohup
# bash train_ours_mnist.sh mnist/seed2_3_12 nohup

##########

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
    --gpu=1 \
    --seed=1 \
    --natural_eps=1e-4 \
    --logw_n=14 \
    --epoch_num=50001 \
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"

# bash train_ours_mnist.sh mnist/h_seed0_4_14 nohup
# bash train_ours_mnist.sh mnist/h_seed1_4_14 nohup
# bash train_ours_mnist.sh mnist/h_seed2_4_14 nohup

    # --natural_eps=1e-3 \
    # --logw_n=12 \
# bash train_ours_mnist.sh mnist/h_seed0_3_12 nohup
# bash train_ours_mnist.sh mnist/h_seed1_3_12 nohup
# bash train_ours_mnist.sh mnist/h_seed2_3_12 nohup

    # --natural_eps=1e-2 \
    # --logw_n=10 \
# bash train_ours_mnist.sh mnist/h_seed0_2_10 nohup
# bash train_ours_mnist.sh mnist/h_seed1_2_10 nohup
# bash train_ours_mnist.sh mnist/h_seed2_2_10 nohup









if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi


