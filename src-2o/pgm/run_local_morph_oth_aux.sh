#!/bin/bash
# exp_name="$1"
# run_cmd="python train_pgm.py \
#     --exp_name=$exp_name \
#     --dataset=morphomnist \
#     --data_dir=../../datasets/morphomnist \
#     --setup=sup_aux \
#     --input_res=32 \
#     --pad=4 \
#     --hflip=0 \
#     --context_dim=12 \
#     --lr=0.001 \
#     --bs=32 \
#     --wd=0.01 \
#     --eval_freq=4 \
#     --epochs 160"



# if [ "$2" = "nohup" ]
# then
#   nohup ${run_cmd} > $exp_name.out 2>&1 &
#   echo "Started training in background with nohup, PID: $!"
# else
#   ${run_cmd}
# fi
# bash run_local_morph_oth_aux.sh morph_other_aux nohup


# testing

run_cmd="python train_pgm.py \
    --dataset=morphomnist \
    --data_dir=../../datasets/morphomnist \
    --load_path=../../checkpoints/t_i_d/morph_other_aux/checkpoint.pt \
    --setup=sup_aux \
    --testing \
    --input_res=32 \
    --pad=4 \
    --hflip=0 \
    --context_dim=12"


exp_name="$1"
if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi

# bash run_local_morph_oth_aux.sh morph_other_aux_test nohup

# Evaluating test set:

#  => eval | loss: -4.5742, logp(thickness_aux): 2.1276, logp(intensity_aux): 2.4622, logp(digit_aux): -0.0157: 100%|███████████████████████████████████████████████████████| 313/313 [00:05<00:00, 52.58it/s]
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 313/313 [00:02<00:00, 156.08it/s]
# test | thickness_mae: 0.0628 - intensity_mae: 1.6203 - digit_acc: 0.9945