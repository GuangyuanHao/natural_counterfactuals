#!/bin/bash
# exp_name="$1"
# run_cmd="python train_pgm.py \
#     --exp_name=$exp_name \
#     --dataset=morphomnist \
#     --data_dir=../../datasets/morphomnist \
#     --setup=sup_pgm \
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

# bash run_local_morph_oth.sh morph_other nohup

# testing
run_cmd="python train_pgm.py \
    --dataset=morphomnist \
    --data_dir=../../datasets/morphomnist \
    --load_path=../../checkpoints/t_i_d/morph_other/checkpoint.pt \
    --setup=sup_pgm \
    --testing \
    --input_res=32 \
    --pad=4 \
    --hflip=0 \
    --context_dim=12 \
    --lr=0.001 \
    --bs=32 \
    --wd=0.01 \
    --eval_freq=4 \
    --epochs 160"


exp_name="$1"
if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi

# bash run_local_morph_oth.sh morph_other_test nohup

# Loading checkpoint: ../../checkpoints/t_i_d/morph_other/checkpoint.pt
# thickness normalization: [-1,1]
# max: 6.255515, min: 0.87598526
# intensity normalization: [-1,1]
# max: 254.90317, min: 66.601204
# #samples: 60000

# thickness normalization: [-1,1]
# max: 6.255515, min: 0.87598526
# intensity normalization: [-1,1]
# max: 254.90317, min: 66.601204
# #samples: 10000

# thickness normalization: [-1,1]
# max: 6.255515, min: 0.87598526
# intensity normalization: [-1,1]
# max: 254.90317, min: 66.601204
# #samples: 10000

# Evaluating test set:

#  => eval | loss: 1.8936, logp(digit): -2.3010, logp(thickness): 0.0605, logp(intensity): 0.3469