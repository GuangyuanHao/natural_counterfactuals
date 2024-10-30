#!/bin/bash
# exp_name="$1"
# run_cmd="python pgm_cfs.py \
#     --exp_name=$exp_name \
#     --ours \
#     --hps morphomnist \
#     --dataset=morphomnist \
#     --data_dir=../../datasets/morphomnist \
#     --load_path=../../checkpoints/t_i_d/ \
#     --setup=sup_pgm \
#     --input_res=32 \
#     --pad=4 \
#     --seed=0 \
#     --context_dim=12 \
#     --vae simple \
#     --bs=10000" #10000

# 

## hvae
exp_name="$1"
run_cmd="python pgm_cfs.py \
    --exp_name=$exp_name \
    --ours \
    --hps morphomnist \
    --dataset=morphomnist \
    --data_dir=../../datasets/morphomnist \
    --load_path=../../checkpoints/t_i_d/ \
    --setup=sup_pgm \
    --input_res=32 \
    --pad=4 \
    --seed=0 \
    --context_dim=12 \
    --bs=10000" #10000

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
# bash run_local_morph_ours.sh morph/ours_cut5w2 nohup
# bash run_local_morph_ours.sh morph/ours_hvae_cut5w2 nohup

