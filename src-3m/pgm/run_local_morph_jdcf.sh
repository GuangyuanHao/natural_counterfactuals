#!/bin/bash
exp_name="$1"
run_cmd="python pgm_cfs.py \
    --exp_name=$exp_name \
    --hps morphomnist \
    --dataset=morphomnist \
    --data_dir=../../datasets/morphomnist \
    --load_path=../../checkpoints/t_i_d/ \
    --setup=sup_pgm \
    --input_res=32 \
    --pad=4 \
    --seed=0 \
    --context_dim=12 \
    --bs=32"
    # --vae simple \

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
# bash run_local_morph_jdcf.sh morph/jdcf nohup

# ours False simple-vae seed=0
# thickness: test | thickness_mae: 0.33490646 - intensity_mae: 4.51369381 - digit_acc: 0.97090000
# intensity: test | thickness_mae: 0.28264943 - intensity_mae: 6.61851883 - digit_acc: 0.99160000
