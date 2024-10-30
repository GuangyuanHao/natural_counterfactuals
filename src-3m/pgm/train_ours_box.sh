#!/bin/bash
exp_name="$1"
run_cmd="python -u train_ours_box.py \
    --exp_name=$exp_name \
    --dataset=box \
    --csm_order prh \
    --parents_x p r h \
    --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/  \
    --pgm_path=../../checkpoints/p_r_h/box_pgm64_2/checkpoint.pt \
    --predictor_path=../../checkpoints/p_r_h/box_aux64/checkpoint.pt \
    --vae_path=../../checkpoints/p_r_h/box_hvae_prh/checkpoint.pt \
    --bs=1100 \
    --logw_s=1 \
    --logw_c=0\
    --gpu=3 \
    --seed=6 \
    --logw_n=11 \
    --natural_eps=1e-3 \
    --epoch_num=50001 \
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"
#840->1100
# bash train_ours_box.sh box/w_allc_3_11 nohup  0
# bash train_ours_box.sh box/w_allc_4_11 nohup  1
# bash train_ours_box.sh box/w_allc_5_11 nohup  2
# bash train_ours_box.sh box/w_allc_6_11 nohup  3

#
# bash train_ours_box.sh box/w_allc_0_11 nohup 
# bash train_ours_box.sh box/w_allc_1_11 nohup  
# bash train_ours_box.sh box/w_allc_2_11 nohup 


if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
