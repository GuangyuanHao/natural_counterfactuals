#!/bin/bash
exp_name="$1"
run_cmd="python train_ours_box.py \
    --exp_name=$exp_name \
    --dataset=box \
    --csm_order prh \
    --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/  \
    --pgm_path=../../checkpoints/p_r_h/box2_pgm/checkpoint.pt \
    --predictor_path=../../checkpoints/p_r_h/box2_aux/checkpoint.pt \
    --vae_path=../../checkpoints/p_r_h/box22_hvae_prh/checkpoint.pt \
    --bs=1100 \
    --logw_s=1 \
    --logw_c=0\
    --gpu=4 \
    --seed=6 \
    --logw_n=11 \
    --natural_eps=1e-3 \
    --epoch_num=50001 \
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"
#1100->840
# bash train_ours_box22.sh box/s_allc_3_11 nohup 7  
# bash train_ours_box22.sh box/s_allc_4_11 nohup 6
# bash train_ours_box22.sh box/s_allc_5_11 nohup 5
# bash train_ours_box22.sh box/s_allc_6_11 nohup 4  


# bash train_ours_box22.sh box/s_allc_0_11 nohup 6  
# bash train_ours_box22.sh box/s_allc_1_11 nohup 7
# bash train_ours_box22.sh box/s_allc_2_11 nohup 0



if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi






