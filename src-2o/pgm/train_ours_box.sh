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
    --bs=840 \
    --logw_s=1 \
    --logw_c=0\
    --gpu=3 \
    --seed=9 \
    --natural_eps=1e-3 \
    --logw_n=9 \
    --epoch_num=50001 \
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"
# 50001
# bash train_ours_box.sh box/w_allc_0_3_9 nohup
# bash train_ours_box.sh box/w_allc_1_3_9 nohup
# bash train_ours_box.sh box/w_allc_2_3_9 nohup
# 1 3 8
# bash train_ours_box.sh box/w_allc_0_3_9 nohup 1
# bash train_ours_box.sh box/w_allc_3_3_9 nohup 2
# bash train_ours_box.sh box/w_allc_4_3_9 nohup 3
# bash train_ours_box.sh box/w_allc_5_3_9 nohup 5
# bash train_ours_box.sh box/w_allc_6_3_9 nohup 6
# bash train_ours_box.sh box/w_allc_7_3_9 nohup 1
# bash train_ours_box.sh box/w_allc_8_3_9 nohup 2
# bash train_ours_box.sh box/w_allc_9_3_9 nohup 3

# bash train_ours_box.sh box/w_allc_0_3_10 nohup
# bash train_ours_box.sh box/w_allc_1_3_10 nohup
# bash train_ours_box.sh box/w_allc_2_3_10 nohup





if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi



