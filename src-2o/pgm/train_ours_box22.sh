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
    --bs=720 \
    --logw_s=1 \
    --logw_c=0\
    --gpu=5 \
    --seed=2 \
    --natural_eps=1e-3 \
    --logw_n=10 \
    --epoch_num=50001 \
    --cf_lr=5.0 \
    --lr_step_size 10000 \
    --lr_gamma 0.1
"
# 
# bash train_ours_box22.sh box/s_allc_0_3_10 nohup
# bash train_ours_box22.sh box/s_allc_1_3_10 nohup
# bash train_ours_box22.sh box/s_allc_2_3_10 nohup





if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
#--logw_n=11
# bash train_ours_box22.sh box/formal_box22_c1s1s3_seed0_3_11 nohup one dataset



#var_list={0:{'c1'}, 2:{'s1'}, 3:{'s3'}, 4:{'c1', 's1', 's3'}}
#var_list={0:{'c2'}, 1:{'c1'}}




# bash train_ours.sh morph/train_cf_vae nohup


## hvae
# exp_name="$1"
# run_cmd="python train_ours.py \
#     --exp_name=$exp_name \
#     --dataset=morphomnist \
#     --data_dir=../../datasets/morphomnist \
#     --pgm_path=../../checkpoints/t_i_d/morph_other/checkpoint.pt \
#     --predictor_path=../../checkpoints/t_i_d/morph_other_aux/checkpoint.pt \
#     --vae_path=../../checkpoints/t_i_d/morph_hvae_prior160/checkpoint.pt \
#     --testing \
#     --bs=10000 \
#     --setup=sup_pgm \
#     --input_res=32 \
#     --pad=4 \
#     --seed=0 \
#     --context_dim=12
# "

# train_cf_debug
# exp_name="$1"
# run_cmd="python train_cf_debug.py \
#     --exp_name=$exp_name \
#     --dataset=morphomnist \
#     --data_dir=../../datasets/morphomnist \
#     --pgm_path=../../checkpoints/t_i_d/morph_other/checkpoint.pt \
#     --predictor_path=../../checkpoints/t_i_d/morph_other_aux/checkpoint.pt \
#     --vae_path=../../checkpoints/t_i_d/morph_dscm_prior/checkpoint.pt \
#     --testing \
#     --bs=10000 \
#     --setup=sup_pgm \
#     --input_res=32 \
#     --pad=4 \
#     --seed=0 \
#     --context_dim=12 \
#     --vae simple
# "



