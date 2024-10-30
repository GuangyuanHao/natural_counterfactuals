#!/bin/bash
#vae
# exp_name="$1"
# run_cmd="python main.py \
#     --exp_name=$exp_name \
#     --gpu=6 \
#     --data_dir=../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
#     --hps box \
#     --csm_order prh \
#     --parents_x p r h \
#     --concat_pa \
#     --lr=0.001 \
#     --bs=32 \
#     --wd=0.01 \
#     --beta=1 \
#     --eval_freq=4 \
#     --vae simple \
#     --epochs 500"

# if [ "$2" = "nohup" ]
# then
#   nohup ${run_cmd} > $exp_name.out 2>&1 &
#   echo "Started training in background with nohup, PID: $!"
# else
#   ${run_cmd}
# fi
# vae# bash run_local_box.sh box_vae_prh nohup
# lr=0.001有问题

# gpu 0 3 4 6 7
##hvae
exp_name="$1"
run_cmd="python main.py \
    --exp_name=$exp_name \
    --gpu=4 \
    --data_dir=../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
    --hps box \
    --csm_order prh \
    --parents_x p r h \
    --concat_pa \
    --lr=0.001 \
    --bs=128 \
    --wd=0.01 \
    --beta=1 \
    --eval_freq=4 \
    --epochs 500"

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi

#hvae # bash run_local_box_h.sh box_hvae_prh nohup