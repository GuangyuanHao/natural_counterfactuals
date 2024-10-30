#!/bin/bash
exp_name="$1"
run_cmd="python train_pgm.py \
    --exp_name=$exp_name \
    --dataset=box2 \
    --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
    --csm_order prh \
    --parents_x p r h \
    --setup=sup_pgm \
    --input_res=64 \
    --input_channels=4 \
    --pad=3 \
    --context_dim=7 \
    --lr=0.001 \
    --bs=1250 \
    --wd=0.01 \
    --eval_freq=4 \
    --epochs 1000"

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi

# bash run_box2_pgm.sh box2_pgm nohup

# testing
# exp_name="$1"
# run_cmd="python train_pgm.py \
#     --dataset=morphomnist \
#     --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
#     --load_path=../../checkpoints/p_r_h/box_pgm64/checkpoint.pt \
#     --setup=sup_pgm \
#     --testing \
#     --input_res=64 \
#     --input_channels=4 \
#     --pad=3 \
#     --context_dim=7"



# if [ "$2" = "nohup" ]
# then
#   nohup ${run_cmd} > $exp_name.out 2>&1 &
#   echo "Started training in background with nohup, PID: $!"
# else
#   ${run_cmd}
# fi

# bash run_box_pgm.sh box_pgm64_test nohup
# Evaluating test set:


#   0%|          | 0/4 [00:00<?, ?it/s]
#  => eval | loss: 5.7066, logp(c2): -1.1171, logp(s2): -1.1167, logp(c3): -1.0660, logp(m): -1.1363, logp(c1): -0.4470, logp(s1): -0.4094, logp(s3): -0.4141:   0%|          | 0/4 [00:03<?, ?it/s]
#  => eval | loss: 5.6776, logp(c2): -1.1258, logp(s2): -1.0919, logp(c3): -1.0882, logp(m): -1.1061, logp(c1): -0.4372, logp(s1): -0.4161, logp(s3): -0.4122:   0%|          | 0/4 [00:03<?, ?it/s]
#  => eval | loss: 5.7020, logp(c2): -1.1224, logp(s2): -1.1216, logp(c3): -1.1023, logp(m): -1.0960, logp(c1): -0.4320, logp(s1): -0.4193, logp(s3): -0.4085:   0%|          | 0/4 [00:03<?, ?it/s]
#  => eval | loss: 5.7203, logp(c2): -1.1209, logp(s2): -1.1041, logp(c3): -1.1146, logp(m): -1.1070, logp(c1): -0.4285, logp(s1): -0.4261, logp(s3): -0.4190:   0%|          | 0/4 [00:03<?, ?it/s]
#  => eval | loss: 5.7203, logp(c2): -1.1209, logp(s2): -1.1041, logp(c3): -1.1146, logp(m): -1.1070, logp(c1): -0.4285, logp(s1): -0.4261, logp(s3): -0.4190: 100%|██████████| 4/4 [00:03<00:00,  1.07it/s]



# exp_name="$1"
# run_cmd="python train_pgm.py \
#     --exp_name=$exp_name \
#     --dataset=box \
#     --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
#     --setup=sup_pgm \
#     --input_res=192 \
#     --pad=9 \
#     --input_channels=4 \
#     --context_dim=7 \
#     --lr=0.001 \
#     --bs=1250 \
#     --wd=0.01 \
#     --eval_freq=4 \
#     --epochs 1000"

# if [ "$2" = "nohup" ]
# then
#   nohup ${run_cmd} > $exp_name.out 2>&1 &
#   echo "Started training in background with nohup, PID: $!"
# else
#   ${run_cmd}
# fi

# bash run_box_pgm.sh box_pgm192_2 nohup

# testing
# exp_name="$1"
# run_cmd="python train_pgm.py \
#     --dataset=morphomnist \
#     --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
#     --load_path=../../checkpoints/p_r_h/box_pgm64/checkpoint.pt \
#     --setup=sup_pgm \
#     --testing \
#     --input_res=64 \
#     --input_channels=4 \
#     --pad=3 \
#     --context_dim=7"



# if [ "$2" = "nohup" ]
# then
#   nohup ${run_cmd} > $exp_name.out 2>&1 &
#   echo "Started training in background with nohup, PID: $!"
# else
#   ${run_cmd}
# fi

# bash run_box_pgm.sh box_pgm64_test nohup


