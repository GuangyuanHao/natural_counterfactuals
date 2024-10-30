#!/bin/bash
exp_name="$1"
run_cmd="python train_pgm.py \
    --exp_name=$exp_name \
    --dataset=box \
    --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
      --csm_order prh \
      --parents_x p r h \
    --setup=sup_aux \
    --input_res=64 \
    --input_channels=4 \
    --pad=3 \
    --context_dim=7 \
    --lr=0.001 \
    --bs=512 \
    --wd=0.01 \
    --eval_freq=4 \
    --epochs 160"



if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
# bash run_box_aux.sh box_aux64 nohup


# testing

# run_cmd="python train_pgm.py \
#     --dataset=box \
#     --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
#     --load_path=../../checkpoints/p_r_h/box_aux64/checkpoint.pt \
#     --setup=sup_aux \
#     --testing \
#     --input_res=64 \
#     --input_channels=4 \
#     --pad=3 \
#     --context_dim=7"


# exp_name="$1"
# if [ "$2" = "nohup" ]
# then
#   nohup ${run_cmd} > $exp_name.out 2>&1 &
#   echo "Started training in background with nohup, PID: $!"
# else
#   ${run_cmd}
# fi

# bash run_box_aux.sh box_aux_test nohup


# Evaluating test set:


#   0%|          | 0/10 [00:00<?, ?it/s]
#  => eval | loss: -23.2276, logp(c2_aux): 3.4724, logp(s2_aux): 3.1232, logp(c3_aux): 2.8907, logp(m_aux): 4.1453, logp(c1_aux): 3.2065, logp(s1_aux): 2.9005, logp(s3_aux): 3.4890:   0%|          | 0/10 [00:02<?, ?it/s]
#  => eval | loss: -23.2510, logp(c2_aux): 3.4748, logp(s2_aux): 3.1210, logp(c3_aux): 2.8949, logp(m_aux): 4.1406, logp(c1_aux): 3.1931, logp(s1_aux): 2.9333, logp(s3_aux): 3.4931:   0%|          | 0/10 [00:02<?, ?it/s]
#  => eval | loss: -16.3586, logp(c2_aux): 3.4629, logp(s2_aux): 3.1109, logp(c3_aux): 2.8877, logp(m_aux): 4.1619, logp(c1_aux): 3.1934, logp(s1_aux): 2.9287, logp(s3_aux): -3.3871:   0%|          | 0/10 [00:02<?, ?it/s]
#  => eval | loss: -17.4082, logp(c2_aux): 3.4685, logp(s2_aux): 2.4493, logp(c3_aux): 2.8860, logp(m_aux): 4.1670, logp(c1_aux): 3.1935, logp(s1_aux): 2.9105, logp(s3_aux): -1.6666:   0%|          | 0/10 [00:02<?, ?it/s]
#  => eval | loss: -18.5513, logp(c2_aux): 3.4680, logp(s2_aux): 2.5789, logp(c3_aux): 2.8838, logp(m_aux): 4.1647, logp(c1_aux): 3.1865, logp(s1_aux): 2.9159, logp(s3_aux): -0.6467:   0%|          | 0/10 [00:03<?, ?it/s]
#  => eval | loss: -19.3225, logp(c2_aux): 3.4667, logp(s2_aux): 2.6645, logp(c3_aux): 2.8834, logp(m_aux): 4.1623, logp(c1_aux): 3.1850, logp(s1_aux): 2.9236, logp(s3_aux): 0.0370:   0%|          | 0/10 [00:03<?, ?it/s] 
#  => eval | loss: -19.8274, logp(c2_aux): 3.4634, logp(s2_aux): 2.6795, logp(c3_aux): 2.8838, logp(m_aux): 4.1625, logp(c1_aux): 3.1865, logp(s1_aux): 2.9286, logp(s3_aux): 0.5231:   0%|          | 0/10 [00:03<?, ?it/s]
#  => eval | loss: -20.2342, logp(c2_aux): 3.4592, logp(s2_aux): 2.7292, logp(c3_aux): 2.8842, logp(m_aux): 4.1556, logp(c1_aux): 3.1894, logp(s1_aux): 2.9285, logp(s3_aux): 0.8881:   0%|          | 0/10 [00:03<?, ?it/s]
#  => eval | loss: -20.5243, logp(c2_aux): 3.4581, logp(s2_aux): 2.7614, logp(c3_aux): 2.8823, logp(m_aux): 4.1579, logp(c1_aux): 3.1903, logp(s1_aux): 2.9114, logp(s3_aux): 1.1628:   0%|          | 0/10 [00:03<?, ?it/s]
#  => eval | loss: -20.6776, logp(c2_aux): 3.4582, logp(s2_aux): 2.7870, logp(c3_aux): 2.8854, logp(m_aux): 4.1562, logp(c1_aux): 3.1899, logp(s1_aux): 2.8600, logp(s3_aux): 1.3408:   0%|          | 0/10 [00:03<?, ?it/s]
#  => eval | loss: -20.6776, logp(c2_aux): 3.4582, logp(s2_aux): 2.7870, logp(c3_aux): 2.8854, logp(m_aux): 4.1562, logp(c1_aux): 3.1899, logp(s1_aux): 2.8600, logp(s3_aux): 1.3408: 100%|██████████| 10/10 [00:03<00:00,  2.74it/s]

#   0%|          | 0/10 [00:00<?, ?it/s]
#  10%|█         | 1/10 [00:01<00:13,  1.47s/it]
#  40%|████      | 4/10 [00:01<00:01,  3.20it/s]
#  60%|██████    | 6/10 [00:02<00:01,  2.79it/s]
#  80%|████████  | 8/10 [00:02<00:00,  4.00it/s]
# 100%|██████████| 10/10 [00:03<00:00,  3.30it/s]
# 100%|██████████| 10/10 [00:03<00:00,  2.88it/s]
# test | c1_mae: 0.0072 - c2_mae: 0.0053 - c3_mae: 0.0093 - s1_mae: 0.0220 - s2_mae: 0.0146 - s3_mae: 0.0091 - m_mae: 0.0041

# exp_name="$1"
# run_cmd="python train_pgm.py \
#     --exp_name=$exp_name \
#     --dataset=box \
#     --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
#     --setup=sup_aux \
#     --input_res=192 \
#     --input_channels=4 \
#     --pad=9 \
#     --context_dim=7 \
#     --lr=0.001 \
#     --bs=512 \
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

# bash run_box_aux.sh box_aux192 nohup


# testing

# run_cmd="python train_pgm.py \
#     --dataset=box \
#     --data_dir=../../3DBoxIdent_part2/CausalMultimodal3DIdent/ \
#     --load_path=../../checkpoints/p_r_h/box_aux64/checkpoint.pt \
#     --setup=sup_aux \
#     --testing \
#     --input_res=64 \
#     --input_channels=4 \
#     --pad=3 \
#     --context_dim=7"


# exp_name="$1"
# if [ "$2" = "nohup" ]
# then
#   nohup ${run_cmd} > $exp_name.out 2>&1 &
#   echo "Started training in background with nohup, PID: $!"
# else
#   ${run_cmd}
# fi

# bash run_box_aux.sh box_aux_test nohup