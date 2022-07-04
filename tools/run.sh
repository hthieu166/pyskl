#!/bin/bash
CONFIG_FILE="configs/k_gtcg++_joint_angle/b.py"
# CONFIG_FILE="configs/stgcn++/stgcn++_ntu60_xsub_hrnet/b.py"
# CHECKPOINT="./work_dirs/stgcn++/stgcn++_ntu60_xsub_hrnet/b/latest.pth"
NUM_GPUS=1
## training
bash tools/dist_train.sh $CONFIG_FILE $NUM_GPUS
## testing
# bash tools/dist_test.sh  $CONFIG_FILE $CHECKPOINT $NUM_GPUS\
#     --out ./output/stgcn++/output_stgcn++_ntu60_xsub_hrnet.pkl \
#     --eval top_k_accuracy mean_class_accuracy