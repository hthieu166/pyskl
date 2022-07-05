#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=$3

MKL_SERVICE_FORCE_INTEL=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG --launcher pytorch ${@:4} \
    # -C $CHECKPOINT
