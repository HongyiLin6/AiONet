#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
SAVEPATH=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/visualize.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} \
    --eval bbox --mode pred --bbox-score 0.3 --out-dir $SAVEPATH
