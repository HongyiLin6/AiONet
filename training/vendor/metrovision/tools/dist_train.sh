#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORKDIR=$3
PORT=${PORT:-28809}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic --work-dir bev+bev+temposense_10
