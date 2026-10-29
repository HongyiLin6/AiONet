#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
LATENCY=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox --latency $LATENCY
