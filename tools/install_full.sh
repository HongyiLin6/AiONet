#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -c 'import sys; assert sys.version_info[:2] == (3,8), "Use a Python 3.8 environment for the detector stack"'
python -m pip install 'pip<25' 'setuptools==59.5.0' wheel
python -m pip install 'torch==1.9.0+cu111' 'torchvision==0.10.0+cu111' -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install --no-deps --only-binary mmcv-full 'mmcv-full==1.6.0' -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python -m pip install -r "$RELEASE_ROOT/requirements-full.txt"
: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install --no-deps --no-build-isolation -e "$RELEASE_ROOT/training/vendor/metrovision/mmdetection3d"
python -c 'import torch,mmcv,mmdet,mmdet3d; print(torch.__version__,mmcv.__version__,mmdet.__version__,mmdet3d.__version__)'
python -m pip check
