# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
from .vovnet import PETRVoVNet
from .vovnetcp import PETRVoVNetCP
__all__ = ['PETRVoVNet', 'PETRVoVNetCP']

