# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from .dgcnn3d_head import PETRDGCNN3DHead
from .detr3d_head import PETRDetr3DHead
from .petr_head import PETRHead
from .petrv2_head import PETRv2Head
from .petr_head_seg import PETRHead_seg
from .petrv2_dnhead import PETRv2DNHead
__all__ = ['PETRDGCNN3DHead', 'PETRDetr3DHead','PETRHead','PETRv2Head','PETRHead_seg', 'PETRv2DNHead']