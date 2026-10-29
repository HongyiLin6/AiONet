import os
from pathlib import Path
_work=Path(os.environ['AIONET_ROOT'])
model = dict(
    type='VIMI',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            str(_work / 'weights/resnet50-0676ba61.pth')),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_dcn=dict(
        type='DeformConv',
        chi=64,
        cho=64,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        deform_groups=1),
    neck_3d=dict(type='OutdoorImVoxelNeck', in_channels=64, out_channels=256),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=7,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[
                0, -51.2, -1.8699854612350464, 51.2, 51.2, -1.8699854612350464
            ], [
                0, -51.2, -1.8933947086334229, 51.2, 51.2, -1.8933947086334229
            ], [
                0, -51.2, -1.8448482751846313, 51.2, 51.2, -1.8448482751846313
            ], [
                0, -51.2, -1.7054882049560547, 51.2, 51.2, -1.7054882049560547
            ], [0, -51.2, -1.875, 51.2, 51.2, -1.875],
                    [
                        0, -51.2, -1.8314599990844727, 51.2, 51.2,
                        -1.8314599990844727
                    ],
                    [
                        0, -51.2, -1.7596135139465332, 51.2, 51.2,
                        -1.7596135139465332
                    ]],
            sizes=[
                [4.286756992340088, 1.8879040479660034, 1.5475579500198364],
                [6.475243091583252, 2.5869619846343994, 3.0665273666381836],
                [4.817861080169678, 2.0125160217285156, 1.9044990539550781],
                [11.640678405761719, 2.901236057281494, 3.255790948867798],
                [1.8254820108413696, 0.7602394819259644, 1.5288209915161133],
                [1.7207446098327637, 0.6717450022697449, 1.5446490049362183],
                [0.5499500036239624, 0.6077960133552551, 1.6538269519805908]
            ],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    n_voxels=[128, 256, 12],
    compress_ratio=64,
    s_compress_ratio=1,
    se_reduction_ratio=1,
    anchor_generator=dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[0, -51.2, -5, 51.2, 51.2, 3]],
        rotations=[0.0]),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.1,
        score_thr=0.001,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=300))

del os, Path, _work
