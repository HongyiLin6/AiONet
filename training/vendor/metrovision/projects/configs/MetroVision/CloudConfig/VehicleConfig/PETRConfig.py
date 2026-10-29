from projects.configs.MetroVision.base_cfg import (
    data_root, point_cloud_range, voxel_size, 
    class_names, img_norm_cfg, input_modality)
# backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='Petr3D',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        # pretrained = 'ckpts/resnet50_msra-5891d200.pth',
        ),
    img_neck=dict(
        type='PETRCPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),    
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='PETRNMSFreeCoder',
            # type='NMSFreeClsCoder',
            post_center_range=[0, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='PETRSinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='PETRHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='PETRBBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'PETRCustomNuScenesDataset'

file_client_args = dict(backend='disk')
db_sampler = dict()
ida_aug_conf = {
        "resize_lim": (0.8, 1.0),
        "final_dim": (512, 1408),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PETRLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='PETRObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PETRObjectNameFilter', classes=class_names),
    dict(type='PETRResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='PETRGlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True
            ),
    dict(type='PETRNormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PETRPadMultiViewImage', size_divisor=32),
    dict(type='PETRDefaultFormatBundle3D', class_names=class_names),
    dict(type='PETRCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PETRResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='PETRNormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PETRPadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'PETR/veh_temporal_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,ann_file=data_root + 'PETR/veh_temporal_infos_val.pkl', pipeline=test_pipeline, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, data_root=data_root, ann_file=data_root + 'PETR/veh_temporal_infos_val.pkl', pipeline=test_pipeline, classes=class_names, modality=input_modality))
