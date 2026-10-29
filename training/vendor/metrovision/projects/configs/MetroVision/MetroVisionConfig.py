from projects.configs.MetroVision.VehicleConfig.BEVFormerConfig import model as vehicleModule
from projects.configs.MetroVision.VehicleConfig.BEVFormerConfig import data as vehicleDataset
from projects.configs.MetroVision.CloudConfig.AiOCloudConfig import model as cloudModule
from projects.configs.MetroVision.CloudConfig.AiOCloudConfig import data as cloudDataset
from projects.configs.MetroVision.base_cfg import data_root, point_cloud_range, voxel_size, class_names, num_query

_base_ = [
    # '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
fusionModes = ['BEV+BEV', 'BEV+Query', 'Query+BEV', 'Query+Query', 'bbox+bbox', 'veh_only']

checkpoint_paths = {
    'BEV+BEV':'/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_bev_bev_concat.pth', 
    'BEV+Query':'/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_infbev_vehq.pth', 
    'Query+BEV':'/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_infq_vehbev.pth', 
    'Query+Query':'/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_q_q.pth', 
    'bbox+bbox':'/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_bbox_bbox.pth',
    'veh_only': '/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_bbox_bbox.pth'
}

fusionMode = fusionModes[0]

checkpoint_path = checkpoint_paths[fusionMode]

model_delay = 0

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 150
bev_w_ = 150
queue_length = 3

model = dict(
    type='metrovision',
    cloudSide=cloudModule,
    vehicleSide=vehicleModule,
    fusionMode=fusionMode,
    BEVFuser=dict(
        type='BEVFuser',
        bevDim=256,
        fusionType='attention',
        point_cloud_range=point_cloud_range),
    queryFuser=dict(
        type='QueryFuser',
        queryDim=256,
        point_cloud_range=point_cloud_range),
    bboxFuser=dict(
        type='BboxFuser'),
    BEVQueryFuser=dict(
        type='BEVQueryFuser',
        bev_h=bev_h_,
        bev_w=bev_w_,
        query_num=num_query,
        point_cloud_range=point_cloud_range),
    queryBEVFuser=dict(
        type='QueryBEVFuser',
        bev_h=bev_h_,
        bev_w=bev_w_,
        query_num=num_query,
        query_dim=256),
    metrovisionHead=dict(
        type='metrovisionHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=num_query,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_)],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1)],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[0, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
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
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))),
    vehiclePthFile='/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_full_veh_epoch_13.pth',
    # vehiclePthFile='/home/omnisky/Desktop/WJN/metrovision/pth/petr_dair_full_veh_epoch_24.pth',
    headPthFile='/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_head.pth')

dataset_type = 'AiODataset'
file_client_args = dict(backend='disk')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        cloudDataset=cloudDataset['train'],
        vehicleDataset=vehicleDataset['train'],
        anno_file=data_root + 'COOPERATION/cop_train_annos.pkl',
        point_cloud_range=point_cloud_range),
    val=dict(
        type=dataset_type,
        cloudDataset=cloudDataset['val'],
        vehicleDataset=vehicleDataset['val'],
        anno_file=data_root + 'COOPERATION/cop_val_annos.pkl',
        point_cloud_range=point_cloud_range),
    test=dict(
        type=dataset_type,
        cloudDataset=cloudDataset['test'],
        vehicleDataset=vehicleDataset['test'],
        anno_file=data_root + 'COOPERATION/cop_val_annos.pkl',
        point_cloud_range=point_cloud_range,
        model_delay=model_delay),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(
    type='AdamW',
    lr=2e-4,  # PETR 2e-4 # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  ## PETR 0.1  # 0.25 only for Focal-PETR with R50-in1k pretrained weights
            'vehicleSide.img_backbone': dict(lr_mult=0.0),
            'vehicleSide.pts_bbox_head': dict(lr_mult=0.0),
            'cloudSide.vehicleSide.img_backbone': dict(lr_mult=0.0),
            'cloudSide.vehicleSide.pts_bbox_head': dict(lr_mult=0.0),
            'cloudSide.infrastructureSide.img_backbone': dict(lr_mult=0.0),
            'cloudSide.infrastructureSide.pts_bbox_head': dict(lr_mult=0.0),
            'metrovisionHead': dict(lr_mult=0.1)
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # BEVFormer
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
# evaluation = dict(interval=24, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# load_from = '/home/omnisky/Desktop/WJN/metrovision/Q+BEV_work_dir_withlr/latest.pth'
load_from = None
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])
find_unused_parameters=True

checkpoint_config = dict(interval=1)
