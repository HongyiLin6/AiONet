# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (init_dist, load_checkpoint,
                         wrap_fp16_model)
from tqdm import tqdm
import numpy as np

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.SideModule.BEVFormer.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from utils.visualize import visualize_camera, visualize_map
from mmdet3d.core import LiDARInstance3DBoxes


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred", "combo"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="viz")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES
    # # palette for visualization in segmentation tasks
    # if 'PALETTE' in checkpoint.get('meta', {}):
    #     model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    
    i = 1
    for data in tqdm(data_loader):
        # print(data)
        # data = data['cloudData']['infrastructureData'][0]
        # print(data)
        if args.mode == "gt":
            metas = data['img_metas'].data[0][0][0]
        else:
            metas_inf = data['cloudData']['infrastructureData'][0]['img_metas'][0].data[0][0]
            metas_veh = data['vehicleData']['img_metas'][0].data[0][0]
            veh2inf = data['veh2inf'].data[0][0].numpy()
            inf2veh = np.linalg.inv(veh2inf)
            print(metas_inf['sample_idx'])
        name = "{}".format(i)
        i += 1

        if args.mode == "pred" or args.mode == "combo":
            with torch.inference_mode():
                outputs = model(**data, return_loss=False)[0]['pts_bbox']
        # print(outputs)
        # return

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()


            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            #bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs:
            bboxes = outputs["boxes_3d"].tensor.numpy()
            scores = outputs["scores_3d"].numpy()
            labels = outputs["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            #bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "combo" and "gt_bboxes_3d" in data and "boxes_3d" in outputs:
            bboxes = outputs["boxes_3d"].tensor.numpy()
            scores = outputs["scores_3d"].numpy()
            labels = outputs["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
            
            #bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

            gtbboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            gtlabels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(gtlabels, args.bbox_classes)
                gtbboxes = gtbboxes[indices]
                gtlabels = gtlabels[indices]

            #bboxes[..., 2] -= bboxes[..., 5] / 2
            gtbboxes = LiDARInstance3DBoxes(gtbboxes, box_dim=9)    
        else:
            bboxes = None
            labels = None

        masks = None

        for k, image_path in enumerate(metas_inf["filename"]):
            image = mmcv.imread(image_path)
            visualize_camera(
                os.path.join(args.out_dir, f"camera_inf", f"{name}.png"),
                image,
                bboxes=bboxes,
                labels=labels,
                transform=metas_inf["lidar2img"][k] @ veh2inf,
                classes=cfg.class_names,
            )

        for k, image_path in enumerate(metas_veh["filename"]):
            image = mmcv.imread(image_path)
            visualize_camera(
                os.path.join(args.out_dir, f"camera_veh", f"{name}.png"),
                image,
                bboxes=bboxes,
                labels=labels,
                transform=metas_veh["lidar2img"][k],# @ inf2veh,
                classes=cfg.class_names,
            )
            # print(metas['lidar2img'][k])
        # print()
        # print(veh2inf)

        # infrastructure_lidar = data["infrastructure_points"].data[0][0].numpy()
        # infrastructure_lidar = infrastructure_lidar[:, :4]
        # vehicle_lidar = data["vehicle_points"].data[0][0].numpy()
        # vehicle_lidar = vehicle_lidar[:, :4]

        # if args.mode == "combo":
        #     visualize_lidar_two_combo(
        #         os.path.join(args.out_dir, "fused-lidar", f"{name}uav1.png"),
        #         vehicle_lidar,
        #         infrastructure_lidar,
        #         gtbboxes=gtbboxes,
        #         bboxes=bboxes,
        #         gtlabels=gtlabels,
        #         labels=labels,
        #         xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
        #         ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
        #         classes=cfg.object_classes,
        #     )
        # else:
        #     visualize_lidar_two(
        #         os.path.join(args.out_dir, "fused-lidar", f"{name}uav1.png"),
        #         vehicle_lidar,
        #         infrastructure_lidar,
        #         bboxes=bboxes,
        #         labels=labels,
        #         xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
        #         ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
        #         classes=cfg.object_classes,
        #     )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == '__main__':
    main()
