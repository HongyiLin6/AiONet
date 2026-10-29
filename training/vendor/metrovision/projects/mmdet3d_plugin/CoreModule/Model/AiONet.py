import torch
from mmcv.runner import load_checkpoint

from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, Base3DDetector, build_fusion_layer
from mmdet3d.models import build_model, build_head
import numpy as np

from projects.mmdet3d_plugin.CoreModule.utils import transform_bbox


@DETECTORS.register_module()
class metrovision(Base3DDetector):
    def __init__(self,
                 cloudSide,
                 vehicleSide,
                 fusionMode,
                 BEVFuser,
                 queryFuser,
                 bboxFuser,
                 BEVQueryFuser,
                 queryBEVFuser,
                 metrovisionHead,
                 headPthFile=None,
                 vehiclePthFile=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.cloudSide = build_model(cloudSide)
        self.vehicleSide = build_model(vehicleSide)
        if vehiclePthFile is not None:
            load_checkpoint(self.vehicleSide, vehiclePthFile, map_location='cpu')
            for p in self.vehicleSide.parameters():
                p.requires_grad = False
        self.fusionMode = fusionMode
        if self.fusionMode == 'BEV+BEV':
            self.BEVFuser = build_fusion_layer(BEVFuser)
        elif self.fusionMode == 'Query+Query':
            self.QueryFuser = build_fusion_layer(queryFuser)
        elif self.fusionMode == 'bbox+bbox':
            self.bboxFuser = build_fusion_layer(bboxFuser)
        elif self.fusionMode == 'BEV+Query':
            self.BEVQueryFuser = build_fusion_layer(BEVQueryFuser)
        elif self.fusionMode == 'Query+BEV':
            self.QueryBEVFuser = build_fusion_layer(queryBEVFuser)
        pts_train_cfg = train_cfg.pts if train_cfg else None
        metrovisionHead.update(train_cfg=pts_train_cfg)
        pts_test_cfg = test_cfg.pts if test_cfg else None
        metrovisionHead.update(test_cfg=pts_test_cfg)
        self.metrovisionHead = build_head(metrovisionHead)
        if headPthFile is not None:
            load_checkpoint(self.metrovisionHead, headPthFile, map_location='cpu')
            # for p in self.metrovisionHead.parameters():
            #     p.requires_grad = False
        self.cnt = 0

    def init_weights(self):
        self.cloudSide.init_weights()
        self.vehicleSide.init_weights()
        self.metrovisionHead.init_weights()

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      cloudData=None,
                      vehicleData=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      veh2inf=None,
                      img_metas=None,
                      **kwargs):
        self.cnt += 1
        # print(self.cnt)
        inf2veh = [torch.inverse(mat.cpu()).cuda() for mat in veh2inf]
        inf2veh_rotation = [rot[:3, :3].cuda() for rot in inf2veh]
        inf2veh_translation = [rot[:3, 3].cuda() for rot in inf2veh]
        veh2inf_rotation = [rot[:3, :3].cuda() for rot in veh2inf]
        veh2inf_translation = [rot[:3, 3].cuda() for rot in veh2inf]

        outs = self.extract_results(cloudData, vehicleData, veh2inf_rotation, veh2inf_translation,
                                    inf2veh_rotation, inf2veh_translation, train=True)

        # loss_inputs = [vehicleData['gt_bboxes_3d'], vehicleData['gt_labels_3d'], outs]
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.metrovisionHead.loss(*loss_inputs)
        return losses

    def forward_test(self,
                     cloudData=None,
                     vehicleData=None,
                     veh2inf=None,
                     img_metas=None,
                     **kwargs):
        inf2veh = [torch.inverse(mat.cpu()).cuda() for mat in veh2inf]
        inf2veh_rotation = [rot[:3, :3].cuda() for rot in inf2veh]
        inf2veh_translation = [rot[:3, 3].cuda() for rot in inf2veh]
        veh2inf_rotation = [rot[:3, :3].cuda() for rot in veh2inf]
        veh2inf_translation = [rot[:3, 3].cuda() for rot in veh2inf]

        outs_veh_view = self.extract_results(cloudData, vehicleData, veh2inf_rotation, veh2inf_translation,
                                             inf2veh_rotation, inf2veh_translation, train=False)

        bbox_list_veh_view = [dict() for _ in range(len(vehicleData['img_metas']))]
        pts_bbox_list = self.metrovisionHead.get_bboxes(outs_veh_view, vehicleData['img_metas'][0])
        bbox_pts = [bbox3d2result(bboxes, score, labels) for bboxes, score, labels in pts_bbox_list]
        for result_dict, pts_bbox in zip(bbox_list_veh_view, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        # print(bbox_list)
        # return
        return bbox_list_veh_view

    def extract_results(self,
                        cloudData,
                        vehicleData,
                        veh2inf_rotation,
                        veh2inf_translation,
                        inf2veh_rotation,
                        inf2veh_translation,
                        train=True):
        if len(cloudData) == 0:
            return self.vehicleSide.getBbox(train=train, **vehicleData)

        cloudData['veh2inf_rotation'] = veh2inf_rotation
        cloudData['veh2inf_translation'] = veh2inf_translation
        cloudData['inf2veh_rotation'] = inf2veh_rotation
        cloudData['inf2veh_translation'] = inf2veh_translation

        if self.fusionMode == 'BEV+BEV':
            cloudBEV = self.cloudSide.getBEV(train=train, **cloudData)  # 1 * 256 * 200 * 200
            vehicleBEV = self.vehicleSide.getBEV(train=train, **vehicleData)
            fusedBEV = self.BEVFuser(cloudBEV, vehicleBEV, inf2veh_rotation, inf2veh_translation)
            outs = self.metrovisionHead('BEV', bev_embed=fusedBEV)
        elif self.fusionMode == 'Query+Query':
            cloudQuery_output = self.cloudSide.getQuery(train=train, **cloudData)
            vehicleQuery_output = self.vehicleSide.getQuery(train=train, **vehicleData)
            fusedQuery_output = self.QueryFuser(cloudQuery_output, vehicleQuery_output, inf2veh_rotation, inf2veh_translation)
            outs = self.metrovisionHead.forwardQuery(fusedQuery_output)
        elif self.fusionMode == 'bbox+bbox':
            cloudBbox = self.cloudSide.getBbox(train=train, **cloudData)
            vehicleBbox = self.vehicleSide.getBbox(train=train, **vehicleData)            
            fusedBbox = self.bboxFuser(cloudBbox, vehicleBbox, inf2veh_rotation, inf2veh_translation)
            outs = fusedBbox
            # outs['all_bbox_preds'] = transform_bbox(outs['all_bbox_preds'], inf2veh_rotation, inf2veh_translation)
            # outs = vehicleBbox
        elif self.fusionMode == 'BEV+Query':
            cloudBEV = self.cloudSide.getBEV(train=train, **cloudData)
            vehicleQuery = self.vehicleSide.getQuery(train=train, **vehicleData)
            fusedQuery = self.BEVQueryFuser(cloudBEV, vehicleQuery, inf2veh_rotation, inf2veh_translation)
            outs = self.metrovisionHead.forwardQuery(fusedQuery)
        elif self.fusionMode == 'Query+BEV':
            cloudQuery = self.cloudSide.getQuery(train=train, **cloudData)
            vehicleBEV = self.vehicleSide.getBEV(train=train, **vehicleData)
            fusedQuery = self.QueryBEVFuser(cloudQuery, vehicleBEV, inf2veh_rotation, inf2veh_translation)
            outs = self.metrovisionHead.forwardBEV(fusedQuery)
        else:
            vehicleBbox = self.vehicleSide.getBbox(train=train, **vehicleData)
            outs = vehicleBbox
        return outs

    def extract_feat(self, imgs):
        return None

    def aug_test(self, imgs, img_metas, **kwargs):
        return None

    def simple_test(self, imgs, img_metas, **kwargs):
        return None
