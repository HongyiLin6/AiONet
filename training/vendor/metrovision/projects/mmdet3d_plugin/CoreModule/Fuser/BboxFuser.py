import torch
import torch.nn as nn
from mmdet3d.models import FUSION_LAYERS

from projects.mmdet3d_plugin.CoreModule.utils import transform_bbox


@FUSION_LAYERS.register_module()
class BboxFuser(nn.Module):
    def __init__(self, iouThreshold=0.5):
        super().__init__()
        self.iouThreshold = iouThreshold

    def forward(self, cloudBboxes, vehBboxes, R, T):
        veh_bboxes_veh_view = vehBboxes['all_bbox_preds']
        # veh_bboxes_inf_view = transform_bbox(veh_bboxes_veh_view, R, T)
        cloud_bboxes_inf_view = cloudBboxes['all_bbox_preds']
        cloud_bboxes_veh_view = transform_bbox(cloud_bboxes_inf_view, R, T)
        fused_bboxes_veh_view = torch.cat((cloud_bboxes_veh_view, veh_bboxes_veh_view), dim=2)

        cloud_all_cls_score = cloudBboxes['all_cls_scores']
        veh_all_cls_score = vehBboxes['all_cls_scores']
        fused_cls_score = torch.cat((cloud_all_cls_score, veh_all_cls_score), dim=2)

        outs = {
            'all_cls_scores': fused_cls_score,
            'all_bbox_preds': fused_bboxes_veh_view,
            'enc_cls_scores': None,
            'enc_bbox_preds': None
        }
        return outs


if __name__ == '__main__':
    num_dec = 6
    B = 2
    R = torch.stack([torch.eye(3)] * B)
    T = torch.tensor([[10.0, 0, 0], [5.0, 0, 0]]).float()

    fuser = BboxFuser(iouThreshold=0.3)

    vehicle_data = {
        'all_bbox_preds': torch.randn(num_dec, B, 800, 10),
        'all_cls_scores': torch.randn(num_dec, B, 800, 6)
    }

    infra_data = {
        'all_bbox_preds': torch.randn(num_dec, B, 900, 10),
        'all_cls_scores': torch.randn(num_dec, B, 900, 6)
    }

    fused_data = fuser(vehicle_data, infra_data, R, T)

    print("Fused bbox shape:", fused_data['all_bbox_preds'].shape)  # [B, 1, 900, 10]
    print("Fused cls shape:", fused_data['all_cls_scores'].shape)  # [B, 1, 900, 6]
