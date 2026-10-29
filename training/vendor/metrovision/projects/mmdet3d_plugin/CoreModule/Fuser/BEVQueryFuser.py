import torch
import torch.nn as nn
from mmdet3d.models import FUSION_LAYERS

from projects.mmdet3d_plugin.CoreModule.utils import transform_bev


@FUSION_LAYERS.register_module()
class BEVQueryFuser(nn.Module):
    def __init__(self, bev_h, bev_w, query_num=900, point_cloud_range=None):
        super().__init__()
        self.to_query = nn.Sequential(
            nn.Linear(bev_h * bev_w, query_num),
            nn.ReLU(),
            nn.Linear(query_num, query_num),
            nn.ReLU()
        )
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))
        self.point_cloud_range = point_cloud_range

    def forward(self, cloudBEV, vehQuery, R, T):
        B, hw, C = cloudBEV.shape
        h = w = int(hw ** 0.5)
        assert h * w == hw

        cloud_bev_inf_view = cloudBEV.view(B, h, w, C).permute(0, 3, 1, 2)
        cloud_bev_veh_view = []
        for i in range(B):
            transformed_cloud_bev = transform_bev(cloud_bev_inf_view[i], R[i], T[i], self.point_cloud_range)
            cloud_bev_veh_view.append(transformed_cloud_bev)
        cloud_bev_veh_view = torch.cat(cloud_bev_veh_view)
        cloud_bev_veh_view = cloud_bev_veh_view.view(B, C, h*w)
        cloud_bev_query = self.to_query(cloud_bev_veh_view)
        cloud_bev_query = cloud_bev_query.unsqueeze(0)
        cloud_bev_query = cloud_bev_query.permute(0, 3, 1, 2)
        num_dec = vehQuery[1].shape[0]
        cloud_bev_query = cloud_bev_query.repeat(num_dec, 1, 1, 1)
        # print(cloud_bev_query.shape, vehQuery[1].shape)
        vehQuery_tmp = vehQuery[1] + cloud_bev_query * self.fusion_weight
        return vehQuery[0], vehQuery_tmp, vehQuery[2], vehQuery[3]

