import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class QueryBEVFuser(nn.Module):
    def __init__(self, bev_h, bev_w, query_num, query_dim):
        super().__init__()
        self.query_num = query_num
        self.query_dim = query_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))
        self.fuse_rt = nn.Linear(query_dim + 12, query_dim, dtype=torch.float32)
        self.to_bev = nn.Sequential(
            nn.Linear(query_num, query_num, dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Linear(query_num, bev_h * bev_w, dtype=torch.float32),
            nn.ReLU(inplace=True))

    def forward(self, cloud_query, veh_bev, R, T):
        cloud_query = cloud_query[1][-1].permute(1, 0, 2)
        B = len(R)
        R = torch.cat(R, dim=0).float()
        T = torch.cat(T, dim=0).float()
        calib_inf2veh = torch.cat((R.reshape(-1, 9), T.reshape(-1, 3)), dim=-1)
        calib_inf2veh = calib_inf2veh.repeat(self.query_num, 1)
        cloud_bev_veh_view = []
        for i in range(B):
            cloud_q = cloud_query[i]
            cloud_q_veh_view = self.fuse_rt(torch.cat((cloud_q, calib_inf2veh), dim=-1))
            cloud_bev_veh_view_ = self.to_bev(cloud_q_veh_view.T)
            cloud_bev_veh_view_ = cloud_bev_veh_view_.view(self.query_dim, self.bev_w * self.bev_h)
            cloud_bev_veh_view_ = cloud_bev_veh_view_.unsqueeze(0)
            cloud_bev_veh_view_ = cloud_bev_veh_view_.permute(0, 2, 1)
            cloud_bev_veh_view.append(cloud_bev_veh_view_)
        cloud_bev_veh_view = torch.cat(cloud_bev_veh_view, dim=0)
        # print(cloud_bev_veh_view.shape, veh_bev.shape)
        return cloud_bev_veh_view * self.fusion_weight + veh_bev


if __name__ == '__main__':
    # Initialize parameters.
    num_dec = 10
    B = 2  # Match the input batch size.

    # QueryFuser
    query_fuser = QueryBEVFuser()
    veh_bev = torch.randn(22500, B, 256)  # Vehicle-side queries
    infra_query = torch.randn(900, B, 256)  # Infrastructure-side queries
    fused_query = query_fuser(infra_query, veh_bev)
    print(fused_query.shape)
