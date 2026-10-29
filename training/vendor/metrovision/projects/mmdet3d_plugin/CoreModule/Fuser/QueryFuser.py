from mmdet3d.models import FUSION_LAYERS
import torch
import torch.nn as nn


@FUSION_LAYERS.register_module()
class QueryFuser(nn.Module):
    def __init__(self, queryDim=256, point_cloud_range=None):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.transform_query = nn.Linear(queryDim + 12, queryDim, dtype=torch.float32)
        self.transform_init_ref_point = nn.Linear(15, 3, dtype=torch.float32)
        self.transform_inter_ref_point = nn.Linear(15, 3, dtype=torch.float32)
        self.w_query = nn.Parameter(torch.tensor(0.0))
        self.w_init = nn.Parameter(torch.tensor(0.0))
        self.w_inter = nn.Parameter(torch.tensor(0.0))

    def _loc_denorm(self, ref_pts, pc_range):
        """
        normalized (x,y,z) ---> absolute (x,y,z) in global coordinate system
        """
        from mmdet.models.utils.transformer import inverse_sigmoid
        locs = inverse_sigmoid(ref_pts)

        locs[..., 0:1] = (locs[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs

    def _loc_norm(self, locs, pc_range):
        """
        absolute (x,y,z) in global coordinate system ---> normalized (x,y,z)
        """
        from mmdet.models.utils.transformer import inverse_sigmoid

        locs[..., 0:1] = (locs[..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

        locs = locs.sigmoid()

        return locs

    def forward(self, cloudQuery, vehicleQuery, R, T):
        # return vehicleQuery
        B = len(R)
        R = torch.cat(R, dim=0).float()
        T = torch.cat(T, dim=0).float()
        # bev_embed, hs, init_reference, inter_references = cloudQuery
        num_dec = cloudQuery[1].shape[0]
        num_query = cloudQuery[1].shape[1]

        cloud_init_ref_point = self._loc_denorm(cloudQuery[2], self.point_cloud_range)
        cloud_inter_ref_point = self._loc_denorm(cloudQuery[3], self.point_cloud_range)
        veh_init_ref_point = self._loc_denorm(vehicleQuery[2], self.point_cloud_range)
        veh_inter_ref_point = self._loc_denorm(vehicleQuery[3], self.point_cloud_range)

        # veh_init_ref_point = self._loc_norm(veh_init_ref_point, self.point_cloud_range)
        # veh_inter_ref_point = self._loc_norm(veh_inter_ref_point, self.point_cloud_range)

        # return vehicleQuery[0], vehicleQuery[1], veh_init_ref_point, veh_inter_ref_point

        calib_inf2veh = torch.cat((R.reshape(-1, 9), T.reshape(-1, 3)), dim=-1)

        calib_inf2veh_init = calib_inf2veh.unsqueeze(0)
        calib_inf2veh_init = calib_inf2veh_init.repeat(B, num_query, 1)

        calib_inf2veh_inter = calib_inf2veh.unsqueeze(0).unsqueeze(0)
        calib_inf2veh_inter = calib_inf2veh_inter.repeat(num_dec, B, num_query, 1)

        calib_inf2veh = calib_inf2veh.unsqueeze(0).unsqueeze(0)
        calib_inf2veh = calib_inf2veh.repeat(num_dec, num_query, B, 1)
        cloud_query_veh_view = self.transform_query(torch.cat((cloudQuery[1], calib_inf2veh), dim=-1))
        cloud_init_ref_point_veh_view = self.transform_init_ref_point(torch.cat((cloud_init_ref_point, calib_inf2veh_init), dim=-1))
        cloud_inter_ref_point_veh_view = self.transform_inter_ref_point(torch.cat((cloud_inter_ref_point, calib_inf2veh_inter),
                                                                 dim=-1))

        # 2. Concatenate and fuse.
        # fused_query = torch.cat([vehicleQuery[1], cloud_query_veh_view], dim=2)
        # fused_init_ref_point = torch.cat([veh_init_ref_point, cloud_init_ref_point_veh_view], dim=2)
        # fused_inter_ref_point = torch.cat([veh_inter_ref_point, cloud_inter_ref_point_veh_view], dim=2)
        fused_query = vehicleQuery[1] + self.w_query * cloud_query_veh_view
        fused_init_ref_point = veh_init_ref_point + self.w_init * cloud_init_ref_point_veh_view
        fused_inter_ref_point = veh_inter_ref_point + self.w_inter * cloud_inter_ref_point_veh_view

        fused_init_ref_point = self._loc_norm(fused_init_ref_point, self.point_cloud_range)
        fused_inter_ref_point = self._loc_norm(fused_inter_ref_point, self.point_cloud_range)
        # print(self.w)

        # fused_query = []
        # for query in allQuery:
        #     q = self.q.unsqueeze(1).repeat(1, B, 1)
        #     attn_output, _ = self.crossAttn(q, query, query)
        #     attn_output = attn_output.unsqueeze(0)
        #     fused_query.append(attn_output)
        # fused_query = torch.cat(fused_query, dim=0)
        return vehicleQuery[0], fused_query, fused_init_ref_point, fused_inter_ref_point


if __name__ == '__main__':
    # Initialize parameters.
    num_dec = 3
    B = 2  # Match the input batch size.
    R = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]).unsqueeze(0).repeat(B, 1, 1)  # [B, 3, 3]
    T = torch.tensor([10.0, 0, 0]).unsqueeze(0).repeat(B, 1)  # [B, 3]

    # QueryFuser
    query_fuser = QueryFuser()

    vehicle_query = torch.randn(num_dec, 100, B, 256)  # Vehicle-side queries
    infra_query = torch.randn(num_dec, 100, B, 256)  # Infrastructure-side queries
    fused_query = query_fuser((vehicle_query), (infra_query), R, T)
    print(fused_query[2].shape)
