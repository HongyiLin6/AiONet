import torch
import torch.nn as nn
from mmdet3d.models import FUSION_LAYERS
from projects.mmdet3d_plugin.CoreModule.utils import transform_bev


@FUSION_LAYERS.register_module()
class BEVFuser(nn.Module):
    def __init__(self,
                 bevDim=256,
                 fusionType='concat',
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):  # Roadside perception bounds
        super().__init__()
        self.fusionType = fusionType
        self.point_cloud_range = point_cloud_range
        if fusionType == 'concat':
            self.fusionConv = nn.Conv2d(2 * bevDim, bevDim, kernel_size=3, padding=1)
        elif fusionType == 'attention':
            self.crossAttn = nn.MultiheadAttention(bevDim, num_heads=8)

    def forward(self, cloud_bev, veh_bev, R, T):
        B, hw, C = cloud_bev.shape
        h = w = int(hw ** 0.5)
        assert h * w == hw

        cloud_bev_inf_view = cloud_bev.view(B, h, w, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        veh_bev_veh_view = veh_bev.view(B, h, w, C).permute(0, 3, 1, 2)

        cloud_bev_veh_view = []
        for i in range(B):
            transformed_cloud_bev = transform_bev(cloud_bev_inf_view[i], R[i], T[i], self.point_cloud_range)
            cloud_bev_veh_view.append(transformed_cloud_bev.unsqueeze(0))
        cloud_bev_veh_view = torch.cat(cloud_bev_veh_view)

        if self.fusionType == 'concat':
            fused = torch.cat([cloud_bev_veh_view, veh_bev_veh_view], dim=1)
            out = self.fusionConv(fused)
            # out = veh_bev_veh_view
        elif self.fusionType == 'attention':
            infra_flat = cloud_bev_veh_view.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            vehicle_flat = veh_bev_veh_view.flatten(2).permute(0, 2, 1)
            fused, _ = self.crossAttn(infra_flat, vehicle_flat, vehicle_flat)
            out = fused.permute(0, 2, 1).view(B, C, h, w)
        else:
            raise NotImplementedError
        B, C, H, W = out.shape
        return out.permute(0, 2, 3, 1).reshape(B, H * W, C)


if __name__ == '__main__':
    B = 2  # Batch Size
    C = 256
    H = W = 150

    # Input data
    vehicleBEV = torch.randn(B, H * W, C)
    infrastructureBEV = torch.randn(B, H * W, C)

    # Rotation and translation for each batch item
    R_list = [
        torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Batch item 1: identity rotation
        torch.tensor([[0.9962, -0.0872, 0], [0.0872, 0.9962, 0], [0, 0, 1]])  # Batch item 2: five-degree rotation about z
    ]
    T_list = [
        torch.tensor([0.0, 0, 0]),  # Batch item 1: zero translation
        torch.tensor([10.0, 0, 0])  # Batch item 2: ten-meter translation along x
    ]

    # Create the module.
    bev_fuser = BEVFuser(fusionType='attention')

    # Run fusion.
    fused_bev = bev_fuser(vehicleBEV, infrastructureBEV, R_list, T_list)
    print("Fused BEV Shape:", fused_bev.shape)  # [B, C, H, W]
