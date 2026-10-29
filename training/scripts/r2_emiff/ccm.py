# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet.models.detectors import BaseDetector
from mmdet3d.models.builder import DETECTORS, build_backbone, build_head, build_neck
from torch import nn
from mmcv.cnn import ConvModule
from torch.nn import functional as F
from mmcv.runner import force_fp32, auto_fp16
import math
import numpy as np


def attention(query, key, mask=None, dropout=None):

    # from IPython import embed
    # embed()

    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn

class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# class Weights_MLP(nn.Module):

#     def __init__(self, in_channels,num_levels):
#         super(Weights_MLP, self).__init__()

#         self.w_mlp = nn.Sequential(
#             nn.Linear(in_channels * num_levels, num_levels),
#             # nn.BatchNorm1d(num_levels, eps=1e-3, momentum=0.01),
#         )

#     def forward(self, x):
#         B,N_levels,C,_H,W = x.shape
#         input = torch.mean(x,dim=(-2,-1)).view(B,N_levels * C)
#         weights = self.w_mlp(input)
#         weights = F.softmax(weights,dim=-1)
#         return weights


class DCN_Up_Conv_List(nn.Module):

    def __init__(self, neck_dcn, channels):
        super(DCN_Up_Conv_List, self).__init__()


        self.upconv0 = nn.Sequential(
            double_conv(channels,channels),
        )

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            double_conv(channels,channels),
        )

        self.dcn0 = build_neck(neck_dcn)
        self.dcn1 = build_neck(neck_dcn)
        self.dcn2 = build_neck(neck_dcn)
        self.dcn3 = build_neck(neck_dcn)

    def forward(self, x):
        assert x.__len__() == 4
        x0 = self.dcn0(x[0])
        x0 = self.upconv0(x0)

        x1 = self.dcn1(x[1])
        x1 = self.upconv1(x1)

        x2 = self.dcn2(x[2])
        x2 = self.upconv2(x2)

        x3 = self.dcn3(x[3])
        x3 = self.upconv3(x3)


        return [x0,x1,x2,x3]


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super(SELayer,self).__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):

        # from IPython import embed 
        # embed(header='SELayer')
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class SE_Inception_Layer(nn.Module):
    def __init__(self, channels, reduction_ratio=1, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super(SE_Inception_Layer,self).__init__()
        self.conv_reduce = nn.Linear(channels, channels//reduction_ratio)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels//reduction_ratio, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):

        # from IPython import embed 
        # embed(header='SE_Inception_Layer')
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)[...,None,None]


class CASELayer(nn.Module):
    def __init__(self, channels, reduction_ratio=1, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super(CASELayer,self).__init__()
        self.FC_reduce = nn.Linear(channels, channels//reduction_ratio)
        self.act1 = act_layer()
        self.FC_expand = nn.Linear(channels//reduction_ratio, channels)
        self.FC_out = nn.Linear(channels*2, channels)
        self.gate = gate_layer()
        self.max_pooling = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x, x_se):

        # from IPython import embed 
        # embed(header='CASELayer')

        # x_channel = torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        x_channel = self.max_pooling(x)
        x_se = self.FC_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.FC_expand(x_se)
        x_se = torch.cat((x_channel[...,0,0], x_se), dim=1)
        x_se = self.FC_out(x_se)

        return x * self.gate(x_se)[...,None,None]

class CCMNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, reduction_ratio=1):
        super(CCMNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(16)
        self.context_mlp = Mlp(16, mid_channels, mid_channels)
        self.context_se = SE_Inception_Layer(mid_channels,reduction_ratio=reduction_ratio)  # NOTE: add camera-aware

        # self.context_se = CASELayer(mid_channels,reduction_ratio=8)  # NOTE: add camera-aware
    
    def ida_mat_cal(self,img_meta):
        img_scale_factor = (img_meta['scale_factor'][:2]
                if 'scale_factor' in img_meta.keys() else 1)

        img_shape = img_meta['img_shape'][:2]
        orig_h, orig_w = img_shape

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        ida_rot *= img_scale_factor
        # ida_tran -= torch.Tensor(crop[:2])
        if 'flip' in img_meta.keys() and img_meta['flip']:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([orig_w, 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b

        ida_mat = ida_rot.new_zeros(4, 4)
        ida_mat[3, 3] = 1
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 3] = ida_tran

        return ida_mat

    def forward(self, x_v, x_i, img_metas):
        # x [bs,num_cams,C,H,W]
        bs, C, H, W = x_v.shape
        num_cams = 2

        x = torch.stack((x_v,x_i),dim=1).reshape(-1, C, H, W)


        extrinsic_v_list = list()
        extrinsic_i_list = list()
        intrinsic_v_list = list()
        intrinsic_i_list = list()
        for img_meta in img_metas:

            extrinsic_v = torch.Tensor(img_meta['lidar2img']['extrinsic'][0])
            extrinsic_i = torch.Tensor(img_meta['lidar2img']['extrinsic'][1])
            intrinsic_v = torch.Tensor(img_meta['lidar2img']['intrinsic'][0])
            intrinsic_i = torch.Tensor(img_meta['lidar2img']['intrinsic'][1])
            # from IPython import embed
            # embed(header='ida')
            ida_mat = self.ida_mat_cal(img_meta)

            intrinsic_v = ida_mat @ intrinsic_v
            intrinsic_i = ida_mat @ intrinsic_i

            extrinsic_v_list.append(extrinsic_v)
            extrinsic_i_list.append(extrinsic_i)
            intrinsic_v_list.append(intrinsic_v)
            intrinsic_i_list.append(intrinsic_i)

            

        extrinsic_v = torch.stack(extrinsic_v_list)
        extrinsic_i = torch.stack(extrinsic_i_list)
        intrinsic_v = torch.stack(intrinsic_v_list)
        intrinsic_i = torch.stack(intrinsic_i_list)

        extrinsic = torch.stack((extrinsic_v,extrinsic_i),dim=1) 
        intrinsic = torch.stack((intrinsic_v,intrinsic_i),dim=1) 

        in_mlp = torch.stack(
                    (
                        intrinsic[..., 0, 0],
                        intrinsic[..., 1, 1],
                        intrinsic[..., 0, 2],
                        intrinsic[ ..., 1, 2],
                    ),
                    dim=-1
                )

        # from IPython import embed
        # embed(header='DCMNet')
        ex_mlp = extrinsic[...,:3,:].view(bs,num_cams,-1)
        mlp_input = torch.cat((in_mlp,ex_mlp),dim=-1)
        mlp_input = mlp_input.reshape(-1,mlp_input.shape[-1]).to(x.device)

        mlp_input = self.bn(mlp_input)
        x = self.reduce_conv(x)
        # context_se = self.context_mlp(mlp_input)[..., None, None]
        context_se = self.context_mlp(mlp_input)
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        context = context.reshape(bs,num_cams,C,H,W)
        x_v_out = context[:,0,...]
        x_i_out = context[:,1,...]

        # from IPython import embed
        # embed(header='DCMNet end')
        return tuple((x_v_out, x_i_out))



