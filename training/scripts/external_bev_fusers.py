"""Explicit camera-BEV adaptations of the authors' public fusion modules."""
import sys
import torch
from torch import nn
from torch.nn import functional as F
from experiment_io import ROOT
from r2_bev_components import warp_bev

class V2XViTFusion(nn.Module):
    def __init__(self):
        super().__init__();sys.path.insert(0,str(ROOT/'src/baselines/v2x-vit'))
        from v2xvit.models.sub_modules.v2xvit_basic import V2XTransformer
        self.projection=nn.Conv2d(256,256,1)
        self.transformer=V2XTransformer(dict(encoder=dict(num_blocks=1,depth=3,use_roi_mask=True,
            cav_att_config=dict(dim=256,use_hetero=True,use_RTE=True,RTE_ratio=2,heads=8,dim_head=32,dropout=.3),
            pwindow_att_config=dict(dim=256,heads=[16,8,4],dim_head=[16,32,64],dropout=.3,window_size=[4,8,16],relative_pos_embedding=True,fusion_method='split_attn'),
            feed_forward=dict(mlp_dim=256,dropout=.3),sttf=dict(voxel_size=[102.4/150],downsample_rate=1))))
    def forward(self,ego,source,transform,age,predicted,context=None):
        aligned,_=warp_bev(self.projection(source.contiguous()),transform)
        b,_,h,w=ego.shape;assert (h,w)==(150,150)
        maps=torch.stack((F.pad(ego,(0,10,0,10)),F.pad(aligned,(0,10,0,10))),1).permute(0,1,3,4,2).contiguous()
        prior=maps.new_zeros(b,2,160,160,3)
        prior[:,1,:,:,1]=torch.clamp(torch.round(age/.1),0,49)[:,None,None]
        prior[:,1,:,:,2]=1.
        # The common metric warp already brings inputs to ego coordinates.
        # Identity STTF avoids applying the same alignment twice.
        spatial=torch.eye(4,device=ego.device,dtype=ego.dtype)[None,None].repeat(b,2,1,1)
        value=self.transformer(torch.cat((maps,prior),-1),torch.ones(b,2,device=ego.device),spatial)
        return value.permute(0,3,1,2)[:,:,:150,:150],None

class Select2ColFusion(nn.Module):
    def __init__(self):
        super().__init__();sys.path.insert(0,str(ROOT/'src/baselines/Select2Col'))
        from opencood.models.fuse_modules.HPHA_fuse import HPHA
        from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
        self.projection=nn.Conv2d(256,256,1)
        self.backbone=BaseBEVBackbone(dict(layer_nums=[3,5,8],layer_strides=[2,2,2],num_filters=[64,128,256],
            upsample_strides=[1,2,4],num_upsample_filter=[128,128,128]),256)
        self.hpha=HPHA(dict(fully=False,voxel_size=[102.4/150],downsample_rate=1,in_channels=256,multi_scale=True,
            layer_nums=[3,5,8],num_filters=[64,128,256],sparse=dict(threshold=.01,gaussian_smooth=dict(k_size=5,c_sigma=1.))))
        self.output=nn.Conv2d(384,256,1)
    def forward(self,ego,source,transform,age,predicted,context=None):
        if context is None:raise ValueError('Select2Col requires two real ego-history maps and detector confidence maps')
        source,_=warp_bev(self.projection(source.contiguous()),transform)
        b,_,h,w=ego.shape;history=context['history'];ht=context['history_to_ego']
        history=warp_bev(history.reshape(b*2,256,h,w),ht.reshape(b*2,4,4))[0].reshape(b,2,256,h,w)
        confidence=context['confidence'];maps=[]
        transforms=torch.cat((torch.eye(4,device=ego.device)[None,None].repeat(b,1,1,1),ht,transform[:,None]),1)
        confidence=warp_bev(confidence.reshape(b*4,1,h,w),transforms.reshape(b*4,4,4))[0].reshape(b,4,1,h,w)
        for index in range(b):
            x=F.pad(torch.stack((ego[index],source[index])),(0,10,0,10));past=F.pad(history[index],(0,10,0,10))
            prob=F.pad(confidence[index],(0,10,0,10)).clamp(1e-5,1-1e-5);logits=torch.log(prob/(1-prob))
            pairwise=torch.eye(4,device=ego.device)[None,None,None].repeat(1,2,2,1,1)
            output=self.hpha(x,past,logits,torch.tensor([2],device=ego.device),pairwise,context['ages'][index:index+1],self.backbone)
            # Crop padding after the twofold spatial downsampling, then restore
            # the shared 150x150 decoder grid with a learned channel projection.
            maps.append(F.interpolate(self.output(output)[:,:,:75,:75],size=(150,150),mode='bilinear',align_corners=False))
        return torch.cat(maps),None
