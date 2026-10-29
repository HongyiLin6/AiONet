"""Registered BEV fusion and inherited object decoder for new experiments."""
import copy
import torch
from torch import nn
from r2_bev_components import AdaptiveBEVFusion

class RegisteredFusion(nn.Module):
    def __init__(self,cfg,vehicle_checkpoint=None,method='aionet'):
        super().__init__()
        from mmdet.models import build_head
        import r2_plugin
        head=copy.deepcopy(cfg.model.pts_bbox_head);head.type='metrovisionHead'
        head.train_cfg=copy.deepcopy(cfg.model.train_cfg.pts)
        self.head=build_head(head);self.head.init_weights();self.fusion=AdaptiveBEVFusion()
        self.method=method
        if method!='aionet':
            from external_bev_fusers import V2XViTFusion,Select2ColFusion
            self.fusion={'v2xvit':V2XViTFusion,'select2col':Select2ColFusion}[method]()
        if vehicle_checkpoint is not None:
            state=torch.load(vehicle_checkpoint,map_location='cpu')['state_dict']
            state={k[len('pts_bbox_head.'):]:v for k,v in state.items() if k.startswith('pts_bbox_head.')}
            self.head.load_state_dict(state,strict=True)
        with torch.no_grad():self.head.code_weights[-2:]=0.
        # Cached features bypass the image/BEV encoder. Only decoder-side weights
        # and the registered adapter are fitted in this new cooperative study.
        for name,p in self.head.named_parameters():
            used=name.startswith(('query_embedding.','cls_branches.','reg_branches.','transformer.decoder.','transformer.reference_points.'))
            p.requires_grad_(used)
    def forward(self,ego,source,transform,age,predicted,context=None):
        extra={'context':context} if self.method!='aionet' else {}
        fused,weights=self.fusion(ego,source,transform,age,predicted,**extra)
        outputs=self.head('BEV',bev_embed=fused.flatten(2).permute(0,2,1).float())
        return outputs,weights
    def decode(self,outputs):
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from structured_fusion import Detections
        batch=outputs['all_cls_scores'].shape[1]
        values=self.head.get_bboxes(outputs,[{'box_type_3d':LiDARInstance3DBoxes} for _ in range(batch)])
        return [Detections(torch.cat((b.gravity_center,b.tensor[:,3:7]),1).detach().cpu().numpy(),
                           s.detach().cpu().numpy(),l.detach().cpu().numpy(),
                           b.tensor[:,7:9].detach().cpu().numpy() if b.tensor.shape[1]>=9 else None).roi() for b,s,l in values]
