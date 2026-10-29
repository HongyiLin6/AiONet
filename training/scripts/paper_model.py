"""Reconstructed registered-pair fusion using the recovered object decoder."""
import copy
import torch
from torch import nn
from aionet.paper.operators import PaperFusion,PC_RANGE

class FullPaperFusion(nn.Module):
    def __init__(self,cfg,operator):
        super().__init__()
        import r2_plugin
        from mmdet.models import build_head
        head=copy.deepcopy(cfg.model.pts_bbox_head);head.type='metrovisionHead';head.train_cfg=copy.deepcopy(cfg.model.train_cfg.pts)
        self.head=build_head(head);self.head.init_weights()
        self.fusion=PaperFusion(operator,channels=256,height=head.bev_h,width=head.bev_w)
        for name,p in self.head.named_parameters():
            p.requires_grad_(name.startswith(('query_embedding.','cls_branches.','reg_branches.','transformer.decoder.','transformer.reference_points.')))
    def forward(self,ego,source,matrix,age,predicted=None):
        packet=self.fusion(ego,source,matrix,age,predicted)
        if 'bev' in packet:return self.head('BEV',bev_embed=packet['bev'].flatten(2).transpose(1,2))
        q=packet['query'];xyz=packet['xyz'];lower=q.new_tensor(PC_RANGE[:3]);span=q.new_tensor(PC_RANGE[3:])-lower
        references=((xyz-lower)/span).clamp(1e-5,1-1e-5)
        states=q.transpose(0,1).unsqueeze(0)
        return self.head.forwardQuery((None,states,references,references.unsqueeze(0)))
    def decode(self,outputs):
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from aionet.structured_fusion import Detections
        batch=outputs['all_cls_scores'].shape[1]
        maximum=self.head.bbox_coder.max_num
        self.head.bbox_coder.max_num=min(maximum,outputs['all_cls_scores'].shape[2]*outputs['all_cls_scores'].shape[3])
        try:values=self.head.get_bboxes(outputs,[{'box_type_3d':LiDARInstance3DBoxes} for _ in range(batch)])
        finally:self.head.bbox_coder.max_num=maximum
        result=[]
        for boxes,score,label in values:
            raw=torch.cat((boxes.gravity_center,boxes.tensor[:,3:7]),1).detach().cpu().numpy()
            velocity=boxes.tensor[:,7:9].detach().cpu().numpy() if boxes.tensor.shape[1]>=9 else None
            result.append(Detections(raw,score.detach().cpu().numpy(),label.detach().cpu().numpy(),velocity).roi())
        return result
