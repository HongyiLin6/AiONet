"""Synthetic interface checks; these do not train or evaluate a dataset."""
import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
mmcv=pytest.importorskip('mmcv')
pytest.importorskip('mmdet3d')
ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT/'tools',ROOT/'training/scripts',ROOT/'training/vendor/metrovision']:
    sys.path.insert(0,str(p))
from paper import detector_configs


def test_generated_car_configs_cover_both_sides_and_fixed_training_budget(tmp_path):
    (tmp_path/'configs').mkdir();detector_configs(tmp_path)
    files=list((tmp_path/'configs').glob('*.py'));assert len(files)==6
    for file in files:
        cfg=mmcv.Config.fromfile(str(file))
        role='vehicle' if file.stem.endswith('_vehicle') else 'infrastructure'
        assert cfg.model.pts_bbox_head.num_classes==1
        assert cfg.data.train.classes==['Car']
        assert Path(cfg.data.train.ann_file).name==role+'_perception_train.pkl'
        assert cfg.runner.max_epochs==24 and cfg.paper_checkpoint_policy=='final_epoch'
        assert cfg.store_queries


@pytest.mark.parametrize('operator',['query_concat','box_mlp','bev_concat','bev_attention','bev_query','bev_conv','adaptive'])
def test_recovered_decoder_accepts_reconstructed_queries_and_empty_targets(tmp_path,operator):
    from paper_model import FullPaperFusion
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    (tmp_path/'configs').mkdir();detector_configs(tmp_path)
    cfg=mmcv.Config.fromfile(str(tmp_path/'configs/bevformer_vehicle.py'))
    cfg.model.pts_bbox_head.num_query=8;cfg.model.pts_bbox_head.bev_h=10;cfg.model.pts_bbox_head.bev_w=10
    model=FullPaperFusion(cfg,operator)
    def packet():
        xyz=torch.rand(2,4,3)*10
        return dict(query=torch.randn(2,4,256),xyz=xyz,boxes=torch.cat((xyz,torch.ones(2,4,3),torch.zeros(2,4,1)),-1),scores=torch.ones(2,4)*.7,velocity=torch.zeros(2,4,2))
    values=model(packet(),packet(),torch.eye(4)[None].repeat(2,1,1),torch.tensor([.1,.2]))
    scores=values['all_cls_scores'];assert scores.ndim==4 and scores.shape[1]==2 and scores.shape[3]==1
    assert values['all_bbox_preds'][-1,...,[0,1,4]].shape==(2,scores.shape[2],3)
    boxes=[LiDARInstance3DBoxes(torch.zeros((0,9)),box_dim=9),LiDARInstance3DBoxes(torch.tensor([[10.,1.,0.,4.,2.,1.5,0.,0.,0.]]),box_dim=9)]
    labels=[torch.zeros(0,dtype=torch.long),torch.zeros(1,dtype=torch.long)]
    loss=sum(model.head.loss(boxes,labels,values).values())
    assert torch.isfinite(loss);loss.backward()
    decoded=model.decode(values)
    assert len(decoded)==2 and all(p.boxes.shape[1]==7 for p in decoded)
