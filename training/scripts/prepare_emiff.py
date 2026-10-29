"""Vendor the public EMIFF modules without replacing installed MMDetection3D."""
import copy
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
import mmcv
import numpy as np
from experiment_io import ROOT,records,gt_roi,sha,atomic_json
from structured_fusion import CLASSES

def main():
    source=ROOT/'src/baselines/EMIFF';out=ROOT/'generated/r2_emiff';out.mkdir(parents=True,exist_ok=True)
    paths={'vimi.py':'mmdet3d/models/detectors/vicfuser_voxel/vimi.py',
        'ccm.py':'mmdet3d/models/detectors/vicfuser_voxel/vicfuser_voxel_ccm.py',
        'naive_compressor.py':'mmdet3d/models/model_utils/naive_compressor.py',
        'dcn.py':'mmdet3d/models/necks/dcn.py'}
    changes={}
    for dest,original in paths.items():
        text=(source/original).read_text()
        if dest=='vimi.py':
            text=text.replace('from mmdet3d.models.model_utils.naive_compressor','from r2_emiff.naive_compressor')
            text=text.replace('from mmdet3d.models.detectors.vicfuser_voxel.vicfuser_voxel_ccm','from r2_emiff.ccm')
        if dest=='ccm.py':text=text[:text.index('@DETECTORS.register_module()')]
        if dest=='dcn.py':text=text.replace('from ..builder import NECKS','from mmdet3d.models.builder import NECKS')
        (out/dest).write_text(text);changes[dest]={'official_path':original,'original_sha256':sha(source/original),'adapted_sha256':sha(out/dest)}
    (out/'__init__.py').write_text('from . import dcn\nfrom .vimi import VIMI\n')
    for name in ['LICENSE','LICENSE.txt']:
        if (source/name).exists():shutil.copy2(source/name,out/name)
    checkpoint=ROOT/'weights/resnet50-0676ba61.pth'
    if not checkpoint.exists():
        with urllib.request.urlopen('https://download.pytorch.org/models/resnet50-0676ba61.pth',timeout=90) as response:
            temp=checkpoint.with_suffix('.pth.downloading')
            with temp.open('wb') as f:shutil.copyfileobj(response,f)
        assert sha(temp).startswith('0676ba61');temp.replace(checkpoint)
    assert sha(checkpoint).startswith('0676ba61')
    cfg=mmcv.Config.fromfile(str(source/'cfgs/vic/vimi_960x540_12e_bs2.py'));m=copy.deepcopy(cfg.model)
    m.backbone.init_cfg.checkpoint=str(checkpoint);m.n_voxels=[128,256,12]
    m.anchor_generator.ranges=[[0,-51.2,-5,51.2,51.2,3]]
    arrays=[gt_roi(r) for r in records('cooperative','perception_train')]
    boxes=np.concatenate([x[0] for x in arrays]);labels=np.concatenate([x[1] for x in arrays]);anchors=[];ranges=[]
    for label in range(7):
        b=boxes[labels==label];assert len(b)>0
        anchors.append(np.median(b[:,3:6],axis=0).tolist());z=float(np.median(b[:,2]-b[:,5]/2))
        ranges.append([0,-51.2,z,51.2,51.2,z])
    m.bbox_head.num_classes=7;m.bbox_head.anchor_generator.sizes=anchors;m.bbox_head.anchor_generator.ranges=ranges
    m.test_cfg.update(score_thr=.001,nms_pre=1000,max_num=300,nms_thr=.1)
    result=mmcv.Config(dict(model=m));result.dump(str(ROOT/'configs/emiff.py'))
    atomic_json(ROOT/'manifests/emiff_adaptation.json',dict(source_commit='4cea36ea9ecb4dfc73ece47b8f446fbdb179e290',modules=changes,
        checkpoint_sha256=sha(checkpoint),config_sha256=sha(ROOT/'configs/emiff.py'),classes=CLASSES,
        adaptation='camera-only SPD; common 7-class vocabulary and ROI; training-only median anchors; 128x256x12 x-y-z voxel grid; original MCA, CCM and feature compression',
        train_split_sha256=sha(ROOT/'manifests/split.json'),image_size=[540,960],normalization='RGB ImageNet TorchVision',
        training=dict(epochs=12,batch_size=4,lr=.0002,weight_decay=.0001,seed=17,checkpoint_selection='development AP25 every two epochs'),
        pretrained_on='ImageNet only'))
    print('Prepared EMIFF camera-only SPD adaptation',flush=True)

if __name__=='__main__':main()
