"""Train registered fusion on Car caches with a fixed epoch budget."""
import argparse,json,time
import numpy as np
import torch
import mmcv
from torch.utils.data import DataLoader
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from experiment_io import ROOT,sha,atomic_json
from aionet.paper.experiments import get_experiment
from aionet.paper.training_state import seed_epoch,save_checkpoint
from paper_model import FullPaperFusion
from paper_data import PaperPairs,collate,forward

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    spec=get_experiment(a.experiment)
    if spec['operator'].endswith('_only'):raise ValueError('Single-detector controls use evaluate directly')
    if spec.get('temporal')=='fixed_delay':spec['delay_ms']=36.4+71.47+81.23
    torch.manual_seed(spec['seed']);np.random.seed(spec['seed'])
    data=PaperPairs(spec,'perception_train');loader=DataLoader(data,batch_size=spec['batch_size'],shuffle=True,num_workers=4,collate_fn=collate)
    model=FullPaperFusion(mmcv.Config.fromfile(str(ROOT/'configs/bevformer_vehicle.py')),spec['operator']).cuda()
    opt=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=spec['learning_rate'],weight_decay=.01)
    out=ROOT/'runs/paper'/a.experiment;out.mkdir(parents=True,exist_ok=True)
    provenance=dict(spec=spec,ego_cache_sha256=data.ego.fingerprint,source_cache_sha256=data.source.fingerprint,split_sha256=sha(ROOT/'manifests/split.json'),implementation='full-paper independent reconstruction')
    start=0
    if (out/'latest.pt').exists():
        if not a.resume:raise ValueError('Existing training run; pass --resume or use a new workspace')
        saved=torch.load(out/'latest.pt',map_location='cpu');assert saved['provenance']==provenance
        model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer']);start=saved['epoch']
    for epoch in range(start,spec['epochs']):
        seed_epoch(spec['seed'],epoch)
        model.train();total=0.;count=0;begin=time.monotonic()
        for batch in loader:
            opt.zero_grad(set_to_none=True);outputs=forward(model,batch);gt=[];labels=[]
            for boxes,vel,lab in zip(batch['boxes'],batch['velocity'],batch['labels']):
                values=np.concatenate((boxes,np.nan_to_num(vel)),1).astype(np.float32)
                gt.append(LiDARInstance3DBoxes(torch.from_numpy(values).cuda(),box_dim=9,origin=(.5,.5,.5)))
                labels.append(torch.as_tensor(lab,device='cuda'))
            loss=sum(model.head.loss(gt,labels,outputs).values())
            if not torch.isfinite(loss):raise ValueError('Nonfinite fusion loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),35.);opt.step()
            total+=loss.item()*len(gt);count+=len(gt)
        save_checkpoint(out/'latest.pt',dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,provenance=provenance))
        print(json.dumps(dict(epoch=epoch+1,training_loss=total/max(count,1),seconds=time.monotonic()-begin)),flush=True)
    save_checkpoint(out/'final.pt',dict(state_dict=model.state_dict(),epoch=spec['epochs'],provenance=provenance))
    atomic_json(out/'complete.json',dict(provenance=provenance,checkpoint_sha256=sha(out/'final.pt')))

if __name__=='__main__':main()
