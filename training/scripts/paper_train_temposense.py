"""Train TempoSense on real roadside BEV sequences with a fixed budget."""
import argparse,json
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from experiment_io import ROOT,DetectorCache,sha,atomic_json
from aionet.r2_bev_components import TempoSense,warp_bev
from aionet.paper.training_state import seed_epoch,save_checkpoint

class History(Dataset):
    def __init__(self,source):
        self.cache=DetectorCache(source,'perception_train');self.windows=[]
        for rows in self.cache.by_scene.values():
            for i in range(3,len(rows)):
                window=rows[i-3:i+1];gaps=np.diff([r['observation_us'] for r in window])
                if (gaps>0).all() and (gaps<=150000).all():self.windows.append(window)
        if not self.windows:raise ValueError('Provide chronological BEV caches containing four-frame real histories')
    def __len__(self):return len(self.windows)
    def __getitem__(self,index):
        rows=self.windows[index]
        if any(not r.get('feature') for r in rows):raise ValueError('TempoSense training requires cached BEV maps from the registered source adapter')
        maps=np.stack([self.cache.feature(r) for r in rows]);inverse=np.linalg.inv(np.asarray(rows[2]['lidar2global']))
        matrices=np.stack([inverse@np.asarray(r['lidar2global']) for r in rows]).astype(np.float32)
        return torch.from_numpy(maps),torch.from_numpy(matrices)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='bevformer_infrastructure');p.add_argument('--epochs',type=int,default=6);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.epochs<1:p.error('--epochs must be positive')
    torch.manual_seed(17);np.random.seed(17);data=History(a.source)
    model=TempoSense().cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    out=ROOT/'runs/paper_temposense'/a.source;out.mkdir(parents=True,exist_ok=True)
    if (out/'final.pt').exists():raise ValueError('Use a new workspace for another temporal fit')
    provenance=dict(source=a.source,epochs=a.epochs,seed=17,cache_sha256=data.cache.fingerprint,split_sha256=sha(ROOT/'manifests/split.json'),checkpoint_policy='fixed final epoch')
    start=0;history=[]
    if (out/'latest.pt').exists():
        if not a.resume:raise ValueError('An interrupted temporal fit exists; pass --resume to continue it')
        state=torch.load(out/'latest.pt',map_location='cpu')
        if state['provenance']!=provenance:raise ValueError('Temporal checkpoint inputs differ from this run')
        model.load_state_dict(state['state_dict']);opt.load_state_dict(state['optimizer']);start=state['epoch'];history=state['history']
    elif a.resume:raise FileNotFoundError(out/'latest.pt')
    for epoch in range(start,a.epochs):
        seed_epoch(17,epoch)
        total=0.;count=0
        for maps,matrices in DataLoader(data,batch_size=4,shuffle=True,num_workers=4):
            maps=maps.cuda();matrices=matrices.cuda();maps=torch.stack([warp_bev(maps[:,i],matrices[:,i])[0] for i in range(4)],1)
            opt.zero_grad(set_to_none=True);loss=(model(maps[:,:3])-maps[:,3]).abs().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite temporal loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),35.);opt.step();total+=loss.item()*len(maps);count+=len(maps)
        row=dict(epoch=epoch+1,training_loss=total/max(1,count));history.append(row)
        save_checkpoint(out/'latest.pt',dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,history=history,provenance=provenance))
        print(json.dumps(row),flush=True)
    save_checkpoint(out/'final.pt',dict(state_dict=model.state_dict(),provenance=provenance));atomic_json(out/'complete.json',provenance)

if __name__=='__main__':main()
