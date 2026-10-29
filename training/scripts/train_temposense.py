"""Train the documented new TempoSense reconstruction on frozen real features."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from r2_bev_components import TempoSense,warp_bev

ROOT=Path(os.environ.get('AIONET_ROOT',str(Path(__file__).resolve().parents[2] / 'workdir')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Sequences(Dataset):
    def __init__(self,partition):
        self.root=ROOT/'cache/detectors/bevformer_infrastructure'/partition
        self.manifest_path=self.root/'manifest.json';self.manifest=json.loads(self.manifest_path.read_text())
        self.rows=self.manifest['records'];self.windows=[]
        for index in range(3,len(self.rows)):
            window=self.rows[index-3:index+1]
            if len({r['scenario'] for r in window})!=1:continue
            gaps=np.diff([r['observation_us'] for r in window])
            if (gaps<50000).any() or (gaps>150000).any():continue
            self.windows.append(index)
        if not self.windows:raise ValueError('No causal four-frame training windows')
    def __len__(self):return len(self.windows)
    def __getitem__(self,index):
        rows=self.rows[self.windows[index]-3:self.windows[index]+1]
        maps=np.stack([np.load(self.root/r['feature'],allow_pickle=False).astype(np.float32) for r in rows])
        target=np.linalg.inv(np.array(rows[2]['lidar2global'],np.float64))
        transforms=np.stack([target@np.array(r['lidar2global'],np.float64) for r in rows]).astype(np.float32)
        return torch.from_numpy(maps),torch.from_numpy(transforms)

def align(maps,transforms):
    return torch.stack([warp_bev(maps[:,i],transforms[:,i])[0] for i in range(4)],1)

def run():
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    torch.manual_seed(17);np.random.seed(17)
    train=Sequences('perception_train');dev=Sequences('development')
    assert train.manifest['checkpoint_sha256']==dev.manifest['checkpoint_sha256']
    out=ROOT/'runs/temposense';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('TempoSense already complete');return
    model=TempoSense().cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(init_scale=128.)
    training=DataLoader(train,batch_size=4,shuffle=True,num_workers=4,pin_memory=True,drop_last=False)
    validation=DataLoader(dev,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)
    best=float('inf');best_epoch=0;history=[]
    provenance={'kind':'new_experiment','source_checkpoint_sha256':train.manifest['checkpoint_sha256'],
        'train_manifest_sha256':sha(train.manifest_path),'development_manifest_sha256':sha(dev.manifest_path),
        'split_sha256':sha(ROOT/'manifests/split.json'),'train_windows':len(train),'development_windows':len(dev),
        'seed':17,'max_epochs':6,'patience':2,'learning_rate':2e-4,'batch_size':4,
        'implementation':'three-map convolution, window self-attention, zero-initialized residual output; 150x150x256 BEV',
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
    start_epoch=0
    if (out/'latest.pt').exists():
        state=torch.load(out/'latest.pt',map_location='cpu');assert state['provenance']==provenance
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];history=state['history']
    for epoch in range(start_epoch,6):
        model.train();total=0.;count=0;started=time.monotonic()
        for maps,transforms in training:
            maps=align(maps.cuda(non_blocking=True),transforms.cuda(non_blocking=True));opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():pred=model(maps[:,:3]);loss=(pred.float()-maps[:,3]).abs().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite temporal loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),35.)
            scaler.step(opt);scaler.update();total+=loss.item()*len(maps);count+=len(maps)
        model.eval();valid_total=0.;valid_count=0
        with torch.no_grad():
            for maps,transforms in validation:
                maps=align(maps.cuda(non_blocking=True),transforms.cuda(non_blocking=True))
                with torch.cuda.amp.autocast():pred=model(maps[:,:3])
                valid_total+=(pred.float()-maps[:,3]).abs().mean().item()*len(maps);valid_count+=len(maps)
        value=valid_total/valid_count
        row={'epoch':epoch+1,'train_l1':total/count,'development_l1':value,'seconds':time.monotonic()-started}
        history.append(row);print(json.dumps(row),flush=True)
        if value<best:
            best=value;best_epoch=epoch+1
            torch.save({'state_dict':model.state_dict(),'provenance':provenance,'epoch':best_epoch,'development_l1':best},out/'best.pt')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),provenance=provenance,
                        epoch=epoch+1,best=best,best_epoch=best_epoch,history=history),out/'latest.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        if epoch+1-best_epoch>=2:break
    (out/'complete.json').write_text(json.dumps(dict(provenance=provenance,best_epoch=best_epoch,development_l1=best,checkpoint_sha256=sha(out/'best.pt'),completed_unix=time.time()),indent=2))

if __name__=='__main__':run()
