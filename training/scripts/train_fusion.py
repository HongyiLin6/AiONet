"""Fit the registered fusion on perception-only scenes; select on development."""
import bisect
import argparse
import fcntl
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from experiment_io import ROOT,DetectorCache,records,source_to_ego,gt_roi,sha,atomic_json

class Pairs(Dataset):
    def __init__(self,partition,method='aionet'):
        self.rows=records('cooperative',partition)
        self.ego=DetectorCache('bevformer_vehicle',partition)
        self.source=DetectorCache('bevformer_infrastructure',partition)
        self.partition=partition
        self.method=method
        if method=='select2col':
            earliest={sid:scene[2]['observation_us'] for sid,scene in self.ego.by_scene.items() if len(scene)>=3}
            self.rows=[r for r in self.rows if r['sequence_id'] in earliest and r['timestamp']>=earliest[r['sequence_id']]]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];e=self.ego.by_token[r['vehicle_token']]
        scene=self.source.by_scene[r['sequence_id']]
        times=[s['observation_us'] for s in scene]
        # Only real source observations at or before the ego capture enter this
        # training pair. Random stale observations span 0--5 available frames.
        latest=bisect.bisect_right(times,r['timestamp'])-1
        if latest<0:return None
        lag=np.random.randint(0,6) if self.partition=='perception_train' else 2
        source=scene[max(0,latest-lag)]
        b,l=gt_roi(r)
        result=dict(ego=torch.from_numpy(self.ego.feature(e)),source=torch.from_numpy(self.source.feature(source)),
            transform=torch.from_numpy(source_to_ego(source,r)),age=(r['timestamp']-source['observation_us'])/1e6,
            boxes=b,labels=l,index=index)
        if self.method=='select2col':
            from select2col_inputs import context
            result['context']=context(self.ego,self.source,e,source)
        return result

def collate(rows):
    rows=[r for r in rows if r is not None]
    if not rows:return None
    result={k:torch.stack([r[k] for r in rows]) for k in ['ego','source','transform']}
    result.update(age=torch.tensor([r['age'] for r in rows]),boxes=[r['boxes'] for r in rows],
        labels=[r['labels'] for r in rows],index=[r['index'] for r in rows])
    if 'context' in rows[0]:result['context']={k:torch.stack([r['context'][k] for r in rows]) for k in rows[0]['context']}
    return result

def forward(model,b):
    context={k:v.cuda() for k,v in b['context'].items()} if 'context' in b else None
    return model(b['ego'].cuda(),b['source'].cuda(),b['transform'].cuda(),b['age'].cuda(),torch.zeros_like(b['age']).cuda(),context=context)[0]

def main(args):
    import mmcv
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from r2_fusion_model import RegisteredFusion
    from evaluation_metrics import summarize
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    torch.manual_seed(17);np.random.seed(17)
    train=Pairs('perception_train',args.method);dev=Pairs('development',args.method)
    ckpt=train.ego.manifest['checkpoint'];cfg=mmcv.Config.fromfile(str(ROOT/'configs/bevformer_vehicle.py'))
    assert train.ego.manifest['checkpoint_sha256']==dev.ego.manifest['checkpoint_sha256']
    assert train.source.manifest['checkpoint_sha256']==dev.source.manifest['checkpoint_sha256']
    out=ROOT/'runs'/('registered_fusion' if args.method=='aionet' else args.method);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('Fusion complete');return
    model=RegisteredFusion(cfg,ckpt,method=args.method).cuda()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    provenance=dict(kind='new_experiment',method=args.method,split_sha256=sha(ROOT/'manifests/split.json'),
        head_code_sha256=sha(ROOT/'src/metrovision/projects/mmdet3d_plugin/CoreModule/Heads/AiONetHead.py'),
        ego_checkpoint_sha256=train.ego.manifest['checkpoint_sha256'],source_checkpoint_sha256=train.source.manifest['checkpoint_sha256'],
        train_manifest_sha256=[train.ego.fingerprint,train.source.fingerprint],development_manifest_sha256=[dev.ego.fingerprint,dev.source.fingerprint],
        seed=17,batch_size=4,max_epochs=6,patience=2,lr=1e-4,precision='float32',
        training_source='random 0--5-frame stale real observation, no future source',
        validation_source='two-frame stale real observation',checkpoint_selection='development AP25',
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    training=DataLoader(train,batch_size=4,shuffle=True,num_workers=4,collate_fn=collate,pin_memory=True)
    validation=DataLoader(dev,batch_size=4,shuffle=False,num_workers=4,collate_fn=collate,pin_memory=True)
    start=0;best=-1.;best_epoch=0;history=[]
    if (out/'latest.pt').exists():
        state=torch.load(out/'latest.pt',map_location='cpu');assert state['provenance']==provenance
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];history=state['history']
    for epoch in range(start,6):
        model.train();total=0.;count=0;started=time.monotonic()
        for b in training:
            if b is None:continue
            optimizer.zero_grad(set_to_none=True);outputs=forward(model,b)
            boxes=[];labels=[]
            for raw,l in zip(b['boxes'],b['labels']):
                # Velocity is not part of this cooperative detection target.
                full=np.concatenate((raw,np.zeros((len(raw),2),np.float32)),1)
                boxes.append(LiDARInstance3DBoxes(torch.from_numpy(full).cuda(),box_dim=9,origin=(.5,.5,.5)))
                labels.append(torch.as_tensor(l,device='cuda'))
            losses=model.head.loss(boxes,labels,outputs);loss=sum(losses.values())
            if not torch.isfinite(loss):raise ValueError('Nonfinite fusion loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),35.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite fusion gradients')
            optimizer.step();total+=loss.item()*len(boxes);count+=len(boxes)
        model.eval();frames=[]
        with torch.no_grad():
            for b in validation:
                if b is None:continue
                outputs=forward(model,b);predictions=model.decode(outputs)
                frames.extend(zip(predictions,b['boxes'],b['labels']))
        metrics=summarize(frames);value=metrics['AP25']
        row=dict(epoch=epoch+1,train_loss=total/max(count,1),development=metrics,seconds=time.monotonic()-started)
        history.append(row);print(json.dumps(row),flush=True)
        if value>best:
            best=value;best_epoch=epoch+1
            torch.save(dict(state_dict=model.state_dict(),provenance=provenance,epoch=best_epoch),out/'best.pt')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),provenance=provenance,
            epoch=epoch+1,best=best,best_epoch=best_epoch,history=history),out/'latest.pt')
        atomic_json(out/'history.json',history)
        if epoch+1-best_epoch>=2:break
    atomic_json(out/'complete.json',dict(provenance=provenance,best_epoch=best_epoch,development_AP25=best,
        checkpoint_sha256=sha(out/'best.pt'),completed_unix=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['aionet','v2xvit','select2col'],default='aionet');args=p.parse_args()
    # A secondary scheduler may borrow another free GPU. Recheck completion
    # under a method-level lock so two schedulers cannot train the same model.
    with (ROOT/'status'/('training_'+args.method+'.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        main(args)
