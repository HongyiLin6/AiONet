"""Fit the paper's 1,761-parameter selector to verified branch caches.

NPZ arrays: x (N,20), delta (N,), scenario (N, string). Split JSON keys:
perception_train, selector_fit, development, calibration, test. The NPZ contains
fit/dev/cal rows only. No test labels are read by fitting or calibration.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from reliability_core import DescriptorNormalizer,FEATURE_NAMES,assert_disjoint_scenarios,calibration_margin

class RelativeLossSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(20,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1),nn.Tanh())
    def forward(self,x): return self.net(x).squeeze(-1)

def scenario_weights(ids):
    unique,count=np.unique(ids,return_counts=True)
    lookup=dict(zip(unique,count))
    return np.asarray([len(ids)/(len(unique)*lookup[s]) for s in ids],np.float32)

def predict(model,x,device,batch=4096):
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):
            out.append(model(torch.from_numpy(x[start:start+batch]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)

def fit(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    parts=json.loads(Path(args.split).read_text())
    assert_disjoint_scenarios(parts)
    required={'perception_train','selector_fit','development','calibration','test'}
    if set(parts)!=required: raise ValueError('Split file must enumerate exactly the five named partitions')
    meta=json.loads(Path(args.cache_manifest).read_text())
    for key in ['perception_checkpoint_sha256','score_threshold','causal_cache_audit','trace_seeds','deadline_budgets_ms','delay_medians_ms','cache_sha256']:
        if key not in meta: raise ValueError('Missing cache provenance: '+key)
    if meta['causal_cache_audit']!='passed': raise ValueError('Causal cache audit has not passed')
    digest=hashlib.sha256(Path(args.cache).read_bytes()).hexdigest()
    if digest!=meta['cache_sha256']: raise ValueError('Cache does not match manifest checksum')
    with np.load(args.cache,allow_pickle=False) as data:
        x=np.asarray(data['x'],float); y=np.asarray(data['delta'],np.float32); ids=data['scenario'].astype(str)
    if x.shape!=(len(y),20) or ids.shape!=y.shape: raise ValueError('Invalid cache shapes')
    if not np.isfinite(y).all() or (np.abs(y)>1+1e-6).any(): raise ValueError('Loss difference outside [-1,1]')
    allowed=set(parts['selector_fit']+parts['development']+parts['calibration'])
    if not set(ids)<=allowed: raise ValueError('Cache contains perception-only, test or unknown scenarios')
    masks={key:np.isin(ids,parts[key]) for key in ['selector_fit','development','calibration']}
    if not masks['selector_fit'].any() or not masks['development'].any(): raise ValueError('Empty fitting/development partition')
    if args.without_age_horizon:
        x=x.copy(); x[:,[0,1]]=0
    normalizer=DescriptorNormalizer().fit(x[masks['selector_fit']]); x=normalizer.transform(x)
    xf,yf,sf=x[masks['selector_fit']],y[masks['selector_fit']],ids[masks['selector_fit']]
    xd,yd,sd=x[masks['development']],y[masks['development']],ids[masks['development']]
    device=torch.device(args.device); model=RelativeLossSelector().to(device)
    assert sum(p.numel() for p in model.parameters())==1761
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loss_fn=nn.HuberLoss(reduction='none',delta=.1)
    weights=scenario_weights(sf); dev_weights=scenario_weights(sd)
    rng=np.random.default_rng(args.seed); best=math.inf; best_state=None; stale=0; history=[]
    for epoch in range(100):
        model.train(); order=rng.permutation(len(xf)); total=0.
        for start in range(0,len(order),256):
            ix=order[start:start+256]
            bx=torch.from_numpy(xf[ix]).to(device); by=torch.from_numpy(yf[ix]).to(device)
            bw=torch.from_numpy(weights[ix]).to(device)
            optimizer.zero_grad(); loss=(loss_fn(model(bx),by)*bw).mean(); loss.backward(); optimizer.step()
            total+=loss.item()*len(ix)
        pred=predict(model,xd,device)
        with torch.no_grad():
            dev=float((loss_fn(torch.from_numpy(pred),torch.from_numpy(yd))*torch.from_numpy(dev_weights)).mean())
        record={'epoch':epoch+1,'train_huber':total/len(xf),'development_huber':dev}
        history.append(record); print(json.dumps(record),flush=True)
        if dev<best:
            best=dev; best_state=copy.deepcopy(model.cpu().state_dict()); model.to(device); stale=0
        else: stale+=1
        if stale>=10: break
    model.load_state_dict(best_state); cm=masks['calibration']; cp=predict(model,x[cm],device)
    residuals=y[cm]-cp; ci=ids[cm]
    margin=calibration_margin({sid:residuals[ci==sid] for sid in parts['calibration']},alpha=.1)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    torch.save({'state_dict':best_state,'normalizer':normalizer.state_dict(),'features':FEATURE_NAMES,
                'without_age_horizon':args.without_age_horizon},out/'selector.pt')
    margin['q']=margin['q'] if math.isfinite(margin['q']) else 'infinity'
    margin['scenario_maxima']={k:v if math.isfinite(v) else '-infinity' for k,v in margin['scenario_maxima'].items()}
    (out/'calibration.json').write_text(json.dumps(margin,indent=2,allow_nan=False))
    (out/'fit_history.json').write_text(json.dumps(history,indent=2))
    (out/'provenance.json').write_text(json.dumps({'cache':meta,'split':parts,'seed':args.seed,
        'parameters':1761,'best_development_huber':best,'device':str(device)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['cache','cache-manifest','split','output']: parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--without-age-horizon',action='store_true')
    fit(parser.parse_args())
