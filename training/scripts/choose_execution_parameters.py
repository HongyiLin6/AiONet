"""Fix runtime estimates and freshness penalties before calibration is read."""
import argparse
import itertools
import json
import math
import time
import numpy as np
import torch
from experiment_io import ROOT,atomic_json,sha
from selector_train import RelativeLossSelector

def rows(family):
    sources=['bevformer_infrastructure'] if family=='F' else ['bevformer_infrastructure','petr_infrastructure']
    result=[];files=[]
    for source in sources:
        path=ROOT/'cache/branches'/family/source/'development/frames.jsonl';files.append(path)
        with path.open() as f:result.extend(json.loads(line) for line in f)
    return result,files

def estimated_cost(candidate,settings):
    kind=candidate['action'];steps=candidate['prediction_steps'];groups=settings['branch_cost_p99_seconds'].get(kind,{})
    if not groups or steps>max(map(int,groups)):return math.inf
    keys=sorted(int(k) for k in groups);chosen=next((k for k in keys if k>=steps),keys[-1])
    return groups[str(chosen)]+settings['descriptor_cost_p99_seconds']+settings['selector_cost_p99_seconds']

def feasible(row,candidate,settings):
    cost=estimated_cost(candidate,settings)
    return math.isfinite(cost) and row['decision_us']+math.ceil(cost*1e6)<=row['deadline_us']

def freshness(row,settings,parameters):
    available=[]
    for action,b in row['candidates'].items():
        gap=0. if action=='P' else (row['frame_us']-b['observation_us'])/1e6
        if gap>parameters['max_mismatch'] or not feasible(row,b,settings):continue
        value=gap+parameters['prediction_penalty']*(action=='P')+parameters['staleness_penalty']*gap
        available.append((value,estimated_cost(b,settings),action!='O',action))
    return min(available)[-1] if available else '0'

def committed(row,action,settings,selector_seconds=None):
    ready=row['decision_us'];deadline=row['deadline_us']
    if action!='0':
        b=row['candidates'][action]
        extra=row['descriptor_seconds']+b['seconds']+(settings['selector_cost_p99_seconds'] if selector_seconds is None else selector_seconds)
        if ready+math.ceil(extra*1e6)<=deadline:return action
    return '0' if ready<=deadline else None

def main(args):
    path=ROOT/'runs/choices'/(args.family+'_execution.json')
    if path.exists():return
    data,files=rows(args.family);costs={};unique={}
    for r in data:
        for b in r['candidates'].values():unique[b['key']]=b
    for b in unique.values():costs.setdefault(b['action'],{}).setdefault(str(b['prediction_steps']),[]).append(b['seconds'])
    network=RelativeLossSelector().cuda().eval();x=torch.zeros(1,20,device='cuda');durations=[]
    with torch.no_grad():
        for index in range(110):
            torch.cuda.synchronize();begin=time.perf_counter();network(x);torch.cuda.synchronize()
            if index>=10:durations.append(time.perf_counter()-begin)
    settings=dict(family=args.family,branch_cost_p99_seconds={a:{k:float(np.quantile(v,.99)) for k,v in groups.items()} for a,groups in costs.items()},
        descriptor_cost_p99_seconds=float(np.quantile([r['descriptor_seconds'] for r in data],.99)),
        selector_cost_p99_seconds=float(np.quantile(durations,.99)),
        estimator_partition='development',development_inputs={str(p):sha(p) for p in files},
        timing_scope='single-frame replay from in-RAM payloads; actual branch elapsed time; no sustained receiver queue guarantee',
        freshness_criterion='scenario-balanced delivered bounded loss on development only')
    candidates=[]
    grid=itertools.product([0.,.05,.1,.2],[0.,.5,1.],[.1,.25,.5,1.,2.]) if args.family=='F' else [(0.,0.,2.)]
    for z,s,limit in grid:
        parameters=dict(prediction_penalty=z,staleness_penalty=s,max_mismatch=limit);scenarios={}
        for r in data:
            action=committed(r,freshness(r,settings,parameters),settings)
            loss=r['empty_loss'] if action is None else r['ego_loss'] if action=='0' else r['candidates'][action]['loss']
            scenarios.setdefault(r['scenario'],[]).append(loss)
        candidates.append(dict(parameters,development_loss=float(np.mean([np.mean(v) for v in scenarios.values()]))))
    best=min(candidates,key=lambda r:(r['development_loss'],r['prediction_penalty'],r['staleness_penalty'],-r['max_mismatch']))
    settings.update(freshness=best,freshness_grid=candidates);atomic_json(path,settings)
    print(json.dumps(settings),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['F','S'],required=True);main(p.parse_args())
