"""Evaluate fixed policies on untouched test caches and paired scenario draws."""
import argparse
import collections
import json
import math
from pathlib import Path
import numpy as np
import torch
from experiment_io import ROOT,sha,atomic_json,records,gt_roi
from structured_fusion import Detections
from evaluation_metrics import overlap_matrix,match
from selector_train import RelativeLossSelector,predict
from reliability_core import DescriptorNormalizer
from choose_execution_parameters import feasible,estimated_cost,freshness,committed

def load_selector(name):
    folder=ROOT/'runs/selectors'/name;state=torch.load(folder/'selector.pt',map_location='cpu')
    model=RelativeLossSelector().cuda().eval();model.load_state_dict(state['state_dict'])
    norm=DescriptorNormalizer()
    for k,v in state['normalizer'].items():setattr(norm,k,np.array(v))
    q=json.loads((folder/'calibration.json').read_text())['q'];q=math.inf if q=='infinity' else q
    return model,norm,q,state['without_age_horizon']

def score_all(data,state):
    model,norm,q,ablate=state;xs=[];positions=[]
    for index,r in enumerate(data):
        for action,b in r['candidates'].items():
            x=np.array([np.nan if v is None else v for v in b['x']],float)
            if ablate:x[[0,1]]=0
            xs.append(x);positions.append((index,action))
    scores=predict(model,norm.transform(np.stack(xs)),torch.device('cuda')) if xs else []
    return {key:float(value) for key,value in zip(positions,scores)},q

def select(row,policy,settings,scores,q,index):
    if policy=='A0':return '0'
    if policy in ('A1','A2'):
        action='O' if policy=='A1' else 'P'
        return action if action in row['candidates'] and feasible(row,row['candidates'][action],settings) else '0'
    if policy=='A3':return freshness(row,settings,settings['freshness'])
    candidates=[(scores[(index,a)]+q,estimated_cost(b,settings),a!='O',a) for a,b in row['candidates'].items() if feasible(row,b,settings)]
    winner=min(candidates) if candidates else None
    return winner[-1] if winner and winner[0]<0 else '0'

class CompiledMetrics:
    def __init__(self,score_threshold):self.threshold=score_threshold;self.cache={}
    def frame(self,file,record):
        key=(str(file),record['frame_id'])
        if key in self.cache:return self.cache[key]
        if file is None:pred=Detections(np.empty((0,7)),[],[])
        else:
            with np.load(file,allow_pickle=False) as d:pred=Detections(d['boxes'],d['scores'],d['labels']).roi()
        gt,labels=gt_roi(record);overlaps=overlap_matrix(pred.boxes,gt)
        result=dict(counts=np.bincount(labels,minlength=7),scores=pred.scores,labels=pred.labels)
        for th in [.25,.5]:result[th]=match(pred,gt,labels,iou=th,overlaps=overlaps)['tp']
        m=match(pred,gt,labels,threshold=self.threshold,overlaps=overlaps)
        result.update(tp=m['count_tp'],fp=m['count_fp'],fn=m['count_fn'],loss=m['loss']);self.cache[key]=result;return result

def aggregate(frames,outcomes,scenario_ids,bootstrap):
    scenes=sorted(set(scenario_ids));scene_index=np.array([scenes.index(s) for s in scenario_ids]);n=len(scenes)
    # All methods use identical scenario multiplicities, preserving their pairing.
    rng=np.random.default_rng(20261027);draw=rng.integers(0,n,(bootstrap,n))
    weights=np.stack([np.bincount(r,minlength=n) for r in draw]);weights=np.vstack((np.ones(n,int),weights))
    counts=np.zeros((n,7));totals=np.zeros((n,10));events={}
    for frame,outcome,sid in zip(frames,outcomes,scene_index):
        counts[sid]+=frame['counts']
        totals[sid]+=[1,frame['tp'],frame['fp'],frame['fn'],outcome['accepted'],outcome['harm'],outcome['miss'],outcome['positive_increment'],outcome['late_branch'],frame['loss']]
    aggregate_counts=weights@counts;values={}
    for th in [.25,.5]:
        per_class=[]
        for label in range(7):
            ss=[];tt=[];ii=[]
            for frame,sid in zip(frames,scene_index):
                mask=frame['labels']==label;ss.extend(frame['scores'][mask]);tt.extend(frame[th][mask]);ii.extend([sid]*int(mask.sum()))
            scores=np.asarray(ss);tp=np.asarray(tt,float);ids=np.asarray(ii,int)
            order=np.argsort(-scores,kind='stable');tp=tp[order];ids=ids[order]
            ap=[]
            for w,total in zip(weights,aggregate_counts[:,label]):
                if total==0:ap.append(np.nan);continue
                mass=w[ids];keep=mass>0;t=np.cumsum(tp[keep]*mass[keep]);f=np.cumsum((1-tp[keep])*mass[keep])
                rr=np.r_[0.,t/total,1.];pp=np.r_[0.,t/np.maximum(1,t+f),0.];pp=np.maximum.accumulate(pp[::-1])[::-1]
                ix=np.flatnonzero(rr[1:]!=rr[:-1]);ap.append(float(((rr[ix+1]-rr[ix])*pp[ix+1]).sum()))
            per_class.append(ap)
        values['AP'+str(int(th*100))]=np.nanmean(np.array(per_class),axis=0)
    v=weights@totals;den=np.maximum(v[:,0],1);accepted=v[:,4]
    values.update(C_coop=accepted/den,H_all=v[:,5]/den,H_acc=np.divide(v[:,5],accepted,out=np.full(len(accepted),np.nan),where=accepted>0),
        output_deadline_miss=v[:,6]/den,mean_positive_increment=v[:,7]/den,late_branch_rate=v[:,8]/den,
        mean_loss=v[:,9]/den,recall25=v[:,1]/np.maximum(v[:,1]+v[:,3],1),fp_per_frame=v[:,2]/den)
    output={}
    for key,value in values.items():
        samples=value[1:];finite=samples[np.isfinite(samples)]
        output[key]=dict(value=float(value[0]) if np.isfinite(value[0]) else None,
            ci95=[float(x) for x in np.quantile(finite,[.025,.975])] if len(finite) else None,
            defined_bootstrap_samples=int(len(finite)))
    return dict(metrics=output,scenario_count=n,frame_trace_instances=len(frames),bootstrap_samples=bootstrap,
        resampling='paired whole-scenario resampling; all prespecified traces retained per scenario; AP pooled over trace instances')

def main(args):
    assert (ROOT/'manifests/evaluation_choices_frozen.json').exists()
    from freeze_evaluation_choices import verify_frozen
    before=verify_frozen()
    source='bevformer_infrastructure' if args.family=='F' else 'detr3d_infrastructure'
    folder=ROOT/'cache/branches'/args.family/source/'test';meta=json.loads((folder/'manifest.json').read_text())
    data=[json.loads(line) for line in (folder/'frames.jsonl').open()]
    settings=json.loads((ROOT/'runs/choices'/(args.family+'_execution.json')).read_text())
    scores,q=score_all(data,load_selector(args.family));ablation,aq=score_all(data,load_selector('F_no_age_horizon')) if args.family=='F' else ({},0)
    gt={r['frame_id']:r for r in records('cooperative','test')};compiled=CompiledMetrics(meta['score_threshold'])
    groups=collections.defaultdict(list)
    for i,r in enumerate(data):groups[(r['trace'].split('_median')[1].split('.npz')[0],r['budget_ms'])].append(i)
    policies=['A0','A1','A2','A3','A4','A5','A6'] if args.family=='F' else ['A0','A1','A4','A5']
    output=ROOT/'results'/args.family;output.mkdir(parents=True,exist_ok=True);results={}
    for group,indices in sorted(groups.items()):
        for policy in policies:
            destination=output/('%s_budget%d_%s.json'%(group[0],group[1],policy))
            if destination.exists():results[destination.stem]=json.loads(destination.read_text());continue
            frames=[];outcomes=[];scenarios=[];decisions=[]
            for index in indices:
                row=data[index];score=ablation if policy=='A6' else scores;margin=aq if policy=='A6' else q if policy=='A5' else 0.
                action=select(row,policy,settings,score,margin,index);done=committed(row,action,settings,selector_seconds=0. if policy in ('A0','A1','A2','A3') else None)
                file=None if done is None else Path(row['ego_file']) if done=='0' else folder/row['candidates'][done]['file']
                frame=compiled.frame(file,gt[row['frame_id']]);accepted=done in ('O','P');delta=row['candidates'][done]['delta'] if accepted else 0.
                frames.append(frame);scenarios.append(row['scenario']);outcomes.append(dict(accepted=accepted,harm=accepted and delta>1e-12,
                    miss=done is None,positive_increment=max(0.,delta),late_branch=action in ('O','P') and done!=action))
                decisions.append(dict(scenario=row['scenario'],frame_id=row['frame_id'],trace=row['trace'],attempted=action,committed=done,delta=delta))
            summary=aggregate(frames,outcomes,scenarios,args.bootstrap)
            summary.update(family=args.family,source=source,policy=policy,delay_median_ms=float(group[0].replace('p','.')),budget_ms=group[1],
                score_threshold=meta['score_threshold'],cache_manifest_sha256=sha(folder/'manifest.json'),calibration_margin=margin if math.isfinite(margin) else 'infinity',
                protocol_limit=settings['timing_scope'],attempted=dict(collections.Counter(d['attempted'] for d in decisions)),committed=dict(collections.Counter(str(d['committed']) for d in decisions)))
            atomic_json(destination,summary);atomic_json(destination.with_suffix('.decisions.json'),decisions);results[destination.stem]=summary
            print(destination.stem,{k:v['value'] for k,v in summary['metrics'].items()},flush=True)
    atomic_json(output/'summary.json',results)
    if args.family=='F':
        from risk_coverage import evaluate
        evaluate()
    after=verify_frozen();assert before==after
    atomic_json(output/'frozen_receiver_audit.json',dict(family=args.family,source=source,
        before_and_after_match=True,verification=after,
        scope='Stored detector/receiver parameters, normalization, calibration, choices and critical code match the pre-test snapshot. Inference uses eval/no_grad; no optimization is performed.'))
    atomic_json(ROOT/'status'/('test_'+args.family+'_complete.json'),dict(family=args.family,result=str(output/'summary.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['F','S'],required=True);p.add_argument('--bootstrap',type=int,default=200);main(p.parse_args())
