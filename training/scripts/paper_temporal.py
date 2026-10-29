"""Evaluate actual predictions under declared causal communication delays."""
import argparse,bisect,json
from pathlib import Path
import mmcv,numpy as np,torch
from experiment_io import ROOT,DetectorCache,records,gt_roi,source_to_ego,atomic_json,sha
from aionet.paper.experiments import get_experiment
from aionet.paper.temporal import lognormal_delays,delay_stratum,propagate_boxes
from aionet.structured_fusion import Detections,map_to_ego,fuse_structured
from aionet.evaluation_metrics import summarize
from aionet.r2_bev_components import TempoSense,warp_bev
from paper_data import PaperPairs
from paper_model import FullPaperFusion

def load_model(spec,path):
    model=FullPaperFusion(mmcv.Config.fromfile(str(ROOT/'configs/bevformer_vehicle.py')),spec['operator']).cuda().eval()
    saved=torch.load(path,map_location='cpu')
    if saved['provenance']['spec']!=spec:raise ValueError('Checkpoint specification differs from the requested temporal experiment')
    model.load_state_dict(saved['state_dict']);return model

def evaluate(spec,temporal_checkpoint,mode):
    data=PaperPairs(spec,'test');ego=data.ego;source=data.source
    model=load_model(spec,ROOT/'runs/paper'/spec['name']/'final.pt')
    fixed_spec=get_experiment('temporal_fixed_delay')
    fixed_path=ROOT/'runs/paper/temporal_fixed_delay/final.pt'
    if mode=='stochastic' and any(spec[k]!=fixed_spec[k] for k in ['vehicle','source','operator']):
        raise ValueError('The stochastic comparison requires the same registered pair and operator as temporal_fixed_delay')
    fixed=load_model(fixed_spec,fixed_path) if mode=='stochastic' else model
    temporal=TempoSense().cuda().eval();temporal.load_state_dict(torch.load(temporal_checkpoint,map_location='cpu')['state_dict'])
    strategies=['vehicle_only','infrastructure_only','observed','fixed_delay','queue','constant_velocity','temposense'] if mode=='stochastic' else (['vehicle_only','infrastructure_only'] if mode=='detector_sweep' else ['vehicle_only','infrastructure_only','observed','temposense'])
    sweeps=[None] if mode=='stochastic' else [0.,50.,100.,150.,200.,250.,300.,500.,1000.]
    delays={};arrival={}
    for sid,rows in source.by_scene.items():
        draw=lognormal_delays(len(rows),17+int(sid)).sum(1)
        for row,delay in zip(rows,draw):delays[row['token']]=float(delay)
    report={}
    for sweep in sweeps:
        groups={}
        for sid,rows in source.by_scene.items():
            arrival[sid]=sorted([(r['observation_us']+(delays[r['token']] if sweep is None else sweep)*1000,r) for r in rows],key=lambda x:x[0])
        for r in records('cooperative','test'):
            er=ego.by_token[r['vehicle_token']];ep=ego.detections(er).roi();gt,labels=gt_roi(r);t=r['timestamp']
            incoming=arrival[r['sequence_id']];end=bisect.bisect_right([x[0] for x in incoming],t)
            available=[x[1] for x in incoming[:end] if x[1]['observation_us']<=t]
            freshest=max(available,key=lambda x:x['observation_us']) if available else None
            latest=available[-1] if available else None
            stratum=delay_stratum(delays[r['infrastructure_token']]) if sweep is None else str(sweep)
            for strategy in strategies:
                pred=ep;sr=freshest if strategy in ('queue','temposense','constant_velocity') else latest
                if sr is None and strategy=='infrastructure_only':pred=Detections([],[],[])
                if mode=='detector_sweep' and strategy=='vehicle_only':
                    history=ego.by_scene[r['sequence_id']];idx=bisect.bisect_right([q['observation_us'] for q in history],t-sweep*1000)-1
                    if idx>=0:
                        old=history[idx];matrix=np.linalg.inv(r['lidar2global'])@np.asarray(old['lidar2global']);pred=map_to_ego(ego.detections(old),matrix).roi()
                    else:pred=Detections([],[],[])
                elif sr is not None and strategy!='vehicle_only':
                    matrix=source_to_ego(sr,r);age=(t-sr['observation_us'])/1e6
                    if strategy in ('infrastructure_only','constant_velocity'):
                        remote=map_to_ego(source.detections(sr),matrix).roi()
                        if strategy=='constant_velocity':
                            if remote.velocity is None:raise ValueError('Constant-velocity baseline requires detector velocity outputs')
                            remote=Detections(propagate_boxes(remote.boxes,remote.velocity,age),remote.scores,remote.labels,remote.velocity)
                            pred=fuse_structured(ep,remote,iou_threshold=.25,source_temperature=1.,device='cuda')
                        else:pred=remote
                    else:
                        local={k:v[None].cuda() for k,v in data.packet(ego,er).items()};remote={k:v[None].cuda() for k,v in data.packet(source,sr).items()};forecast=False
                        if strategy=='temposense' and 'bev' in remote:
                            history=sorted([q for q in available if q['observation_us']<=sr['observation_us']],key=lambda q:q['observation_us'])[-3:]
                            times=[q['observation_us'] for q in history]
                            if len(history)==3 and 0<age<=2 and np.diff(times).min()>0 and np.diff(times).max()<=150000:
                                inverse=np.linalg.inv(np.asarray(sr['lidar2global']))
                                transforms=np.stack([inverse@np.asarray(q['lidar2global']) for q in history]).astype(np.float32)
                                maps=torch.from_numpy(np.stack([source.feature(q) for q in history])).cuda()
                                with torch.no_grad():remote['bev']=temporal.predict_at(warp_bev(maps,torch.from_numpy(transforms).cuda())[0][None],times,t)
                                forecast=True
                        decoder=fixed if strategy=='fixed_delay' else model
                        with torch.no_grad():pred=decoder.decode(decoder(local,remote,torch.from_numpy(matrix)[None].cuda(),torch.tensor([age],device='cuda'),torch.tensor([float(forecast)],device='cuda')))[0]
                groups.setdefault((stratum,strategy),[]).append((pred,gt,labels))
        for (condition,strategy),frames in groups.items():report.setdefault(condition,{})[strategy]=dict(summarize(frames),frames=len(frames))
    destination=ROOT/'results/paper_temporal';destination.mkdir(parents=True,exist_ok=True)
    atomic_json(destination/(mode+'.json'),dict(mode=mode,experiment=spec,seed=17,temporal_checkpoint_sha256=sha(temporal_checkpoint),results=report,
        protocol='single-frame causal communication replay; observed uses last arrival, queue uses freshest observation among arrivals; shared frame strata; prediction replaces only BEV'))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--mode',choices=['stochastic','sweep','detector_sweep'],default='stochastic');p.add_argument('--temposense',type=Path,required=True);a=p.parse_args()
    evaluate(get_experiment(a.experiment),a.temposense,a.mode)
