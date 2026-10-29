"""Actual batch-one receiver wall times on frozen, causal real input snapshots.

Includes image decoding/preprocessing, GPU inference, descriptors, selection,
any chosen cooperative branch, and serialization. It is a per-frame benchmark;
it does not claim sustained input-queue throughput. A cached ego version is
emitted first and replaced only by a cooperative version completed in time.
"""
import argparse
import io
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from experiment_io import ROOT,DetectorCache,records,sha,atomic_json
from structured_fusion import Detections
from feature_bank import Observation,HistorySnapshot
from build_branch_cache import BranchRunner
from descriptors import make_descriptor
from choose_execution_parameters import feasible,estimated_cost,freshness
from evaluate_policy_cache import load_selector
from aionet.selector import score as portable_score
from aionet.replay import choose as portable_choose

def serialize(detection):
    data=io.BytesIO();np.savez(data,boxes=detection.boxes,scores=detection.scores,labels=detection.labels);return data.getvalue()

def main(args):
    import mmcv
    from mmcv.parallel import MMDataParallel,collate
    from mmcv.runner import load_checkpoint,wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    from freeze_evaluation_choices import verify_frozen
    verify_frozen()
    target=ROOT/'results/runtime'/args.policy;target.mkdir(parents=True,exist_ok=True)
    if (target/'summary.json').exists():return
    branch_folder=ROOT/'cache/branches/F/bevformer_infrastructure/test'
    cached=[json.loads(s) for s in (branch_folder/'frames.jsonl').open()]
    cached=[r for r in cached if r['trace']=='network_seed17_median234p9.npz' and r['budget_ms']==100]
    selected=cached[:138];by_frame={r['frame_id']:r for r in selected};cooperative={r['frame_id']:r for r in records('cooperative','test')}
    ego=DetectorCache('bevformer_vehicle','test');source=DetectorCache('bevformer_infrastructure','test')
    cfg=mmcv.Config.fromfile(str(ROOT/'configs/bevformer_vehicle.py'));ds=build_dataset(cfg.data.test)
    base=build_model(cfg.model);base.init_weights();load_checkpoint(base,ego.manifest['checkpoint'],map_location='cpu',strict=True)
    wrap_fp16_model(base);base=MMDataParallel(base.cuda().eval(),device_ids=[0])
    runner=None;selector=None;portable=None;calibration=None
    if args.policy!='A0':runner=BranchRunner('F',ego,source,'test',target,cache_results=False)
    if args.policy=='A5':selector=load_selector('F')
    settings=json.loads((ROOT/'runs/choices/F_execution.json').read_text())
    if args.policy in ('R-BH','R-MLP'):
        import joblib
        if args.model_dir is None:raise ValueError('Pass the fitted descriptor selector with --model-dir')
        portable=joblib.load(args.model_dir/'model.joblib')
        if portable['dimensions']!=20:raise ValueError('This receiver benchmark uses the 20-descriptor BEV policy')
        calibration=json.loads((args.model_dir/'calibration.json').read_text());settings=calibration['settings']
    score_threshold=json.loads((ROOT/'runs/choices/F_ego_threshold.json').read_text())['threshold']
    observations={r['frame_id']:tuple(Observation(**p) for p in r['history_records']) for r in selected}
    captured={}
    def hook(_module,_args,value):captured['bev']=value['bev_embed'].detach()
    handle=base.module.pts_bbox_head.register_forward_hook(hook)
    results=[];seen=0;torch.cuda.reset_peak_memory_stats()
    for index,info in enumerate(ds.data_infos):
        frame=info['frame_id'];row=by_frame.get(frame)
        if row is not None and runner is not None:
            # These source tensors represent payloads already delivered into
            # the Feature Bank; preloading is outside the receiver frame clock.
            for p in observations[frame]:runner.feature(source.by_token[p.payload_ref])
        torch.cuda.synchronize();start=time.perf_counter();sample=collate([ds[index]],samples_per_gpu=1)
        preprocessed=time.perf_counter()
        with torch.no_grad():raw=base(return_loss=False,rescale=True,**sample)[0]['pts_bbox']
        torch.cuda.synchronize();inferred=time.perf_counter()
        if row is None:
            if seen>=len(selected):break
            continue
        b=raw['boxes_3d'];ego_pred=Detections(torch.cat((b.gravity_center,b.tensor[:,3:7]),1).cpu().numpy(),
            raw['scores_3d'].cpu().numpy(),raw['labels_3d'].cpu().numpy(),b.tensor[:,7:9].cpu().numpy()).roi()
        # Make the runtime Feature Bank contain the just-computed ego feature,
        # rather than reusing a counterfactual output from disk.
        if runner is not None:
            bev=captured['bev'];bev=bev.permute(1,0,2) if bev.shape[0]==22500 else bev
            runner.feature_lru[(True,info['token'])]=bev[0].reshape(150,150,256).permute(2,0,1).float().cpu().numpy()
        ego_payload=serialize(ego_pred);ego_emission=time.perf_counter()-start
        deadline=start+.1;delivered='0' if time.perf_counter()<=deadline else None;chosen='0'
        descriptor_seconds=0.;selection_seconds=0.;branch_seconds=0.;output_bytes=len(ego_payload)
        if runner is not None and row['candidates']:
            # Reproduce the same bank polling instant as the causal replay. Any
            # wait until that snapshot becomes available counts in wall time.
            available=start+(row['decision_us']-row['frame_us'])/1e6
            wait=available-time.perf_counter()
            if wait>0:time.sleep(wait)
            decision_us=row['frame_us']+int((time.perf_counter()-start)*1e6)
            snapshot=HistorySnapshot(row['scenario'],'bevformer_infrastructure','F',row['frame_us'],decision_us,observations[frame])
            temporary=dict(row,decision_us=decision_us);xs={};begin=time.perf_counter()
            for action,metadata in row['candidates'].items():
                # Source summaries and history length are bank metadata. Motion
                # and confidence summaries use the actual newly inferred ego.
                old=np.array([np.nan if v is None else v for v in metadata['x']],float)
                x=make_descriptor(snapshot,action,ego_pred,score_threshold,row['deadline_us'],
                    temporal_change=None if old[12] else old[10],fov_overlap=None if old[14] else old[9])
                x[11]=row['history_length'];xs[action]=x
            descriptor_seconds=time.perf_counter()-begin;begin=time.perf_counter()
            if args.policy=='A3':chosen=freshness(temporary,settings,settings['freshness'])
            elif portable is not None:
                actions=list(xs);values=portable_score(portable,np.stack([xs[a] for a in actions]))
                temporary['candidates']={a:dict(row['candidates'][a],x=xs[a]) for a in actions}
                chosen=portable_choose(temporary,'selector',settings,{(0,a):float(v) for a,v in zip(actions,values)},0,calibration['chosen']['threshold'])
            else:
                network,norm,q,_=selector;actions=list(xs)
                with torch.no_grad():values=network(torch.from_numpy(norm.transform(np.stack([xs[a] for a in actions]))).cuda()).cpu().numpy()
                available_actions=[(float(v)+q,estimated_cost(row['candidates'][a],settings),a!='O',a) for a,v in zip(actions,values) if feasible(temporary,row['candidates'][a],settings)]
                winner=min(available_actions) if available_actions else None
                chosen=winner[-1] if winner and winner[0]<0 else '0'
            torch.cuda.synchronize();selection_seconds=time.perf_counter()-begin
            if chosen!='0':
                begin=time.perf_counter()
                prediction,_=runner.branch(cooperative[frame],ego.by_token[info['token']],snapshot,chosen)
                emitted=serialize(prediction);torch.cuda.synchronize();branch_seconds=time.perf_counter()-begin
                if time.perf_counter()<=deadline:delivered=chosen;output_bytes=len(emitted)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;seen+=1
        results.append(dict(frame_id=frame,scenario=row['scenario'],warmup=seen<=10,wall_seconds=elapsed,
            preprocessing_seconds=preprocessed-start,ego_model_seconds=inferred-preprocessed,ego_emission_seconds=ego_emission,
            descriptor_seconds=descriptor_seconds,selection_seconds=selection_seconds,branch_seconds=branch_seconds,
            attempted=chosen,committed=delivered,output_bytes=output_bytes,deadline_miss=delivered is None))
        if seen>=len(selected):break
    handle.remove();timed=[r for r in results if not r['warmup']];assert len(timed)>=100
    payload=json.loads((branch_folder/'manifest.json').read_text())['payload_bytes']
    summary=dict(policy=args.policy,timed_frames=len(timed),warmup_frames=10,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        device=torch.cuda.get_device_name(0),torch_version=torch.__version__,batch_size=1,
        p50_ms=float(np.median([r['wall_seconds'] for r in timed])*1000),p95_ms=float(np.quantile([r['wall_seconds'] for r in timed],.95)*1000),
        output_deadline_miss=float(np.mean([r['deadline_miss'] for r in timed])),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        payload_MiB_per_source_packet=0. if args.policy=='A0' else payload['median']/2**20,
        descriptor_mean_ms=float(np.mean([r['descriptor_seconds'] for r in timed])*1000),selector_mean_ms=float(np.mean([r['selection_seconds'] for r in timed])*1000),
        attempted={a:sum(r['attempted']==a for r in timed) for a in ['0','O','P']},committed={str(a):sum(r['committed']==a for r in timed) for a in ['0','O','P',None]},
        timing_scope='actual per-frame wall time including image decode/preprocessing, model, bank polling wait, descriptors, selection, selected branch, serialization; GPU synchronized',
        queue_scope='isolated frame latency with fixed causal bank-poll snapshots; does not establish sustained receiver-queue throughput',
        emission='ego version emitted first; cooperative update committed only if finished by the 100-ms deadline',
        selector_model_sha256=sha(args.model_dir/'model.joblib') if args.model_dir else None,
        frozen_protocol_sha256=sha(ROOT/'manifests/evaluation_choices_frozen.json'),rows=results)
    atomic_json(target/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=['A0','A3','A5','R-BH','R-MLP'],required=True);p.add_argument('--model-dir',type=Path)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main(p.parse_args())
