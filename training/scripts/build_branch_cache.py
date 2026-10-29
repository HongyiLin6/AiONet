"""Replay causal arrivals and cache immutable paired branch outputs.

This is counterfactual offline labelling, not parallel online execution. Measured
costs cover one branch at a time. Labels are computed only after inference.
"""
import argparse
import collections
import fcntl
import hashlib
import io
import json
import math
import os
import time
from dataclasses import asdict
import numpy as np
import torch
from experiment_io import ROOT,DetectorCache,records,gt_roi,source_to_ego,sha,atomic_json
from feature_bank import Observation,FeatureBank,serial_source_ready
from descriptors import make_descriptor
from structured_fusion import Detections,map_to_ego,fuse_structured
from evaluation_metrics import choose_score_threshold,overlap_matrix,match

def sanitized(x):
    return [None if not np.isfinite(v) else float(v) for v in x]

class BranchRunner:
    def __init__(self,family,ego,source,partition,output,cache_results=True):
        self.family=family;self.ego=ego;self.source=source;self.partition=partition;self.output=output
        self.memo={};self.feature_lru=collections.OrderedDict()
        self.cache_results=cache_results
        self.fusion=None;self.temporal=None
        if family=='F':
            import mmcv
            from r2_fusion_model import RegisteredFusion
            from r2_bev_components import TempoSense
            for folder in ['registered_fusion','temposense']:assert (ROOT/'runs'/folder/'complete.json').is_file()
            self.fusion=RegisteredFusion(mmcv.Config.fromfile(str(ROOT/'configs/bevformer_vehicle.py')))
            self.fusion.load_state_dict(torch.load(ROOT/'runs/registered_fusion/best.pt',map_location='cpu')['state_dict'],strict=True)
            self.fusion.cuda().eval();self.temporal=TempoSense()
            self.temporal.load_state_dict(torch.load(ROOT/'runs/temposense/best.pt',map_location='cpu')['state_dict'],strict=True)
            self.temporal.cuda().eval()
        else:
            choice=json.loads((ROOT/'runs/choices/S_structured_parameters.json').read_text())
            self.nms=choice['nms_iou'];self.temperature=choice['temperature']
    def feature(self,row,ego=False):
        cache=self.ego if ego else self.source;key=(ego,row['token'])
        if key not in self.feature_lru:
            self.feature_lru[key]=cache.feature(row)
            if len(self.feature_lru)>48:self.feature_lru.popitem(last=False)
        self.feature_lru.move_to_end(key)
        return self.feature_lru[key]
    def temporal_change(self,snapshot):
        if self.family!='F' or len(snapshot.real)<2:return None
        a,b=[self.source.by_token[x.payload_ref] for x in snapshot.real[-2:]]
        if not np.allclose(a['lidar2global'],b['lidar2global'],atol=1e-4):return None
        # Registered fixed roadside frame: channel summaries of real features.
        for r in [a,b]:
            if '_summary' not in r:r['_summary']=self.feature(r).mean((1,2))
        return float(np.abs(a['_summary']-b['_summary']).mean())
    def branch(self,record,ego_row,snapshot,action):
        selected=snapshot.prediction_history() if action=='P' else (snapshot.latest,)
        if not selected:raise ValueError('Unsupported prediction history')
        key='|'.join([record['token'],action]+[r.packet_id for r in selected])
        digest=hashlib.sha256(key.encode()).hexdigest()[:24]
        if self.cache_results and digest in self.memo:return self.memo[digest]
        cached=self.output/'branches'/(digest+'.npz');metadata=cached.with_suffix('.json')
        if self.cache_results and cached.exists() and metadata.exists():
            with np.load(cached,allow_pickle=False) as d:pred=Detections(d['boxes'],d['scores'],d['labels'])
            result=(pred,json.loads(metadata.read_text()));self.memo[digest]=result;return result
        source_row=self.source.by_token[selected[-1].payload_ref]
        ego_pred=self.ego.detections(ego_row).roi();transform=source_to_ego(source_row,record)
        # Simulated Feature Bank payloads reside in receiver RAM. Disk loading
        # is outside execution timing; tensor transfers and output serialization
        # are inside the measured branch, with GPU synchronization.
        if self.family=='F':
            ego_feature=self.feature(ego_row,True)
            if action=='P':
                rows=[self.source.by_token[r.payload_ref] for r in selected]
                history=np.stack([self.feature(r) for r in rows]);latest_inv=np.linalg.inv(np.array(source_row['lidar2global']))
                transforms=np.stack([latest_inv@np.array(r['lidar2global']) for r in rows]).astype(np.float32)
            else:source_feature=self.feature(source_row)
        else:source_pred=self.source.detections(source_row)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():
            if self.family=='F':
                from r2_bev_components import warp_bev
                ego_tensor=torch.from_numpy(ego_feature)[None].cuda()
                if action=='P':
                    stack=torch.from_numpy(history).cuda();trans=torch.from_numpy(transforms).cuda()
                    stack=warp_bev(stack,trans)[0][None]
                    source_tensor=self.temporal.predict_at(stack,[r.observation_us for r in selected],record['timestamp'])
                else:source_tensor=torch.from_numpy(source_feature)[None].cuda()
                outputs,_=self.fusion(ego_tensor,source_tensor,torch.from_numpy(transform)[None].cuda(),
                    torch.tensor([(record['timestamp']-source_row['observation_us'])/1e6],device='cuda'),torch.tensor([float(action=='P')],device='cuda'))
                pred=self.fusion.decode(outputs)[0]
            else:pred=fuse_structured(ego_pred,map_to_ego(source_pred,transform),self.nms,self.temperature,device='cuda')
        buffer=io.BytesIO()
        if self.cache_results:np.savez(buffer,boxes=pred.boxes,scores=pred.scores,labels=pred.labels)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        information=dict(key=key,action=action,seconds=seconds,serialized_output_bytes=buffer.tell(),file=str(cached.relative_to(self.output)),
            observation_us=source_row['observation_us'],frame_us=record['timestamp'],real_history=[r.packet_id for r in selected],
            prediction_steps=int(math.ceil((record['timestamp']-selected[-1].observation_us)/((selected[-1].observation_us-selected[0].observation_us)/2))) if action=='P' else 0)
        result=(pred,information)
        if self.cache_results:
            cached.parent.mkdir(exist_ok=True);cached.write_bytes(buffer.getvalue());atomic_json(metadata,information);self.memo[digest]=result
        return result

def choose_threshold(family):
    name='bevformer_vehicle';cache=DetectorCache(name,'development')
    frames=[]
    for r in records('cooperative','development'):
        gt,labels=gt_roi(r);frames.append((cache.detections(cache.by_token[r['vehicle_token']]).roi(),gt,labels))
    threshold,scores=choose_score_threshold(frames)
    result=dict(threshold=threshold,ego_development_f1=scores,cache_sha256=cache.fingerprint,partition='development')
    path=ROOT/'runs/choices'/(family+'_ego_threshold.json');atomic_json(path,result);return result

def main(args):
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    if args.partition=='test':
        from freeze_evaluation_choices import verify_frozen
        verify_frozen()
    if args.partition=='test' and not (ROOT/'manifests/evaluation_choices_frozen.json').is_file():raise ValueError('Test locked until all choices freeze')
    threshold_file=ROOT/'runs/choices'/(args.family+'_ego_threshold.json')
    choice=json.loads(threshold_file.read_text()) if threshold_file.exists() else choose_threshold(args.family)
    threshold=choice['threshold'];ego=DetectorCache('bevformer_vehicle',args.partition)
    source=DetectorCache(args.source,args.partition)
    output=ROOT/'cache/branches'/args.family/args.source/args.partition;output.mkdir(parents=True,exist_ok=True)
    inputs=dict(family=args.family,source=args.source,partition=args.partition,ego_cache_sha256=ego.fingerprint,
        source_cache_sha256=source.fingerprint,threshold_sha256=sha(threshold_file),code_sha256=sha(__file__))
    for name in (['registered_fusion','temposense'] if args.family=='F' else []):inputs[name]=sha(ROOT/'runs'/name/'best.pt')
    if args.family=='S':inputs['structured_choices']=sha(ROOT/'runs/choices/S_structured_parameters.json')
    if (output/'inputs.json').exists():assert json.loads((output/'inputs.json').read_text())==inputs,'Cached branches belong to different inputs/code'
    else:atomic_json(output/'inputs.json',inputs)
    manifest=output/'manifest.json'
    if manifest.exists():print('Branch cache complete',str(output),flush=True);return
    runner=BranchRunner(args.family,ego,source,args.partition,output);coop=records('cooperative',args.partition)
    per_scene=collections.defaultdict(list)
    for r in coop:per_scene[r['sequence_id']].append(r)
    traces=json.loads((ROOT/'traces/trace_manifest.json').read_text());training_rows=[];frozen_rows=[]
    # Real NPY/NPZ payload decoding is profiled before replay, with bytes already
    # in RAM as after transport. Measured decoding is charged to packet usability.
    decode={};payload_bytes=[]
    for sid,rows in source.by_scene.items():
        for r in rows:
            file=source.root/r['feature' if args.family=='F' else 'detections'];data=file.read_bytes()
            start=time.perf_counter();loaded=np.load(io.BytesIO(data),allow_pickle=False)
            if isinstance(loaded,np.lib.npyio.NpzFile):
                for k in loaded.files:loaded[k]
                loaded.close()
            elif args.family=='F':r['_summary']=loaded.mean((1,2),dtype=np.float32)
            decode[r['token']]=int(math.ceil((time.perf_counter()-start)*1e6));payload_bytes.append(len(data))
    sink=output/'frames.jsonl.writing'
    with sink.open('w') as stream:
        for trace_file in traces['files']:
            active=traces.get('active_conditions',{})
            if active and trace_file not in active:continue
            trace_path=ROOT/'traces'/trace_file;assert sha(trace_path)==traces['files'][trace_file]
            with np.load(trace_path,allow_pickle=False) as data:delays=dict(zip(data['frame_id'].tolist(),data['network_delay_us'].tolist()))
            for scenario,frames in sorted(per_scene.items()):
                src=source.by_scene[scenario];ready=serial_source_ready([r['observation_us'] for r in src],[r['inference_seconds']*1e6 for r in src])
                packets=[]
                for r,when in zip(src,ready):
                    arrival=when+delays[r['frame_id']]
                    packets.append(Observation(r['token'],scenario,args.source,args.family,r['observation_us'],arrival,arrival+decode[r['token']],r['token'],r['token']))
                packets.sort(key=lambda p:(p.usable_us,p.packet_id));bank=FeatureBank();bank.reset(scenario);cursor=0;previous_decision=None
                for r in frames:
                    er=ego.by_token[r['vehicle_token']];ego_pred=ego.detections(er).roi();gt,labels=gt_roi(r)
                    eo=overlap_matrix(ego_pred.boxes,gt);ego_loss=match(ego_pred,gt,labels,threshold=threshold,overlaps=eo)['loss']
                    # An asynchronous ego stream uses its own measured encoder
                    # time; monotone decision timestamps handle duration jitter.
                    decision=r['timestamp']+int(math.ceil(er['inference_seconds']*1e6))
                    if previous_decision is not None:decision=max(decision,previous_decision)
                    previous_decision=decision;arrived=[]
                    while cursor<len(packets) and packets[cursor].usable_us<=decision:arrived.append(packets[cursor]);cursor+=1
                    bank.advance(decision,arrived);snapshot=bank.snapshot(args.source,args.family,r['timestamp'])
                    possible=[]
                    if snapshot.latest is not None:possible=['O']
                    if args.family=='F' and snapshot.prediction_history():possible.append('P')
                    results={};before=snapshot.fingerprint
                    for action in possible:
                        prediction,info=runner.branch(r,er,snapshot,action);matrix=overlap_matrix(prediction.boxes,gt)
                        loss=match(prediction,gt,labels,threshold=threshold,overlaps=matrix)['loss']
                        results[action]=dict(info,loss=loss,delta=loss-ego_loss)
                    assert before==snapshot.fingerprint,'Counterfactual execution changed history'
                    for budget in active.get(trace_file,traces['deadline_budgets_ms']):
                        deadline=r['timestamp']+budget*1000;descriptor_start=time.perf_counter();descriptors={}
                        for action in possible:descriptors[action]=make_descriptor(snapshot,action,ego_pred,threshold,deadline,temporal_change=runner.temporal_change(snapshot))
                        descriptor_seconds=time.perf_counter()-descriptor_start
                        candidates={}
                        for action,x in descriptors.items():
                            candidates[action]=dict(results[action],x=sanitized(x))
                            if deadline>=decision and args.partition!='test':training_rows.append((x,results[action]['delta'],scenario))
                        row=dict(scenario=scenario,frame_id=r['frame_id'],trace=trace_file,budget_ms=budget,
                            frame_us=r['timestamp'],decision_us=decision,deadline_us=deadline,
                            ego_inference_seconds=er['inference_seconds'],ego_loss=ego_loss,
                            empty_loss=float(len(gt)>0),
                            descriptor_seconds=descriptor_seconds,candidates=candidates,
                            history_records=[asdict(p) for p in snapshot.real[-3:]],history_length=len(snapshot.real),
                            snapshot=before,ego_file=str(ego.root/er['detections']))
                        stream.write(json.dumps(row,allow_nan=False)+'\n');frozen_rows.append((scenario,budget))
                print(args.family,args.source,args.partition,trace_file,scenario,'complete',flush=True)
    sink.replace(output/'frames.jsonl')
    if args.partition!='test':
        xs=np.stack([r[0] for r in training_rows]) if training_rows else np.empty((0,20))
        np.savez_compressed(output/'selector_rows.npz',x=xs,delta=np.array([r[1] for r in training_rows],np.float32),scenario=np.array([r[2] for r in training_rows],dtype='<U8'))
    provenance=dict(kind='real_frozen_counterfactual_cache',family=args.family,source=args.source,partition=args.partition,
        ego_manifest_sha256=ego.fingerprint,source_manifest_sha256=source.fingerprint,
        frames_sha256=sha(output/'frames.jsonl'),score_threshold=threshold,causal_cache_audit='passed',
        source_runtime='actual synchronized detector forward; includes source task head; excludes image read and network; receiver payload decode charged',
        receiver_runtime='ego model forward, descriptors, selected branch including H2D and output serialization; disk excluded',
        prediction='immutable three-real-frame snapshots; future-source timestamps excluded',
        trace_seeds=traces['seeds'],delay_medians_ms=traces['medians_ms'],deadline_budgets_ms=traces['deadline_budgets_ms'],
        active_conditions=traces.get('active_conditions',{}),
        payload_bytes=dict(min=min(payload_bytes),median=float(np.median(payload_bytes)),max=max(payload_bytes)),
        counterfactual_branches=len(runner.memo),rows=len(frozen_rows),completed_unix=time.time())
    atomic_json(manifest,provenance)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['F','S'],required=True);p.add_argument('--source',required=True)
    p.add_argument('--partition',choices=['selector_fit','development','calibration','test'],required=True);args=p.parse_args()
    with (ROOT/'status'/('branch_'+args.family+'_'+args.source+'_'+args.partition+'.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        main(args)
