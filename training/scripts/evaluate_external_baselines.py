"""Compare detection quality on exactly the same delayed camera-frame pairs.

This table is an uncensored accuracy comparison. Deadline delivery is evaluated
separately by evaluate_policy_cache.py. Full EMIFF and camera-BEV adaptations
are identified explicitly; their runtime scopes are not conflated.
"""
import argparse
import io
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from experiment_io import ROOT,records,gt_roi,DetectorCache,source_to_ego,sha,atomic_json
from structured_fusion import Detections
from evaluation_metrics import summarize
from evaluate_policy_cache import load_selector,score_all

def from_file(path):
    with np.load(path,allow_pickle=False) as d:return Detections(d['boxes'],d['scores'],d['labels']).roi()

def main(args):
    import mmcv
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    from freeze_evaluation_choices import verify_frozen
    verify_frozen()
    branch=ROOT/'cache/branches/F/bevformer_infrastructure/test'
    manifest=json.loads((branch/'manifest.json').read_text())
    rows=[json.loads(line) for line in (branch/'frames.jsonl').open()]
    rows=[r for r in rows if r['trace']=='network_seed17_median234p9.npz' and r['budget_ms']==100]
    output=ROOT/'results/external'/args.method;output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():return
    coop={r['frame_id']:r for r in records('cooperative','test')}
    ego=DetectorCache('bevformer_vehicle','test');source=DetectorCache('bevformer_infrastructure','test')
    camera_v={r['token']:r for r in records('vehicle','test')};camera_i={r['token']:r for r in records('infrastructure','test')}
    model=None;scores=None;q=None;checkpoint=None;portable=None;threshold=None
    if args.method in ('v2xvit','select2col'):
        from r2_fusion_model import RegisteredFusion
        checkpoint=ROOT/'runs'/args.method/'best.pt';assert (checkpoint.parent/'complete.json').exists()
        model=RegisteredFusion(mmcv.Config.fromfile(str(ROOT/'configs/bevformer_vehicle.py')),method=args.method)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict']);model.cuda().eval()
    elif args.method=='emiff':
        import r2_emiff
        from mmdet3d.models import build_model
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from emiff_data import image_pair
        checkpoint=ROOT/'runs/emiff/best.pt';assert (checkpoint.parent/'complete.json').exists()
        model=build_model(mmcv.Config.fromfile(str(ROOT/'configs/emiff.py')).model)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict']);model.cuda().eval()
    elif args.method=='aionet_calibrated':scores,q=score_all(rows,load_selector('F'))
    elif args.method=='aionet_selected':
        import joblib
        from aionet.selector import score
        if args.model_dir is None:raise ValueError('Pass a fitted descriptor selector with --model-dir')
        portable=joblib.load(args.model_dir/'model.joblib')
        if portable['dimensions']!=20:raise ValueError('This comparison uses the 20-descriptor registered path')
        threshold=json.loads((args.model_dir/'calibration.json').read_text())['chosen']['threshold']
        scores={}
        for index,row in enumerate(rows):
            for action,candidate in row['candidates'].items():
                x=np.array([[np.nan if v is None else v for v in candidate['x']]],np.float32)
                scores[index,action]=float(score(portable,x)[0])
    frames=[];times=[];actions=[]
    with torch.no_grad():
        for index,row in enumerate(rows):
            record=coop[row['frame_id']];er=ego.by_token[record['vehicle_token']];prediction=ego.detections(er).roi();action='0'
            candidates=row['candidates'];file=output/(row['frame_id']+'.npz')
            if args.method=='aionet_selected':
                choices=[(scores[index,a],a=='O',a) for a in candidates]
                winner=max(choices) if choices else None
                if winner and winner[0]>threshold:action=winner[-1];prediction=from_file(branch/candidates[action]['file'])
            elif args.method=='aionet_calibrated':
                viable=[(scores[(index,a)]+q,a!='O',a) for a in candidates] if row['decision_us']<=row['deadline_us'] else []
                chosen=min(viable) if viable else None
                if chosen and chosen[0]<0:action=chosen[-1];prediction=from_file(branch/candidates[action]['file'])
            elif args.method=='aionet_component':
                if 'O' in candidates:action='O';prediction=from_file(branch/candidates['O']['file'])
            elif 'O' in candidates:
                token=candidates['O']['real_history'][-1];sr=source.by_token[token];context=None;available=True
                if args.method=='select2col':
                    from select2col_inputs import context as make_context
                    try:context={k:v[None].cuda() for k,v in make_context(ego,source,er,sr).items()}
                    except ValueError:available=False
                if available:
                    if args.method=='emiff':
                        image,meta=image_pair(camera_v[er['token']],camera_i[sr['token']],record)
                        meta['box_type_3d']=LiDARInstance3DBoxes
                    else:
                        e=torch.from_numpy(ego.feature(er))[None].cuda();s=torch.from_numpy(source.feature(sr))[None].cuda()
                        transform=torch.from_numpy(source_to_ego(sr,record))[None].cuda()
                        age=torch.tensor([(record['timestamp']-sr['observation_us'])/1e6],device='cuda')
                    torch.cuda.synchronize();begin=time.perf_counter()
                    if args.method=='emiff':
                        r=model.simple_test(image[None].cuda(),[meta])[0];b=r['boxes_3d']
                        prediction=Detections(torch.cat((b.gravity_center,b.tensor[:,3:7]),1).cpu().numpy(),r['scores_3d'].cpu().numpy(),r['labels_3d'].cpu().numpy()).roi()
                    else:
                        values,_=model(e,s,transform,age,torch.zeros_like(age),context=context);prediction=model.decode(values)[0]
                    torch.cuda.synchronize();times.append(time.perf_counter()-begin);action='O'
            np.savez_compressed(file,boxes=prediction.boxes,scores=prediction.scores,labels=prediction.labels)
            gt,labels=gt_roi(record);frames.append((prediction,gt,labels));actions.append(action)
            if index%100==0:print(args.method,index+1,'/',len(rows),flush=True)
    metrics=summarize(frames,score_threshold=manifest['score_threshold'])
    provenance=dict(method=args.method,metrics=metrics,source_frames_manifest_sha256=sha(branch/'frames.jsonl'),
        checkpoint_sha256=sha(checkpoint) if checkpoint else None,selector_sha256=sha(args.model_dir/'model.joblib') if args.model_dir else None,delay_seed=17,delay_median_ms=234.9,
        protocol='uncensored cooperative detection accuracy on common causal delayed frame pairs; deadline-filtered scores are a separate experiment',
        gate='same fitted calibrated relative-loss model, without runtime feasibility filtering; late ego decision retains ego' if args.method=='aionet_calibrated' else None,
        missing_input_fallback='common frozen BEVFormer ego output',observed_cooperative_frames=actions.count('O'),predicted_cooperative_frames=actions.count('P'),
        timing_scope='two-camera model forward' if args.method=='emiff' else 'fusion module and shared decoder only' if model else 'reused cached predictions',
        model_forward_p50_ms=float(np.median(times)*1000) if times else None,
        model_forward_p95_ms=float(np.quantile(times,.95)*1000) if times else None)
    atomic_json(output/'summary.json',provenance);print(json.dumps(provenance),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['emiff','v2xvit','select2col','aionet_component','aionet_calibrated','aionet_selected'],required=True);p.add_argument('--model-dir',type=Path)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main(p.parse_args())
