"""Evaluate registered fusion and detector controls on every cooperative frame."""
import argparse,json
from pathlib import Path
import mmcv,numpy as np,torch
from torch.utils.data import DataLoader
from experiment_io import ROOT,DetectorCache,records,gt_roi,source_to_ego,sha,atomic_json
from aionet.paper.experiments import get_experiment
from aionet.paper.metrics import evaluate_car
from aionet.structured_fusion import map_to_ego,Detections
from paper_data import PaperPairs,collate,forward
from paper_model import FullPaperFusion

def evaluate(spec,compute_nds=False,attributes=None):
    predictions={};ego=DetectorCache(spec['vehicle']+'_vehicle','test');source=DetectorCache(spec['source']+'_infrastructure','test')
    if not spec['operator'].endswith('_only'):
        data=PaperPairs(spec,'test');model=FullPaperFusion(mmcv.Config.fromfile(str(ROOT/'configs/bevformer_vehicle.py')),spec['operator']).cuda().eval()
        saved=torch.load(ROOT/'runs/paper'/spec['name']/'final.pt',map_location='cpu')
        if saved['provenance']['spec']!=spec:raise ValueError('Model configuration differs from the requested experiment')
        model.load_state_dict(saved['state_dict'])
        with torch.no_grad():
            for b in DataLoader(data,batch_size=spec['batch_size'],shuffle=False,num_workers=4,collate_fn=collate):
                for r,pred in zip(b['record'],model.decode(forward(model,b))):predictions[r['frame_id']]=pred
    frames=[];directory=ROOT/'results/paper'/spec['name'];directory.mkdir(parents=True,exist_ok=True)
    annotations={} if attributes is None else json.loads(Path(attributes).read_text())
    for r in records('cooperative','test'):
        er=ego.by_token[r['vehicle_token']];pred=predictions.get(r['frame_id'],ego.detections(er).roi())
        if spec['operator']=='infrastructure_only':
            sr=source.by_token[r['infrastructure_token']];pred=map_to_ego(source.detections(sr),source_to_ego(sr,r)).roi()
        boxes,labels=gt_roi(r);mask=(r['gt_boxes'][:,0]>=0)&(r['gt_boxes'][:,0]<=51.2)&(abs(r['gt_boxes'][:,1])<=51.2)
        velocity=np.asarray(r['gt_velocity'][mask],float).copy()
        velocity[~r['velocity_valid'][mask]]=np.nan
        frame=dict(token=r['frame_id'],pred=pred,gt=boxes,gt_velocity=velocity)
        if r['frame_id'] in annotations:
            extra=annotations[r['frame_id']]
            if set(extra)-{'gt_velocity','pred_velocity','gt_attributes','pred_attributes'}:raise ValueError('Attribute sidecar contains unsupported fields')
            frame.update(extra)
        frames.append(frame)
        np.savez_compressed(directory/(r['frame_id']+'.npz'),boxes=pred.boxes,scores=pred.scores,labels=pred.labels,velocity=pred.velocity if pred.velocity is not None else np.empty((0,2)))
    result=evaluate_car(frames,compute_nds=compute_nds,device='cuda')
    atomic_json(directory/'metrics.json',dict(experiment=spec,metrics=result,frames=len(frames),ego_cache_sha256=ego.fingerprint,source_cache_sha256=source.fingerprint))
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--nds',action='store_true');p.add_argument('--attributes',type=Path);a=p.parse_args()
    evaluate(get_experiment(a.experiment),a.nds,a.attributes)
