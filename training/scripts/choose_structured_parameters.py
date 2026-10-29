"""Freeze box-interface choices using only known-model development outputs."""
import bisect
import json
import os
import itertools
from experiment_io import ROOT,DetectorCache,records,gt_roi,source_to_ego,atomic_json
from structured_fusion import map_to_ego,fuse_structured
from evaluation_metrics import summarize

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    out=ROOT/'runs/choices/S_structured_parameters.json'
    if out.exists():return
    ego=DetectorCache('bevformer_vehicle','development');sources=[DetectorCache(n,'development') for n in ['bevformer_infrastructure','petr_infrastructure']]
    examples=[]
    for source in sources:
        for r in records('cooperative','development'):
            scene=source.by_scene[r['sequence_id']];times=[x['observation_us'] for x in scene];latest=bisect.bisect_right(times,r['timestamp'])-1
            if latest<0:continue
            sr=scene[max(0,latest-2)];ep=ego.detections(ego.by_token[r['vehicle_token']]).roi()
            sp=map_to_ego(source.detections(sr),source_to_ego(sr,r)).roi();gt,labels=gt_roi(r)
            examples.append((ep,sp,gt,labels))
    values=[]
    for threshold,temperature in itertools.product([.1,.25,.5],[.5,1.,2.]):
        frames=[(fuse_structured(e,s,threshold,temperature,device='cuda'),g,l) for e,s,g,l in examples]
        scores=summarize(frames);row=dict(nms_iou=threshold,temperature=temperature,development_AP25=scores['AP25']);values.append(row)
        print(json.dumps(row),flush=True)
    best=min(values,key=lambda r:(-r['development_AP25'],abs(r['temperature']-1),abs(r['nms_iou']-.25)))
    atomic_json(out,dict(best,selection='known-model development AP25; no DETR3D data or weights',
        candidates=values,ego_manifest_sha256=ego.fingerprint,known_sources={s.manifest['name']:s.fingerprint for s in sources}))

if __name__=='__main__':main()
