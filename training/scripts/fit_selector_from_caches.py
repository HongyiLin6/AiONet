"""Merge prescribed non-test branch caches, fit, and calibrate the selector."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from experiment_io import ROOT,sha,atomic_json

def main(args):
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    sources=['bevformer_infrastructure'] if args.family=='F' else ['bevformer_infrastructure','petr_infrastructure']
    arrays={'x':[],'delta':[],'scenario':[]};manifests={}
    for source in sources:
        for partition in ['selector_fit','development','calibration']:
            folder=ROOT/'cache/branches'/args.family/source/partition
            meta=json.loads((folder/'manifest.json').read_text());assert meta['causal_cache_audit']=='passed'
            with np.load(folder/'selector_rows.npz',allow_pickle=False) as d:
                for k in arrays:arrays[k].append(d[k])
            manifests[source+':'+partition]=meta
    folder=ROOT/'cache/selector'/args.family;folder.mkdir(parents=True,exist_ok=True)
    output=folder/'rows.npz';np.savez_compressed(output,**{k:np.concatenate(v) for k,v in arrays.items()})
    first=next(iter(manifests.values()));threshold=first['score_threshold']
    assert all(m['score_threshold']==threshold for m in manifests.values())
    meta=dict(perception_checkpoint_sha256={k:m['ego_manifest_sha256']+':'+m['source_manifest_sha256'] for k,m in manifests.items()},
        score_threshold=threshold,causal_cache_audit='passed',trace_seeds=first['trace_seeds'],deadline_budgets_ms=first['deadline_budgets_ms'],
        delay_medians_ms=first['delay_medians_ms'],cache_sha256=sha(output),source_manifests=manifests)
    atomic_json(folder/'manifest.json',meta)
    for ablate in ([False,True] if args.family=='F' else [False]):
        destination=ROOT/'runs/selectors'/(args.family+('_no_age_horizon' if ablate else ''))
        if (destination/'calibration.json').exists():continue
        if destination.exists():raise RuntimeError('Incomplete selector folder must be inspected before rerun: '+str(destination))
        command=[sys.executable,str(Path(__file__).resolve().parent/'selector_train.py'),'--cache',str(output),'--cache-manifest',str(folder/'manifest.json'),
            '--split',str(ROOT/'manifests/split.json'),'--output',str(destination)]
        if ablate:command.append('--without-age-horizon')
        subprocess.run(command,check=True)
    if args.family=='F':
        from risk_coverage import freeze
        freeze()
    atomic_json(ROOT/'status'/('selector_'+args.family+'_complete.json'),dict(family=args.family,cache_sha256=sha(output)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['F','S'],required=True);main(p.parse_args())
