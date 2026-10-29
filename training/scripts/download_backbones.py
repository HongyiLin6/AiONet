"""Fetch official ImageNet initialization; never treat it as an SPD checkpoint."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

from experiment_io import ROOT
NAMES=['resnet50_msra-5891d200.pth','resnet101_msra-6cc46731.pth']
def fetch(name):
    url='https://download.openmmlab.com/pretrain/third_party/'+name
    path=ROOT/'weights'/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        ss=requests.Session()
        r=ss.get(url,stream=True,timeout=(30,90)); r.raise_for_status()
        if 'text/html' in r.headers.get('Content-Type',''): raise RuntimeError('HTML response')
        temp=path.with_suffix('.part')
        with temp.open('wb') as f:
            for b in r.iter_content(4*1024*1024): f.write(b)
        temp.replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    prefix=name.rsplit('-',1)[1].split('.')[0]
    if not digest.startswith(prefix): raise RuntimeError('Official hash prefix mismatch for '+name)
    info={'name':name,'source':url,'sha256':digest,'bytes':path.stat().st_size,
          'role':'ImageNet backbone initialization; not a trained SPD detector'}
    print(json.dumps(info),flush=True); return info

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: manifest=list(pool.map(fetch,NAMES))
    (ROOT/'manifests/initialization_weights.json').write_text(json.dumps(manifest,indent=2))
