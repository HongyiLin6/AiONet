"""Shared, provenance-checked I/O for the new SPD study."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import numpy as np
from structured_fusion import Detections

ROOT=Path(os.environ.get('AIONET_ROOT',str(Path(__file__).resolve().parents[2] / 'workdir')))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)

def records(role,partition):
    with (ROOT/'infos'/(role+'_'+partition+'.pkl')).open('rb') as f:return pickle.load(f)['infos']

class DetectorCache:
    def __init__(self,name,partition):
        self.root=ROOT/'cache/detectors'/name/partition
        self.manifest_path=self.root/'manifest.json';self.manifest=json.loads(self.manifest_path.read_text())
        self.rows=self.manifest['records'];self.by_token={r['token']:r for r in self.rows}
        self.by_frame={r['frame_id']:r for r in self.rows}
        assert len(self.rows)==len(self.by_token)==self.manifest['frames']
        self.by_scene={}
        for r in self.rows:self.by_scene.setdefault(r['scenario'],[]).append(r)
        for v in self.by_scene.values():v.sort(key=lambda r:r['observation_us'])
    def feature(self,row):return np.load(self.root/row['feature'],allow_pickle=False).astype(np.float32)
    def detections(self,row):
        with np.load(self.root/row['detections'],allow_pickle=False) as d:
            return Detections(d['boxes'],d['scores'],d['labels'],d['velocity'] if 'velocity' in d else None)
    @property
    def fingerprint(self):return sha(self.manifest_path)

def source_to_ego(source_row,cooperative_row):
    matrix=np.linalg.inv(cooperative_row['lidar2global'])@np.asarray(source_row['lidar2global'],np.float64)
    offset=cooperative_row['system_error_offset']
    matrix[:2,3]+=[float(offset['delta_x'] or 0),float(offset['delta_y'] or 0)]
    return matrix.astype(np.float32)

def gt_roi(record):
    b=record['gt_boxes'];mask=(b[:,0]>=0)&(b[:,0]<=51.2)&(abs(b[:,1])<=51.2)
    return b[mask],record['class'][mask]
