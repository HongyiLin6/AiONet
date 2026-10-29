"""Paired real detector representations for registered-pair experiments."""
import bisect,json
import numpy as np
import torch
from torch.utils.data import Dataset
from experiment_io import ROOT,DetectorCache,records,source_to_ego,gt_roi

class PaperPairs(Dataset):
    def __init__(self,config,partition,delay_ms=None):
        self.config=config;self.rows=records('cooperative',partition);self.partition=partition
        self.ego=DetectorCache(config['vehicle']+'_vehicle',partition)
        self.source=DetectorCache(config['source']+'_infrastructure',partition)
        self.delay_ms=config.get('delay_ms',0.) if delay_ms is None else delay_ms
        self.valid=[]
        for i,r in enumerate(self.rows):
            available=self.source.by_scene.get(r['sequence_id'],[])
            times=[q['observation_us'] for q in available]
            if bisect.bisect_right(times,r['timestamp']-self.delay_ms*1000)>0:self.valid.append(i)
    def __len__(self):return len(self.valid)
    def packet(self,cache,row):
        with np.load(cache.root/row['query'],allow_pickle=False) as a:
            packet={k:torch.from_numpy(a[k].astype(np.float32)) for k in ['query','xyz']}
        pred=cache.detections(row);order=np.argsort(-pred.scores,kind='stable')[:300]
        boxes=np.zeros((300,7),np.float32);boxes[:,3:6]=1.
        scores=np.zeros(300,np.float32);velocity=np.zeros((300,2),np.float32)
        boxes[:len(order)]=pred.boxes[order];scores[:len(order)]=pred.scores[order]
        if pred.velocity is not None:velocity[:len(order)]=np.nan_to_num(pred.velocity[order])
        packet.update(boxes=torch.from_numpy(boxes),scores=torch.from_numpy(scores),velocity=torch.from_numpy(velocity))
        if row.get('feature'):packet['bev']=torch.from_numpy(cache.feature(row))
        return packet
    def __getitem__(self,index):
        r=self.rows[self.valid[index]];er=self.ego.by_token[r['vehicle_token']]
        scene=self.source.by_scene[r['sequence_id']];times=[q['observation_us'] for q in scene]
        index=bisect.bisect_right(times,r['timestamp']-self.delay_ms*1000)-1;sr=scene[index]
        boxes,labels=gt_roi(r);mask=(r['gt_boxes'][:,0]>=0)&(r['gt_boxes'][:,0]<=51.2)&(abs(r['gt_boxes'][:,1])<=51.2)
        velocity=r['gt_velocity'][mask]
        return dict(ego=self.packet(self.ego,er),source=self.packet(self.source,sr),matrix=torch.from_numpy(source_to_ego(sr,r)),
            age=(r['timestamp']-sr['observation_us'])/1e6,boxes=boxes,labels=labels,velocity=velocity,record=r,source_row=sr,ego_row=er)

def collate(items):
    result={name:{key:torch.stack([row[name][key] for row in items]) for key in items[0][name]} for name in ['ego','source']}
    result.update(matrix=torch.stack([r['matrix'] for r in items]),age=torch.tensor([r['age'] for r in items],dtype=torch.float32))
    for name in ['boxes','labels','velocity','record','source_row','ego_row']:result[name]=[r[name] for r in items]
    return result

def forward(model,batch):
    packets=[{k:v.cuda(non_blocking=True) for k,v in batch[name].items()} for name in ['ego','source']]
    return model(*packets,batch['matrix'].cuda(),batch['age'].cuda())
