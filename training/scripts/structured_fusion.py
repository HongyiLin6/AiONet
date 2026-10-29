"""Fixed structured-output interface for the held-out-model experiment."""
from dataclasses import dataclass
import numpy as np
from reliability_core import iou3d

CLASSES=('Car','Truck','Van','Bus','Motorcyclist','Cyclist','Pedestrian')

@dataclass
class Detections:
    boxes:np.ndarray
    scores:np.ndarray
    labels:np.ndarray
    velocity:np.ndarray=None
    def __post_init__(self):
        self.boxes=np.asarray(self.boxes,np.float32).reshape(-1,7)
        self.scores=np.asarray(self.scores,np.float32).reshape(-1)
        self.labels=np.asarray(self.labels,np.int64).reshape(-1)
        if not len(self.boxes)==len(self.scores)==len(self.labels):raise ValueError('Mismatched detection arrays')
        if not np.isfinite(self.boxes).all() or not np.isfinite(self.scores).all():raise ValueError('Nonfinite detection')
        if (self.boxes[:,3:6]<=0).any() or not ((self.scores>=0)&(self.scores<=1)).all():raise ValueError('Invalid dimensions or confidence')
        if not ((self.labels>=0)&(self.labels<len(CLASSES))).all():raise ValueError('Unknown class')
        if self.velocity is not None:
            self.velocity=np.asarray(self.velocity,np.float32).reshape(-1,2)
            if len(self.velocity)!=len(self.boxes):raise ValueError('Velocity length differs')
    def select(self,indices):
        return Detections(self.boxes[indices],self.scores[indices],self.labels[indices],None if self.velocity is None else self.velocity[indices])
    def roi(self):
        b=self.boxes;return self.select((b[:,0]>=0)&(b[:,0]<=51.2)&(abs(b[:,1])<=51.2))

def map_to_ego(detection,source_to_ego):
    matrix=np.asarray(source_to_ego,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Invalid frame transform')
    if not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-3):raise ValueError('Expected rigid frame transform')
    boxes=detection.boxes.copy();boxes[:,:3]=boxes[:,:3]@matrix[:3,:3].T+matrix[:3,3]
    # Upright-box convention: transform heading and retain its planar yaw.
    direction=np.column_stack((np.cos(boxes[:,6]),np.sin(boxes[:,6]),np.zeros(len(boxes))))@matrix[:3,:3].T
    boxes[:,6]=np.arctan2(direction[:,1],direction[:,0])
    velocity=None
    if detection.velocity is not None:
        v=np.column_stack((detection.velocity,np.zeros(len(boxes))))@matrix[:3,:3].T;velocity=v[:,:2]
    return Detections(boxes,detection.scores.copy(),detection.labels.copy(),velocity)

def temperature_score(scores,temperature):
    if temperature<=0:raise ValueError('Positive temperature required')
    p=np.clip(np.asarray(scores),1e-6,1-1e-6)
    return 1/(1+np.exp(-np.log(p/(1-p))/temperature))

def fuse_structured(ego,source_in_ego,iou_threshold=.3,source_temperature=1.,device='cpu'):
    if not 0<iou_threshold<1:raise ValueError('Invalid NMS threshold')
    ego=ego.roi();source=source_in_ego.roi()
    combined=Detections(np.concatenate((ego.boxes,source.boxes)),
        np.concatenate((ego.scores,temperature_score(source.scores,source_temperature))),np.concatenate((ego.labels,source.labels)))
    selected=[]
    for label in range(len(CLASSES)):
        indices=np.flatnonzero(combined.labels==label);indices=indices[np.argsort(-combined.scores[indices],kind='stable')]
        b=combined.boxes[indices]
        if device!='cpu' and len(b):
            import torch
            from mmdet3d.core.bbox import LiDARInstance3DBoxes
            boxes=LiDARInstance3DBoxes(torch.from_numpy(b).to(device),box_dim=7,origin=(.5,.5,.5))
            overlap=boxes.overlaps(boxes,boxes).detach().cpu().numpy()
        else:
            overlap=np.eye(len(b))
            for i in range(len(b)):
                for j in range(i+1,len(b)):overlap[i,j]=overlap[j,i]=iou3d(b[i],b[j])
        active=list(range(len(b)))
        while active:
            first=active.pop(0);selected.append(int(indices[first]))
            active=[j for j in active if overlap[first,j]<=iou_threshold]
    selected=np.array(sorted(selected,key=lambda j:(-combined.scores[j],j)),np.int64)
    return combined.select(selected)
