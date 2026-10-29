"""AiONet R2 loss, causal candidate contract, calibration and commit rules.

Boxes use [x,y,z,l,w,h,yaw] with geometric centre z, metres and radians.
All times are seconds on one common clock. Callers supply detections from their frozen perception stack.
"""
from dataclasses import dataclass
import math
import numpy as np

FEATURE_NAMES = (
    'observation_age','forecast_horizon','target_gap','deadline_slack',
    'ego_count','ego_confidence_mean','ego_confidence_std','ego_speed_median',
    'ego_speed_p90','fov_overlap','real_temporal_change','real_history_length',
    'missing_temporal_change','missing_speed','missing_fov',
    'representation_F','representation_Q','representation_S','action_O','action_P')

def _corners(box):
    x,y,_,length,width,_,yaw = box
    vertices=np.array([[-length/2,-width/2],[length/2,-width/2],
                       [length/2,width/2],[-length/2,width/2]],dtype=float)
    c,s=np.cos(yaw),np.sin(yaw)
    return vertices @ np.array([[c,s],[-s,c]]) + [x,y]

def _cross(a,b):
    return float(a[0]*b[1]-a[1]*b[0])

def _clip(subject,clip):
    poly=list(subject)
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        old=poly; poly=[]
        if not old: break
        previous=old[-1]; dp=_cross(b-a,previous-a)
        for current in old:
            dc=_cross(b-a,current-a)
            if (dc>=0)!=(dp>=0):
                poly.append(previous+(current-previous)*(dp/(dp-dc)))
            if dc>=0: poly.append(current)
            previous,dp=current,dc
    return np.asarray(poly)

def iou3d(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=(7,) or b.shape!=(7,) or not np.isfinite([a,b]).all():
        raise ValueError('Expected two finite seven-dimensional boxes')
    if (a[3:6]<=0).any() or (b[3:6]<=0).any():
        raise ValueError('Box dimensions must be positive')
    height=max(0.,min(a[2]+a[5]/2,b[2]+b[5]/2)-max(a[2]-a[5]/2,b[2]-b[5]/2))
    if height==0: return 0.
    poly=_clip(_corners(a),_corners(b))
    if len(poly)<3: return 0.
    area=abs(np.dot(poly[:,0],np.roll(poly[:,1],-1))-np.dot(poly[:,1],np.roll(poly[:,0],-1)))/2
    inter=area*height
    return float(np.clip(inter/(np.prod(a[3:6])+np.prod(b[3:6])-inter),0,1))

def detection_loss(pred_boxes,pred_scores,pred_classes,gt_boxes,gt_classes,score_threshold):
    """Descending-confidence, same-class greedy match; strict IoU > 0.25."""
    boxes=np.asarray(pred_boxes,float).reshape(-1,7)
    scores=np.asarray(pred_scores,float).reshape(-1)
    classes=np.asarray(pred_classes).reshape(-1)
    gt=np.asarray(gt_boxes,float).reshape(-1,7)
    labels=np.asarray(gt_classes).reshape(-1)
    if len(boxes)!=len(scores) or len(scores)!=len(classes) or len(gt)!=len(labels):
        raise ValueError('Prediction/ground-truth array lengths differ')
    if not np.isfinite(scores).all() or not 0<=score_threshold<=1:
        raise ValueError('Invalid score or threshold')
    eligible=np.flatnonzero(scores>=score_threshold)
    order=eligible[np.argsort(-scores[eligible],kind='stable')]
    available=set(range(len(gt))); matches=[]
    for i in order:
        candidates=[(iou3d(boxes[i],gt[j]),j) for j in sorted(available) if classes[i]==labels[j]]
        if not candidates: continue
        overlap,j=max(candidates,key=lambda item:(item[0],-item[1]))
        if overlap>.25:
            available.remove(j); matches.append((int(i),j,overlap))
    n=len(matches); npred=len(order); ngt=len(gt)
    loss=(ngt+npred-2*n+sum(1-m[2] for m in matches))/max(1,ngt+npred)
    return {'loss':float(loss),'matches':matches,'tp':n,'fp':npred-n,'fn':ngt-n}

def calibration_margin(residuals_by_scenario,alpha=.1):
    """Scenario maxima include every prespecified trace, deadline and action."""
    if not 0<alpha<1: raise ValueError('alpha must lie strictly between zero and one')
    maxima=[]
    for values in residuals_by_scenario.values():
        values=np.asarray(values,float)
        if values.size and not np.isfinite(values).all():
            raise ValueError('Nonfinite observed residual')
        maxima.append(float(np.max(values)) if values.size else -math.inf)
    n=len(maxima); k=math.ceil((n+1)*(1-alpha)-1e-12)
    q=math.inf if k>n else max(0.,sorted(maxima)[k-1])
    return {'q':q,'n_scenarios':n,'order_statistic':k,'alpha':alpha,
            'scenario_maxima':dict(zip(residuals_by_scenario,maxima))}

def assert_disjoint_scenarios(parts):
    seen={}
    for name,ids in parts.items():
        if len(set(ids))!=len(ids): raise ValueError('Duplicate scenario within '+name)
        for sid in ids:
            if sid in seen: raise ValueError('Scenario %s overlaps %s and %s'%(sid,seen[sid],name))
            seen[sid]=name

@dataclass(frozen=True)
class Candidate:
    action: str
    representation: str
    observation_time: float
    represented_time: float
    arrival_time: float
    usable_time: float
    remaining_cost: float
    score: float
    compatible: bool=True

    def feasible(self,frame_time,decision_time,deadline):
        vals=(self.observation_time,self.represented_time,self.arrival_time,
              self.usable_time,self.remaining_cost,self.score,frame_time,decision_time,deadline)
        if not np.isfinite(vals).all(): return False
        if self.action not in ('O','P') or self.representation not in ('F','Q','S'): return False
        if self.remaining_cost<0 or self.represented_time<self.observation_time: return False
        if self.action=='O' and not math.isclose(self.represented_time,self.observation_time,abs_tol=1e-9): return False
        if self.action=='P' and not math.isclose(self.represented_time,frame_time,abs_tol=1e-9): return False
        return (self.compatible and self.observation_time<=frame_time
                and self.observation_time<=self.arrival_time<=self.usable_time<=decision_time
                and decision_time+self.remaining_cost<=deadline)

def choose(candidates,frame_time,decision_time,deadline,q,epsilon=0.):
    if q<0 or math.isnan(q) or epsilon<0: raise ValueError('Invalid calibration margin/tolerance')
    feasible=[c for c in candidates if c.feasible(frame_time,decision_time,deadline)]
    if not feasible: return '0'
    best=min(feasible,key=lambda c:(c.score+q,c.remaining_cost,c.action!='O'))
    return best.action if best.score+q < -epsilon else '0'

def commit(action,branch_ready_time,ego_ready_time,deadline):
    """None denotes no output by the deadline; count an empty set in AP."""
    if action not in ('0','O','P'): raise ValueError('Unknown action')
    if action!='0' and branch_ready_time is not None and branch_ready_time<=deadline:
        return action
    return '0' if ego_ready_time is not None and ego_ready_time<=deadline else None

class DescriptorNormalizer:
    """Fit once on selector-fitting scenarios. Flags/one-hot entries stay binary."""
    def fit(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=20 or not len(x): raise ValueError('Expected N x 20 fitting data')
        continuous=x[:,:12]
        med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in continuous.T])
        filled=np.where(np.isfinite(continuous),continuous,med)
        self.median=med; self.mean=filled.mean(0); self.scale=filled.std(0)
        self.scale=np.where(self.scale>1e-12,self.scale,1.)
        return self
    def transform(self,x):
        x=np.array(x,dtype=float,copy=True)
        if x.ndim!=2 or x.shape[1]!=20: raise ValueError('Expected N x 20 data')
        if not np.isin(x[:,12:],[0,1]).all(): raise ValueError('Missing flags and one-hot entries must be binary')
        if not (x[:,15:18].sum(1)==1).all() or not (x[:,18:20].sum(1)==1).all():
            raise ValueError('Invalid representation/action one-hot encoding')
        x[:,:12]=np.clip((np.where(np.isfinite(x[:,:12]),x[:,:12],self.median)-self.mean)/self.scale,-5,5)
        return x.astype(np.float32)
    def state_dict(self):
        return {k:getattr(self,k).tolist() for k in ['median','mean','scale']}

def delay_trace(n,seed,median_ms):
    """Fenton-Wilkinson matched log-normal, paired standardized draws across regimes."""
    means=np.array([36.400,71.472,81.234]); sds=np.array([10.920,19.545,13.478])
    sigma=np.sqrt(np.log1p(np.sum(sds*sds)/np.sum(means)**2))
    return np.random.default_rng(seed).lognormal(np.log(median_ms),sigma,n)/1000.
