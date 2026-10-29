"""Shared exact-3D matching for AP, F1 and bounded relative-loss labels."""
import numpy as np
from .structured_fusion import Detections,CLASSES
from .reliability_core import iou3d

def overlap_matrix(pred,gt,device='cuda'):
    if not len(pred) or not len(gt):return np.zeros((len(pred),len(gt)),np.float32)
    if device=='cpu':return np.array([[iou3d(a,b) for b in gt] for a in pred])
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    a=LiDARInstance3DBoxes(torch.as_tensor(pred,device=device),box_dim=7,origin=(.5,.5,.5))
    b=LiDARInstance3DBoxes(torch.as_tensor(gt,device=device),box_dim=7,origin=(.5,.5,.5))
    return a.overlaps(a,b).cpu().numpy()

def match(pred,gt_boxes,gt_labels,iou=.25,threshold=0.,overlaps=None,device='cuda'):
    overlaps=overlap_matrix(pred.boxes,gt_boxes,device) if overlaps is None else overlaps
    eligible=np.flatnonzero(pred.scores>=threshold);order=eligible[np.argsort(-pred.scores[eligible],kind='stable')]
    available=np.ones(len(gt_boxes),bool);tp=np.zeros(len(pred.scores),bool);quality=np.zeros(len(pred.scores))
    for i in order:
        choices=np.flatnonzero(available&(gt_labels==pred.labels[i]))
        if not len(choices):continue
        j=choices[np.argmax(overlaps[i,choices])]
        if overlaps[i,j]>iou:tp[i]=True;quality[i]=overlaps[i,j];available[j]=False
    n=int(tp.sum());ng=len(gt_boxes);nd=len(order)
    return dict(order=order,tp=tp,quality=quality,count_tp=n,count_fp=nd-n,count_fn=ng-n,
        loss=float((ng+nd-2*n+(1-quality[tp]).sum())/max(1,ng+nd)))

def summarize(frames,score_threshold=.1,device='cuda'):
    """Frames are (Detections, GT boxes, GT labels), already in the fixed ROI."""
    events={.25:{c:[] for c in range(7)},.5:{c:[] for c in range(7)}};counts=np.zeros(7,int)
    totals=np.zeros(3,int);loss=[]
    for pred,gt,labels in frames:
        pred=pred.roi();overlaps=overlap_matrix(pred.boxes,gt,device)
        counts+=np.bincount(labels,minlength=7)
        for iou in events:
            m=match(pred,gt,labels,iou=iou,overlaps=overlaps)
            for c in range(7):
                ix=m['order'][pred.labels[m['order']]==c]
                events[iou][c].extend(zip(pred.scores[ix].tolist(),m['tp'][ix].tolist()))
        m=match(pred,gt,labels,threshold=score_threshold,overlaps=overlaps)
        totals+=[m['count_tp'],m['count_fp'],m['count_fn']];loss.append(m['loss'])
    result={'frames':len(frames),'score_threshold':score_threshold,'class_gt_counts':dict(zip(CLASSES,counts.tolist()))}
    for iou in events:
        aps={}
        for c in range(7):
            if counts[c]==0:continue
            rows=sorted(events[iou][c],key=lambda r:-r[0]);tp=np.cumsum([r[1] for r in rows]);fp=np.cumsum([not r[1] for r in rows])
            recall=tp/counts[c];precision=tp/np.maximum(1,tp+fp)
            rr=np.r_[0.,recall,1.];pp=np.r_[0.,precision,0.];pp=np.maximum.accumulate(pp[::-1])[::-1]
            ix=np.flatnonzero(rr[1:]!=rr[:-1]);aps[CLASSES[c]]=float(((rr[ix+1]-rr[ix])*pp[ix+1]).sum())
        result['AP'+str(int(iou*100))]=float(np.mean(list(aps.values()))) if aps else 0.
        result['class_AP'+str(int(iou*100))]=aps
    tp,fp,fn=map(int,totals)
    result.update(tp25=tp,fp25=fp,fn25=fn,recall25=tp/max(1,tp+fn),precision25=tp/max(1,tp+fp),
        f1=2*tp/max(1,2*tp+fp+fn),fp_per_frame=fp/max(1,len(frames)),mean_loss=float(np.mean(loss)) if loss else None)
    return result

def choose_score_threshold(frames,device='cuda'):
    counts={x:np.zeros(3,int) for x in [.05,.1,.2,.3,.5]}
    for pred,gt,labels in frames:
        pred=pred.roi();overlaps=overlap_matrix(pred.boxes,gt,device)
        for threshold in counts:
            m=match(pred,gt,labels,threshold=threshold,overlaps=overlaps)
            counts[threshold]+=[m['count_tp'],m['count_fp'],m['count_fn']]
    f1={k:float(2*v[0]/max(1,2*v[0]+v[1]+v[2])) for k,v in counts.items()}
    best=min(f1,key=lambda k:(-f1[k],k))
    return best,f1
