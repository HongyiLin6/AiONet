"""Car AP, official nuScenes-style center mAP, and explicit IoU-NDS.

Composite evaluation requires real velocity and attribute fields. Metric values
are computed from supplied detections; manuscript table values are not inputs.
"""
import numpy as np
from aionet.structured_fusion import Detections
from aionet.evaluation_metrics import summarize
from aionet.reliability_core import iou3d

def evaluate_car(frames,compute_nds=False,device='cpu'):
    """Each frame has pred/gt boxes and optional pred/gt velocity/attributes."""
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.algo import accumulate,calc_ap,calc_tp
    from nuscenes.eval.common.utils import center_distance
    from pyquaternion import Quaternion
    predictions=EvalBoxes();truth=EvalBoxes();iou_frames=[]
    for i,frame in enumerate(frames):
        token=str(frame.get('token',i));pred=frame['pred'];gt=np.asarray(frame['gt'],float).reshape(-1,7)
        iou_frames.append((pred,gt,np.zeros(len(gt),dtype=int)))
        for side,boxes,target in [('pred',pred.boxes,predictions),('gt',gt,truth)]:
            velocity=frame.get(side+'_velocity')
            if side=='pred' and velocity is None:velocity=pred.velocity
            attrs=frame.get(side+'_attributes')
            if compute_nds:
                if velocity is None or np.shape(velocity)!=(len(boxes),2) or not np.isfinite(velocity).all():
                    raise ValueError('NDS requires finite '+side+' velocities for every evaluated object')
                if attrs is None or len(attrs)!=len(boxes) or any(not str(a) for a in attrs):
                    raise ValueError('NDS requires explicit '+side+' attribute labels; supply the evaluation annotation protocol')
            velocity=np.asarray(velocity if velocity is not None else np.zeros((len(boxes),2)))
            attrs=attrs if attrs is not None else ['']*len(boxes)
            converted=[]
            for j,b in enumerate(boxes):
                converted.append(DetectionBox(sample_token=token,translation=tuple(b[:3]),size=(float(b[4]),float(b[3]),float(b[5])),
                    rotation=tuple(Quaternion(axis=[0,0,1],radians=float(b[6])).elements),velocity=tuple(velocity[j]),
                    detection_name='car',detection_score=float(pred.scores[j]) if side=='pred' else -1.,attribute_name=str(attrs[j])))
            target.add_boxes(token,converted)
    allpoint=summarize(iou_frames,device=device)
    result={'AP25':allpoint['AP25'],'AP50':allpoint['AP50'],'protocol':'Car; official nuScenes AP interpolation for center mAP; all-point AP25/AP50'}
    curves={d:accumulate(truth,predictions,'car',center_distance,d) for d in (.5,1.,2.,4.)}
    result['mAP']=float(np.mean([calc_ap(v,.1,.1) for v in curves.values()]))
    if compute_nds:
        names={'mATE':'trans_err','mASE':'scale_err','mAOE':'orient_err','mAVE':'vel_err','mAAE':'attr_err'}
        errors={name:calc_tp(curves[2.],.1,key) for name,key in names.items()}
        result.update(errors);result['NDS']=(5*result['mAP']+sum(max(0.,1-v) for v in errors.values()))/10
        def overlap_distance(a,b):
            def unpack(box):
                yaw=Quaternion(box.rotation).yaw_pitch_roll[0]
                return [*box.translation,box.size[1],box.size[0],box.size[2],yaw]
            return 1.-iou3d(np.asarray(unpack(a)),np.asarray(unpack(b)))
        for threshold in (.7,.8):
            curve=accumulate(truth,predictions,'car',overlap_distance,1-threshold)
            ap=calc_ap(curve,.1,.1);err={name:calc_tp(curve,.1,key) for name,key in names.items()}
            result['NDS'+str(int(threshold*100))]=(5*ap+sum(max(0.,1-v) for v in err.values()))/10
            result['IoU_'+str(threshold)]=dict(AP=ap,errors=err)
    return result
