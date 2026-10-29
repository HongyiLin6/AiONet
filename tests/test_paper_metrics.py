import pytest,numpy as np
pytest.importorskip('nuscenes')
from aionet.paper.metrics import evaluate_car
from aionet.structured_fusion import Detections

def example():
    b=np.array([[10.,0,0,4,2,2,0]])
    return dict(pred=Detections(b,[.9],[0],[[0,0]]),gt=b,gt_velocity=np.array([[0.,0.]]),pred_attributes=['vehicle.parked'],gt_attributes=['vehicle.parked'])

def test_perfect_detection_has_perfect_center_and_iou_scores():
    metrics=evaluate_car([example()],compute_nds=True)
    for key in ['AP25','AP50','mAP','NDS','NDS70','NDS80']:assert metrics[key]==pytest.approx(1.)

def test_nds_requires_attribute_evidence():
    row=example();row.pop('gt_attributes')
    with pytest.raises(ValueError,match='attribute'):evaluate_car([row],compute_nds=True)
