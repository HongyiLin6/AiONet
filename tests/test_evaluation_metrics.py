import unittest
import numpy as np
from aionet.structured_fusion import Detections
from aionet.evaluation_metrics import match,summarize,choose_score_threshold
from aionet.reliability_core import detection_loss

class MetricsTest(unittest.TestCase):
    def test_loss_matches_reference_with_ties_and_duplicates(self):
        b=np.array([[10,0,0,4,2,2,0],[10,0,0,4,2,2,0],[20,0,0,4,2,2,.1],[21,0,0,4,2,2,.1]],float)
        p=Detections(b,[.9,.9,.3,.1],[0,0,0,1]);gt=b[[0,2]];labels=np.array([0,0])
        a=match(p,gt,labels,threshold=.2,device='cpu')
        z=detection_loss(p.boxes,p.scores,p.labels,gt,labels,.2)
        self.assertAlmostEqual(a['loss'],z['loss']);self.assertEqual(a['count_tp'],z['tp'])
    def test_empty_output_and_perfect_detection(self):
        b=np.array([[10,0,0,4,2,2,.2]],np.float32);labels=np.array([0])
        p=Detections(b,[.9],labels);m=summarize([(p,b,labels)],device='cpu')
        self.assertEqual(m['AP25'],1.);self.assertEqual(m['AP50'],1.);self.assertEqual(m['mean_loss'],0.)
        e=Detections(np.empty((0,7)),[],[]);m=summarize([(e,b,labels)],device='cpu')
        self.assertEqual(m['AP25'],0.);self.assertEqual(m['mean_loss'],1.)
        threshold,_=choose_score_threshold([(p,b,labels)],device='cpu');self.assertEqual(threshold,.05)

if __name__=='__main__':unittest.main()
