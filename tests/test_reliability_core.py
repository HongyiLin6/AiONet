"""Contract tests with synthetic geometry, not experimental performance evidence."""
import math
import unittest
import numpy as np
from aionet.reliability_core import (iou3d,detection_loss,calibration_margin,assert_disjoint_scenarios,
                              Candidate,choose,commit,DescriptorNormalizer,delay_trace)

class GeometryTests(unittest.TestCase):
    def test_geometry(self):
        a=[0,0,0,2,2,2,0]
        self.assertAlmostEqual(iou3d(a,a),1)
        self.assertAlmostEqual(iou3d(a,[1,0,0,2,2,2,0]),1/3)
        self.assertEqual(iou3d(a,[0,0,3,2,2,2,0]),0)
        self.assertAlmostEqual(iou3d([0,0,0,4,2,2,0],[0,0,0,4,2,2,math.pi/2]),1/3)
    def test_loss_empty_and_duplicates(self):
        b=[0,0,0,2,2,2,0]
        self.assertEqual(detection_loss([],[],[],[],[],.1)['loss'],0)
        self.assertEqual(detection_loss([],[],[],[b],[0],.1)['loss'],1)
        r=detection_loss([b,b],[.9,.8],[0,0],[b],[0],.1)
        self.assertEqual(r['tp'],1); self.assertEqual(r['fp'],1)
        self.assertAlmostEqual(r['loss'],1/3)
        self.assertEqual(detection_loss([b],[.9],[1],[b],[0],.1)['loss'],1)
    def test_strict_iou_threshold(self):
        # Intersection volume 2, union volume 8 -> IoU exactly 0.25.
        a=[0,0,0,5,1,1,0]; b=[3,0,0,5,1,1,0]
        self.assertAlmostEqual(iou3d(a,b),.25)
        self.assertEqual(detection_loss([a],[.9],[0],[b],[0],.1)['tp'],0)

class CalibrationTests(unittest.TestCase):
    def test_pack_maximum_and_minimum_count(self):
        self.assertTrue(math.isinf(calibration_margin({str(i):[0.] for i in range(8)})['q']))
        packs={str(i):[-.5,i/100.] for i in range(9)}
        self.assertAlmostEqual(calibration_margin(packs)['q'],.08)
        self.assertEqual(calibration_margin({str(i):[-.2] for i in range(9)})['q'],0)
        self.assertEqual(calibration_margin({str(i):[] for i in range(9)})['q'],0)
    def test_leakage_rejected(self):
        with self.assertRaises(ValueError): assert_disjoint_scenarios({'fit':['0001'],'cal':['0001']})
    def test_normalization_and_trace_pairing(self):
        x=np.zeros((3,20)); x[:,15]=1; x[:,18]=1; x[:,0]=[1,2,3]
        norm=DescriptorNormalizer().fit(x); before=norm.state_dict()
        extreme=x.copy(); extreme[:,0]=100
        self.assertTrue((norm.transform(extreme)[:,0]==5).all())
        self.assertEqual(before,norm.state_dict())
        np.testing.assert_allclose(delay_trace(100,17,258)/delay_trace(100,17,187.3),258/187.3)

class DecisionTests(unittest.TestCase):
    def candidate(self,**kw):
        args=dict(action='O',representation='F',observation_time=.7,represented_time=.7,
                  arrival_time=.9,usable_time=.9,remaining_cost=.02,score=-.3)
        args.update(kw); return Candidate(**args)
    def test_causality_and_deadline(self):
        c=self.candidate()
        self.assertEqual(choose([c],1.,1.04,1.1,.1),'O')
        self.assertEqual(choose([c],1.,1.09,1.1,.1),'0')
        self.assertEqual(choose([self.candidate(arrival_time=1.05,usable_time=1.05)],1.,1.04,1.1,.1),'0')
        self.assertEqual(choose([self.candidate(observation_time=1.01,represented_time=1.01)],1.,1.04,1.1,0),'0')
        self.assertEqual(choose([c],1.,1.04,1.1,math.inf),'0')
    def test_ties_threshold_and_fallback(self):
        o=self.candidate(); p=self.candidate(action='P',represented_time=1.)
        self.assertEqual(choose([p,o],1.,1.04,1.1,0),'O')
        self.assertEqual(choose([o],1.,1.04,1.1,.3),'0')
        self.assertEqual(commit('P',1.11,1.03,1.1),'0')
        self.assertIsNone(commit('P',1.11,1.12,1.1))
        self.assertEqual(commit('P',1.09,1.03,1.1),'P')

if __name__=='__main__': unittest.main()
