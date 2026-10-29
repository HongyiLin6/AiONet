import unittest
import numpy as np
from aionet.structured_fusion import Detections,map_to_ego,fuse_structured
from aionet.feature_bank import FeatureBank,Observation
from aionet.descriptors import make_descriptor

class InterfaceTests(unittest.TestCase):
    def test_transform_heading_velocity_and_no_input_mutation(self):
        d=Detections([[10,0,0,4,2,2,0]],[.9],[0],[[2,0]])
        m=np.array([[0,-1,0,5],[1,0,0,0],[0,0,1,0],[0,0,0,1]],float)
        out=map_to_ego(d,m)
        np.testing.assert_allclose(out.boxes[0,:3],[5,10,0]);self.assertAlmostEqual(out.boxes[0,6],np.pi/2,places=6)
        np.testing.assert_allclose(out.velocity,[[0,2]]);np.testing.assert_allclose(d.boxes[0,:3],[10,0,0])
    def test_nms_is_classwise_and_uses_height(self):
        ego=Detections([[10,0,0,2,2,2,0]],[.8],[0])
        source=Detections([[10,0,0,2,2,2,0],[10,0,3,2,2,2,0],[10,0,0,2,2,2,0]],[.7,.6,.5],[0,0,1])
        out=fuse_structured(ego,source)
        self.assertEqual(len(out.boxes),3);self.assertEqual(list(out.labels),[0,0,1])
    def test_descriptor_preserves_observation_age_after_forecast(self):
        bank=FeatureBank();bank.reset('s')
        bank.advance(1050000,[Observation('p','s','i','F',700000,900000,900000,'cache/p','pose/p')])
        snapshot=bank.snapshot('i','F',1000000);ego=Detections([],[],[])
        observed=make_descriptor(snapshot,'O',ego,.1,1100000)
        predicted=make_descriptor(snapshot,'P',ego,.1,1100000)
        np.testing.assert_allclose(observed[:4],[.3,0,.3,.05])
        np.testing.assert_allclose(predicted[:4],[.3,.3,0,.05])
        self.assertEqual(list(predicted[12:15]),[1,1,1])

if __name__=='__main__':unittest.main()
