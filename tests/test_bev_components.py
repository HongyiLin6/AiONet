"""Geometry, batch isolation and immutable rollout checks, not model scores."""
import unittest
import pytest
torch=pytest.importorskip('torch')
from aionet.r2_bev_components import warp_bev,TempoSense,AdaptiveBEVFusion

class ComponentTests(unittest.TestCase):
    def test_identity_and_metric_translation(self):
        source=torch.zeros(1,1,10,10);source[0,0,4,4]=1
        transform=torch.eye(4)[None]
        result,coverage=warp_bev(source,transform,(-5,-5,-1,5,5,1))
        torch.testing.assert_allclose(result,source,rtol=0,atol=1e-6)
        transform[:,0,3]=1.
        shifted,_=warp_bev(source,transform,(-5,-5,-1,5,5,1))
        self.assertAlmostEqual(shifted[0,0,4,5].item(),1.,places=5)
        self.assertAlmostEqual(shifted[0,0,4,3].item(),0.,places=5)
        transform[:,0,3]=.5
        shifted,_=warp_bev(source,transform,(-5,-5,-1,5,5,1))
        self.assertAlmostEqual(shifted[0,0,4,4].item(),.5,places=5)
        self.assertAlmostEqual(shifted[0,0,4,5].item(),.5,places=5)
    def test_temposense_has_no_cross_batch_or_mutable_history(self):
        torch.manual_seed(11);model=TempoSense(8,5).eval()
        x=torch.randn(2,3,8,10,10);before=x.clone()
        joint=model(x);separate=torch.cat([model(x[i:i+1]) for i in range(2)])
        torch.testing.assert_allclose(joint,separate,rtol=1e-5,atol=1e-6)
        y1=model.predict_at(x,[0,100000,200000],450000)
        y2=model.predict_at(x,[0,100000,200000],450000)
        torch.testing.assert_allclose(y1,y2);torch.testing.assert_allclose(x,before)
        with self.assertRaises(ValueError):model.predict_at(x,[0,100000,200000],2300000)
    def test_alignment_and_fusion_are_differentiable(self):
        torch.manual_seed(12);model=AdaptiveBEVFusion(8)
        ego=torch.randn(2,8,10,10,requires_grad=True);source=torch.randn_like(ego,requires_grad=True)
        output,w=model(ego,source,torch.eye(4)[None].repeat(2,1,1),torch.tensor([.1,.3]),torch.tensor([0.,1.]))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(ego.grad).all() and torch.isfinite(source.grad).all())
        torch.testing.assert_allclose(w.sum(1),torch.ones(2))

if __name__=='__main__':unittest.main()
