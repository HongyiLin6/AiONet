"""Verify the fitted control's normalization and portable inference on synthetic data."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from aionet.mlp import fit_mlp
from aionet.selector import score
from aionet.reliability_core import DescriptorNormalizer


def test_mlp_portable_scoring_uses_only_fitting_normalization():
    rng=np.random.default_rng(17);x=np.zeros((48,20),np.float32)
    x[:,:12]=rng.normal(size=(48,12));x[:,15]=1;x[:,18]=1
    dev=x[:16].copy();dev[:,:12]+=3
    y=(.1*np.tanh(x[:,0])).astype(np.float32);yd=(.1*np.tanh(dev[:,0])).astype(np.float32)
    model,choice=fit_mlp(x,y,np.array(['fit1']*24+['fit2']*24),dev,yd,np.array(['dev']*16))
    np.testing.assert_allclose(model['normalizer']['mean'],x[:,:12].mean(0),atol=1e-7)
    norm=DescriptorNormalizer()
    for k,v in model['normalizer'].items():setattr(norm,k,np.array(v))
    value=torch.from_numpy(norm.transform(dev))
    for i,layer in enumerate(model['layers']):
        value=value@torch.from_numpy(layer['weight']).T+torch.from_numpy(layer['bias'])
        value=torch.relu(value) if i<2 else torch.tanh(value)
    np.testing.assert_allclose(score(model,dev),-value.numpy().ravel(),atol=1e-7)
    assert np.isfinite(choice['development_huber'])
