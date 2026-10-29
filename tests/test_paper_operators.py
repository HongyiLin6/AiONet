import pytest
torch=pytest.importorskip('torch')
from aionet.paper.operators import PaperFusion,SpatialAdapter

def packet(batch=2,n=5,c=8):
    xyz=torch.rand(batch,n,3)*10
    return dict(query=torch.randn(batch,n,c),xyz=xyz,boxes=torch.cat((xyz,torch.ones(batch,n,3),torch.zeros(batch,n,1)),-1),scores=torch.ones(batch,n)*.7,velocity=torch.zeros(batch,n,2))

@pytest.mark.parametrize('operator',['query_concat','box_mlp','bev_concat','bev_attention','bev_query','bev_conv','adaptive'])
def test_registered_operators_have_finite_gradients_and_isolated_batches(operator):
    torch.manual_seed(17);model=PaperFusion(operator,channels=8,height=10,width=10,heads=2,window=5).eval()
    ego=packet();source=packet();matrix=torch.eye(4)[None].repeat(2,1,1);age=torch.tensor([.1,.2])
    output=model(ego,source,matrix,age);key='bev' if 'bev' in output else 'query'
    assert torch.isfinite(output[key]).all()
    separate=model({k:v[:1] for k,v in ego.items()},{k:v[:1] for k,v in source.items()},matrix[:1],age[:1])[key]
    torch.testing.assert_allclose(output[key][:1],separate,atol=1e-5,rtol=1e-5)
    output[key].square().mean().backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)

def test_spatial_projection_supports_native_and_query_interfaces():
    model=SpatialAdapter(8,10,10)
    assert model(packet()) .shape==(2,8,10,10)
    assert model({'bev':torch.randn(2,8,10,10)}).shape==(2,8,10,10)
