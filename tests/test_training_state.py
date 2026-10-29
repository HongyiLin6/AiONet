"""A resumed synthetic epoch must match an uninterrupted one."""
import copy
import pytest,numpy as np
torch=pytest.importorskip('torch')
from aionet.paper.training_state import seed_epoch,save_checkpoint


def test_checkpoint_and_epoch_seed_preserve_resumed_updates(tmp_path):
    seed_epoch(17,0);model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.25),torch.nn.Linear(4,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.arange(18,dtype=torch.float32).reshape(6,3)/10
    def epoch(number):
        seed_epoch(17,number)
        for i in np.random.permutation(len(x)):
            optimizer.zero_grad();loss=(model(x[i])-x[i].sum()).square().mean();loss.backward();optimizer.step()
    epoch(0);path=tmp_path/'epoch.pt';save_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
    epoch(1);expected=copy.deepcopy(model.state_dict())
    saved=torch.load(path,map_location='cpu');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);epoch(1)
    for k,v in model.state_dict().items():torch.testing.assert_allclose(v,expected[k],rtol=0,atol=0)
    assert not path.with_suffix('.pt.writing').exists()
