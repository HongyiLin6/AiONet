"""Signed-loss control fitted on the fitting split, selected on development."""
import copy
import numpy as np
from .selector import scenario_weights
from .reliability_core import DescriptorNormalizer

def fit_mlp(x,y,scenes,xd,yd,dev_scenes,seed=17):
    import torch
    from torch import nn
    torch.manual_seed(seed);norm=DescriptorNormalizer().fit(x);x=norm.transform(x);xd=norm.transform(xd)
    net=nn.Sequential(nn.Linear(20,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1),nn.Tanh())
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4);objective=nn.HuberLoss(reduction='none',delta=.1)
    weights=scenario_weights(scenes);dw=torch.from_numpy(scenario_weights(dev_scenes));rng=np.random.default_rng(seed)
    best=float('inf');best_state=None;stale=0;history=[]
    for epoch in range(100):
        net.train();order=rng.permutation(len(x))
        for begin in range(0,len(x),256):
            ix=order[begin:begin+256];opt.zero_grad()
            loss=(objective(net(torch.from_numpy(x[ix])).flatten(),torch.from_numpy(y[ix]))*torch.from_numpy(weights[ix])).mean()
            loss.backward();opt.step()
        net.eval()
        with torch.no_grad():dev=float((objective(net(torch.from_numpy(xd)).flatten(),torch.from_numpy(yd))*dw).mean())
        history.append(dict(epoch=epoch+1,development_huber=dev))
        if dev<best:best=dev;best_state=copy.deepcopy(net.state_dict());stale=0
        else:stale+=1
        if stale>=10:break
    net.load_state_dict(best_state)
    model=dict(kind='mlp',dimensions=20,without_age_horizon=False,normalizer=norm.state_dict(),seed=seed,
        layers=[dict(weight=net[i].weight.detach().numpy(),bias=net[i].bias.detach().numpy()) for i in (0,2,4)])
    return model,dict(regressor='mlp',development_huber=best,history=history)
