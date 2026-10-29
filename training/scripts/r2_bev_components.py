"""Explicit reconstruction for NEW experiments, not recovered historical code.

Maps use [batch,channel,y,x]. Alignment is differentiable inverse sampling.
TempoSense uses three real/predicted maps and 10x10-window self-attention to
bound memory. This implementation and its settings must accompany new scores.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_RANGE=(-51.2,-51.2,-5.,51.2,51.2,3.)

def warp_bev(source,source_to_target,pc_range=DEFAULT_RANGE,target_size=None):
    if source.ndim!=4:raise ValueError('Expected B,C,H,W feature maps')
    batch,_,height,width=source.shape
    th,tw=target_size or (height,width)
    if source_to_target.shape!=(batch,4,4):raise ValueError('Expected B,4,4 transforms')
    # Projection is evaluated on the target z=0 plane; keep matrix precision.
    dtype=torch.float32;device=source.device
    y=pc_range[1]+(torch.arange(th,device=device,dtype=dtype)+.5)*(pc_range[4]-pc_range[1])/th
    x=pc_range[0]+(torch.arange(tw,device=device,dtype=dtype)+.5)*(pc_range[3]-pc_range[0])/tw
    yy,xx=torch.meshgrid(y,x)
    grid=torch.stack((xx,yy,torch.zeros_like(xx),torch.ones_like(xx)),-1).reshape(1,-1,4).expand(batch,-1,-1)
    inverse=torch.inverse(source_to_target.float())
    points=torch.bmm(grid,inverse.transpose(1,2))
    nx=2*(points[...,0]-pc_range[0])/(pc_range[3]-pc_range[0])-1
    ny=2*(points[...,1]-pc_range[1])/(pc_range[4]-pc_range[1])-1
    sample_grid=torch.stack((nx,ny),-1).reshape(batch,th,tw,2)
    values=F.grid_sample(source.float(),sample_grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    coverage=F.grid_sample(torch.ones((batch,1,height,width),device=device,dtype=dtype),sample_grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    return values.to(source.dtype),coverage

class WindowSelfAttention(nn.Module):
    def __init__(self,channels=256,heads=8,window=10):
        super().__init__();self.window=window
        self.norm1=nn.LayerNorm(channels);self.attention=nn.MultiheadAttention(channels,heads,dropout=0.)
        self.norm2=nn.LayerNorm(channels)
        self.ffn=nn.Sequential(nn.Linear(channels,channels*2),nn.GELU(),nn.Linear(channels*2,channels))
    def forward(self,x):
        b,c,h,w=x.shape;k=self.window
        if h%k or w%k:raise ValueError('BEV dimensions must be divisible by attention window')
        tokens=x.reshape(b,c,h//k,k,w//k,k).permute(3,5,0,2,4,1).reshape(k*k,b*(h//k)*(w//k),c)
        q=self.norm1(tokens);a,_=self.attention(q,q,q,need_weights=False)
        tokens=tokens+a;tokens=tokens+self.ffn(self.norm2(tokens))
        return tokens.reshape(k,k,b,h//k,w//k,c).permute(2,5,3,0,4,1).reshape(b,c,h,w)

class TempoSense(nn.Module):
    def __init__(self,channels=256,window=10):
        super().__init__();groups=8 if channels%8==0 else 1
        self.conv=nn.Sequential(nn.Conv2d(3*channels,channels,3,padding=1),nn.GroupNorm(groups,channels),nn.ReLU(),nn.Conv2d(channels,channels,3,padding=1))
        self.attention=WindowSelfAttention(channels,8 if channels%8==0 else 1,window)
        self.output=nn.Conv2d(channels,channels,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,history):
        if history.ndim!=5 or history.shape[1]!=3:raise ValueError('Expected B,3,C,H,W history')
        b,_,c,h,w=history.shape
        return history[:,-1]+self.output(self.attention(self.conv(history.reshape(b,3*c,h,w))))
    def predict_at(self,history,observation_times_us,target_time_us,max_steps=20):
        times=list(observation_times_us)
        if len(times)!=3 or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Need three increasing real timestamps')
        cadence=(times[2]-times[0])/2
        steps=(target_time_us-times[-1])/cadence
        if not 0<steps<=max_steps:raise ValueError('Unsupported prediction horizon')
        whole=int(math.floor(steps));fraction=steps-whole;buffer=history
        prev=history[:,-1]
        for index in range(int(math.ceil(steps))):
            current=self(buffer)
            if index==whole:return prev+(current-prev)*fraction
            prev=current;buffer=torch.cat((buffer[:,1:],current[:,None]),dim=1)
        return prev

class AdaptiveBEVFusion(nn.Module):
    """Registered two-source utility fusion with explicit age/prediction inputs.

Descriptors are source mean, std, absolute mean, RMS, planar visibility,
confidence, prediction flag, age, plus a learned registered-source embedding.
This is trained before selector fitting and then held fixed for A0--A6.
"""
    def __init__(self,channels=256,source_embedding_dim=8):
        super().__init__();groups=8 if channels%8==0 else 1
        self.source_embedding=nn.Embedding(2,source_embedding_dim)
        self.source_projection=nn.Conv2d(channels,channels,1)
        self.utility=nn.Sequential(nn.Linear(8+source_embedding_dim,32),nn.ReLU(),nn.Linear(32,1))
        self.enhance=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(groups,channels),nn.ReLU())
        self.output=nn.Conv2d(channels,channels,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.age_penalty=nn.Parameter(torch.tensor(-2.));self.prediction_penalty=nn.Parameter(torch.tensor(-2.))
    def forward(self,ego,source,source_to_ego,age_seconds,predicted,source_confidence=None,ego_confidence=None):
        ego=ego.contiguous();source=source.contiguous()
        aligned,visibility=warp_bev(self.source_projection(source),source_to_ego)
        b=ego.shape[0];zero=ego.new_zeros(b)
        age=age_seconds.to(ego).reshape(b);predicted=predicted.to(ego).reshape(b)
        confidences=[ego_confidence,source_confidence];maps=[ego,aligned];scores=[]
        for index,value in enumerate(maps):
            flat=value.flatten(1);confidence=confidences[index]
            confidence=zero if confidence is None else confidence.to(ego).reshape(b)
            descriptor=torch.stack((flat.mean(1),flat.std(1,unbiased=False),flat.abs().mean(1),flat.square().mean(1).sqrt(),
                                    torch.ones_like(zero) if index==0 else visibility.flatten(1).mean(1),confidence,
                                    zero if index==0 else predicted,zero if index==0 else age),1)
            embedding=self.source_embedding.weight[index].expand(b,-1)
            utility=self.utility(torch.cat((descriptor,embedding),1)).squeeze(-1)
            if index:utility=utility-F.softplus(self.age_penalty)*age-F.softplus(self.prediction_penalty)*predicted
            scores.append(utility)
        weight=torch.softmax(torch.stack(scores,1),1)
        mixed=weight[:,0,None,None,None]*self.enhance(ego)+weight[:,1,None,None,None]*self.enhance(aligned)
        return ego+self.output(mixed),weight
