"""Registered spatial adapters and seven explicit cooperative fusion operators.

Query splatting and windowed attention are reconstruction choices. Parameters
are learned separately for each registered pair; no tensor is a paper result.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from aionet.r2_bev_components import warp_bev,AdaptiveBEVFusion

PC_RANGE=(-51.2,-51.2,-5.,51.2,51.2,3.)

def transform_points(x,matrix):
    return torch.einsum('bij,bnj->bni',matrix[:,:3,:3],x)+matrix[:,None,:3,3]

class SpatialAdapter(nn.Module):
    def __init__(self,channels=256,height=150,width=150,pc_range=PC_RANGE):
        super().__init__();self.height=height;self.width=width;self.pc_range=pc_range
        self.query_projection=nn.Linear(channels,channels)
        self.native_projection=nn.Conv2d(channels,channels,1)
        groups=math.gcd(channels,8)
        self.refine=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(groups,channels),nn.ReLU(),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,packet):
        if 'bev' in packet:return self.native_projection(packet['bev'])
        q=self.query_projection(packet['query']);xyz=packet['xyz'];b,n,c=q.shape;h,w=self.height,self.width
        x=(xyz[...,0]-self.pc_range[0])/(self.pc_range[3]-self.pc_range[0])*w-.5
        y=(xyz[...,1]-self.pc_range[1])/(self.pc_range[4]-self.pc_range[1])*h-.5
        valid=packet.get('valid',torch.ones((b,n),dtype=torch.bool,device=q.device)) & torch.isfinite(x)&torch.isfinite(y)
        x=torch.nan_to_num(x);y=torch.nan_to_num(y);x0=x.floor();y0=y.floor()
        grid=q.new_zeros((b,c,h*w));mass=q.new_zeros((b,1,h*w))
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            xx=x0+dx;yy=y0+dy
            weight=(1-(x-xx).abs())*(1-(y-yy).abs())
            weight=weight*valid*(xx>=0)*(xx<w)*(yy>=0)*(yy<h)
            index=(yy.clamp(0,h-1)*w+xx.clamp(0,w-1)).long()
            grid.scatter_add_(2,index[:,None].expand(-1,c,-1),q.transpose(1,2)*weight[:,None])
            mass.scatter_add_(2,index[:,None],weight[:,None])
        return self.refine((grid/mass.clamp_min(1)).reshape(b,c,h,w))

class QueryCrossAttention(nn.Module):
    def __init__(self,channels=256,heads=8):
        super().__init__();self.attention=nn.MultiheadAttention(channels,heads)
        self.norm=nn.LayerNorm(channels);self.position=nn.Sequential(nn.Linear(3,channels),nn.ReLU(),nn.Linear(channels,channels))
    def forward(self,q,memory,position=None,padding=None):
        if position is not None:q=q+self.position(position/51.2)
        value,_=self.attention(q.transpose(0,1),memory.transpose(0,1),memory.transpose(0,1),key_padding_mask=padding,need_weights=False)
        return self.norm(q+value.transpose(0,1))

class PaperFusion(nn.Module):
    def __init__(self,operator,channels=256,height=150,width=150,heads=8,window=10):
        super().__init__();self.operator=operator;self.height=height;self.width=width;self.window=window
        self.ego_adapter=SpatialAdapter(channels,height,width);self.source_adapter=SpatialAdapter(channels,height,width)
        self.ego_query=nn.Linear(channels,channels);self.source_query=nn.Linear(channels,channels)
        self.position=nn.Linear(3,channels);self.cross=QueryCrossAttention(channels,heads)
        self.concat=nn.Conv2d(2*channels,channels,1)
        self.conv=nn.Sequential(nn.Conv2d(2*channels,channels,3,padding=1),nn.ReLU(),nn.Conv2d(channels,channels,3,padding=1))
        self.box_mlp=nn.Sequential(nn.Linear(11,channels),nn.ReLU(),nn.Linear(channels,channels))
        self.adaptive=AdaptiveBEVFusion(channels)
    def query_packet(self,packet,source=False,matrix=None):
        q=(self.source_query if source else self.ego_query)(packet['query'])
        xyz=transform_points(packet['xyz'],matrix) if source else packet['xyz']
        return q+self.position(xyz/51.2),xyz
    def window_attention(self,ego,source):
        b,c,h,w=ego.shape;k=self.window;ph=(-h)%k;pw=(-w)%k
        def windows(x):
            x=F.pad(x,(0,pw,0,ph));hh,ww=x.shape[-2:]
            return x.reshape(b,c,hh//k,k,ww//k,k).permute(0,2,4,3,5,1).reshape(-1,k*k,c)
        out=self.cross(windows(ego),windows(source))
        return out.reshape(b,(h+ph)//k,(w+pw)//k,k,k,c).permute(0,5,1,3,2,4).reshape(b,c,h+ph,w+pw)[...,:h,:w]
    def forward(self,ego,source,matrix,age,predicted=None):
        op=self.operator;predicted=torch.zeros_like(age) if predicted is None else predicted
        if op=='query_concat':
            eq,ex=self.query_packet(ego);sq,sx=self.query_packet(source,True,matrix)
            return {'query':torch.cat((eq,sq),1),'xyz':torch.cat((ex,sx),1)}
        if op=='box_mlp':
            items=[];locations=[]
            for packet,remote in [(ego,False),(source,True)]:
                boxes=packet['boxes'];xyz=transform_points(boxes[...,:3],matrix) if remote else boxes[...,:3]
                velocity=packet['velocity'];yaw=boxes[...,6]
                if remote:
                    velocity=torch.einsum('bij,bnj->bni',matrix[:,:2,:2],velocity)
                    yaw=yaw+torch.atan2(matrix[:,1,0],matrix[:,0,0])[:,None]
                encoded=torch.cat((xyz/51.2,boxes[...,3:6]/10.,yaw.sin()[...,None],yaw.cos()[...,None],velocity/20.,packet['scores'][...,None]),-1)
                items.append(self.box_mlp(encoded));locations.append(xyz)
            return {'query':torch.cat(items,1),'xyz':torch.cat(locations,1)}
        raw=self.source_adapter(source);aligned,_=warp_bev(raw,matrix,PC_RANGE)
        if op=='bev_query':
            eq,ex=self.query_packet(ego);memory=F.adaptive_avg_pool2d(aligned,(min(30,self.height),min(30,self.width))).flatten(2).transpose(1,2)
            return {'query':self.cross(eq,memory,ex),'xyz':ex}
        local=self.ego_adapter(ego)
        if op=='bev_concat':return {'bev':self.concat(torch.cat((local,aligned),1))}
        if op=='bev_conv':return {'bev':self.conv(torch.cat((local,aligned),1))}
        if op=='bev_attention':return {'bev':self.window_attention(local,aligned)}
        if op=='adaptive':
            fused,_=self.adaptive(local,raw,matrix,age,predicted)
            eq,ex=self.query_packet(ego);sq,_=self.query_packet(source,True,matrix)
            memory=F.adaptive_avg_pool2d(fused,(min(30,self.height),min(30,self.width))).flatten(2).transpose(1,2)
            return {'query':self.cross(eq,torch.cat((memory,sq),1),ex),'xyz':ex}
        raise ValueError('Unknown fusion operator: '+op)
