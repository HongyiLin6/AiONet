"""Project actual predictions and ground truth onto a camera image."""
import argparse
from pathlib import Path
import numpy as np

def corners(box):
    x,y,z,length,width,height,yaw=box
    points=np.array([[i,j,k] for k in [-.5,.5] for i,j in [(-.5,-.5),(-.5,.5),(.5,.5),(.5,-.5)]])*[length,width,height]
    c,s=np.cos(yaw),np.sin(yaw);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return points@rotation.T+[x,y,z]

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--ground-truth',type=Path,required=True);p.add_argument('--projection',type=Path,required=True);p.add_argument('--threshold',type=float,default=.3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from PIL import Image,ImageDraw
    image=Image.open(a.image).convert('RGB');draw=ImageDraw.Draw(image);matrix=np.load(a.projection,allow_pickle=False)
    if matrix.shape not in [(3,4),(4,4)]:raise ValueError('Projection must map homogeneous LiDAR coordinates into camera pixels')
    matrix=matrix[:3]
    with np.load(a.predictions,allow_pickle=False) as z:pred=z['boxes'][z['scores']>=a.threshold]
    with np.load(a.ground_truth,allow_pickle=False) as z:gt=z['boxes']
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for boxes,color in [(gt,'#d77b70'),(pred,'#459b95')]:
        for box in boxes:
            points=corners(box);camera=np.c_[points,np.ones(8)]@matrix.T
            for i,j in edges:
                if min(camera[i,2],camera[j,2])<=.1:continue
                xy=camera[[i,j],:2]/camera[[i,j],2:3];draw.line(tuple(map(tuple,xy)),fill=color,width=2)
    draw.rectangle((5,5,280,34),fill='white');draw.text((12,12),'Ground truth: coral | Prediction: teal',fill='black')
    a.output.parent.mkdir(parents=True,exist_ok=True);image.save(a.output)

if __name__=='__main__':main()
