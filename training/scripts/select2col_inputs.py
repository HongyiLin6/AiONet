"""Causal ego history and GT-free confidence rasters for the camera adaptation."""
import bisect
import numpy as np
import torch
import cv2

def confidence_raster(detection):
    image=np.zeros((150,150),np.float32)
    for box,score in zip(detection.boxes,detection.scores):
        x,y,_,length,width,_,yaw=box;c,s=np.cos(yaw),np.sin(yaw)
        p=np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]])@np.array([[c,s],[-s,c]])+[x,y]
        pixels=np.rint((p+51.2)*(150/102.4)-.5).astype(np.int32)
        mask=np.zeros_like(image);cv2.fillConvexPoly(mask,pixels,float(score));image=np.maximum(image,mask)
    return image[None]

def context(cache,source,ego_row,source_row):
    scene=cache.by_scene[ego_row['scenario']];times=[r['observation_us'] for r in scene];position=bisect.bisect_left(times,ego_row['observation_us'])
    if position<2:raise ValueError('Two prior real ego observations required')
    history=scene[position-2:position];inverse=np.linalg.inv(np.asarray(ego_row['lidar2global']))
    transform=np.stack([inverse@np.asarray(r['lidar2global']) for r in history]).astype(np.float32)
    confidence=[confidence_raster(cache.detections(r)) for r in [ego_row]+history]
    confidence.append(confidence_raster(source.detections(source_row)))
    ages=[0.]+[(ego_row['observation_us']-r['observation_us'])/1e6 for r in history]+[(ego_row['observation_us']-source_row['observation_us'])/1e6]
    return dict(history=torch.from_numpy(np.stack([cache.feature(r) for r in history])),
        history_to_ego=torch.from_numpy(transform),confidence=torch.from_numpy(np.stack(confidence)),ages=torch.tensor(ages))
