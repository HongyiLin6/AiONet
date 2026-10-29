"""Two calibrated images in the current vehicle LiDAR coordinate frame."""
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
import mmcv
from experiment_io import records,gt_roi,source_to_ego

def camera_matrices(record):
    c=record['cams']['CAM_FRONT'];e=np.eye(4);e[:3,:3]=c['lidar2cam_r'];e[:3,3]=c['lidar2cam_t']
    k=np.eye(4);k[:3,:3]=c['cam_intrinsic'];return k,e

def image_pair(vehicle,source,cooperative,fixture=None):
    images=[]
    for record in [vehicle,source]:
        img=mmcv.imread(fixture or record['cams']['CAM_FRONT']['data_path']);img=mmcv.imresize(img,(960,540))
        img=mmcv.imnormalize(img.astype(np.float32),np.array([123.675,116.28,103.53],np.float32),np.array([58.395,57.12,57.375],np.float32),to_rgb=True)
        # EMIFF's four-scale upsampling needs feature height divisible by eight.
        img=mmcv.impad(img,shape=(544,960));images.append(torch.from_numpy(img.transpose(2,0,1).copy()))
    kv,ev=camera_matrices(vehicle);ki,ei=camera_matrices(source)
    source_pose=dict(lidar2global=source['lidar2global'])
    inf2veh=source_to_ego(source_pose,cooperative)
    # Intrinsics stay in original-image coordinates; point_sample and CCM apply
    # the same resize factor once, as in the official EMIFF pipeline.
    meta=dict(lidar2img=dict(intrinsic=[kv,ki],extrinsic=[ev,ei@np.linalg.inv(inf2veh)]),
        scale_factor=np.array([.5,.5,.5,.5],np.float32),img_shape=(540,960,3),ori_shape=(1080,1920,3),
        pad_shape=(544,960,3),flip=False,transformation_3d_flow=[],box_mode_3d=0)
    return torch.stack(images),meta

class EMIFFPairs(Dataset):
    def __init__(self,partition,fixture=None):
        self.partition=partition;self.rows=records('cooperative',partition);self.fixture=fixture
        self.vehicle={r['token']:r for r in records('vehicle',partition)};self.source={r['token']:r for r in records('infrastructure',partition)}
        self.scenes={}
        for r in self.source.values():self.scenes.setdefault(r['sequence_id'],[]).append(r)
        for v in self.scenes.values():v.sort(key=lambda r:r['timestamp'])
        # No detector input at a time later than the vehicle image.
        self.rows=[r for r in self.rows if self.scenes[r['sequence_id']][0]['timestamp']<=r['timestamp']]
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];scene=self.scenes[r['sequence_id']];times=[x['timestamp'] for x in scene]
        latest=bisect.bisect_right(times,r['timestamp'])-1
        lag=np.random.randint(0,6) if self.partition=='perception_train' else 2
        source=scene[max(0,latest-lag)]
        image,meta=image_pair(self.vehicle[r['vehicle_token']],source,r,self.fixture)
        boxes,labels=gt_roi(r)
        return dict(image=image,meta=meta,boxes=boxes,labels=labels,index=index)

def collate(rows):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    for r in rows:r['meta']['box_type_3d']=LiDARInstance3DBoxes
    return dict(image=torch.stack([r['image'] for r in rows]),meta=[r['meta'] for r in rows],
        boxes=[r['boxes'] for r in rows],labels=[r['labels'] for r in rows],index=[r['index'] for r in rows])
