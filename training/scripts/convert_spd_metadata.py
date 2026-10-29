"""Build camera-only detector and cooperative records from verified SPD JSON.

Actual microsecond clocks are retained. Ground-truth velocity is an offline
finite-difference target, never an input descriptor. No image pixels are read.
Run on the training environment to keep numpy pickle versions consistent.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import zipfile
import numpy as np

CLASSES=('Car','Truck','Van','Bus','Motorcyclist','Cyclist','Pedestrian')
SIDES={'vehicle':'vehicle-side','infrastructure':'infrastructure-side'}

def transform(value, strict_rigid=True):
    value=value.get('transform',value)
    result=np.eye(4,dtype=np.float64)
    result[:3,:3]=np.asarray(value['rotation']).reshape(3,3)
    result[:3,3]=np.asarray(value['translation']).reshape(3)
    if not np.isfinite(result).all() or abs(np.linalg.det(result[:3,:3]))<1e-8:
        raise ValueError('Nonfinite or singular transform')
    if strict_rigid and not np.allclose(result[:3,:3].T@result[:3,:3],np.eye(3),atol=1e-3):
        raise ValueError('Invalid rigid transform')
    if strict_rigid and np.linalg.det(result[:3,:3])<.99:raise ValueError('Reflection/nonrigid rotation')
    return result

def quaternion(rotation):
    # Symmetric eigenproblem, returned in nuScenes/pyquaternion w,x,y,z order.
    r=rotation;xx,yy,zz=r[0,0],r[1,1],r[2,2]
    k=np.array([[xx-yy-zz,r[0,1]+r[1,0],r[0,2]+r[2,0],r[2,1]-r[1,2]],
                [r[0,1]+r[1,0],yy-xx-zz,r[1,2]+r[2,1],r[0,2]-r[2,0]],
                [r[0,2]+r[2,0],r[1,2]+r[2,1],zz-xx-yy,r[1,0]-r[0,1]],
                [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1],xx+yy+zz]])/3
    _,vec=np.linalg.eigh(k);q=vec[:,-1][[3,0,1,2]]
    return q if q[0]>=0 else -q

def annotation(anns):
    selected=[a for a in anns if a['type'] in CLASSES]
    boxes=np.asarray([[a['3d_location'][k] for k in ('x','y','z')]+[a['3d_dimensions'][k] for k in ('l','w','h')]+[a['rotation']] for a in selected],np.float32).reshape(-1,7)
    names=np.asarray([a['type'] for a in selected],dtype='<U16')
    return {'gt_boxes':boxes,'gt_names':names,'gt_num':len(selected),
            'class':np.asarray([CLASSES.index(a['type']) for a in selected],np.int64),
            'gt_track_ids':np.asarray([str(a['track_id']) for a in selected]),
            'valid_flag':np.ones(len(selected),dtype=bool),
            'gt_velocity':np.zeros((len(selected),2),np.float32),
            'velocity_valid':np.zeros(len(selected),dtype=bool)}

def add_velocity(records, clock='timestamp'):
    tracks=collections.defaultdict(list)
    for rec in records:
        mat=rec['lidar2global']
        for j,(box,tid) in enumerate(zip(rec['gt_boxes'],rec['gt_track_ids'])):
            tracks[(rec['scene_token'],str(tid))].append((rec[clock],rec,j,mat[:3,:3]@box[:3]+mat[:3,3]))
    for rows in tracks.values():
        rows.sort(key=lambda x:x[0])
        for j,(t,rec,bi,pos) in enumerate(rows):
            a=rows[max(0,j-1)];b=rows[min(len(rows)-1,j+1)]
            dt=(b[0]-a[0])/1e6
            if dt<=0 or (t-a[0])>500000 or (b[0]-t)>500000:continue
            velocity_world=(b[3]-a[3])/dt
            velocity_local=rec['lidar2global'][:3,:3].T@velocity_world
            rec['gt_velocity'][bi]=velocity_local[:2];rec['velocity_valid'][bi]=True

def build(args):
    split=json.loads(Path(args.split).read_text(encoding='utf-8'))
    partition={sid:name for name,ids in split.items() for sid in ids}
    if len(partition)!=sum(map(len,split.values())):raise ValueError('Overlapping partitions')
    target_root=PurePosixPath(args.data_root)
    all_records={};indices={};manifest={}
    with zipfile.ZipFile(args.archive) as z:
        def read(name):return json.loads(z.read('V2X-Seq-SPD/'+name))
        for role,side in SIDES.items():
            rows=read(side+'/data_info.json'); records=[]
            rows.sort(key=lambda r:(r['sequence_id'],int(r['image_timestamp']),r['frame_id']))
            for r in rows:
                load=lambda key:read(side+'/'+r[key])
                # Published infrastructure camera calibration blocks are
                # invertible projection transforms, not orthonormal rotations.
                # Preserve their values and use a full inverse, never transpose.
                l2c=transform(load('calib_lidar_to_camera_path' if role=='vehicle' else 'calib_virtuallidar_to_camera_path'),strict_rigid=False)
                c2l=np.linalg.inv(l2c)
                if role=='vehicle':
                    l2e=transform(load('calib_lidar_to_novatel_path'));e2g=transform(load('calib_novatel_to_world_path'))
                else:l2e=np.eye(4);e2g=transform(load('calib_virtuallidar_to_world_path'))
                l2g=e2g@l2e
                k=np.asarray(load('calib_camera_intrinsic_path')['cam_K'],np.float64).reshape(3,3)
                cam={'data_path':str(target_root/side/r['image_path']),'timestamp':int(r['image_timestamp']),
                     'cam_intrinsic':k,'sensor2lidar_rotation':c2l[:3,:3],'sensor2lidar_translation':c2l[:3,3],
                     'lidar2cam_r':l2c[:3,:3],'lidar2cam_t':l2c[:3,3]}
                rec={'token':role+':'+r['frame_id'],'frame_id':r['frame_id'],'sequence_id':r['sequence_id'],
                     'scene_token':role+':'+r['sequence_id'],'timestamp':int(r['image_timestamp']),
                     'label_timestamp_us':int(r['image_timestamp']),'lidar_timestamp_us':int(r['pointcloud_timestamp']),
                     'image_timestamp_us':int(r['image_timestamp']),'partition':partition[r['sequence_id']],
                     'lidar_path':str(target_root/side/r['pointcloud_path']),'sweeps':[],
                     'lidar2ego_rotation':quaternion(l2e[:3,:3]),'lidar2ego_translation':l2e[:3,3],
                     'ego2global_rotation':quaternion(e2g[:3,:3]),'ego2global_translation':e2g[:3,3],
                     'lidar2global':l2g,'cams':{'CAM_FRONT':cam},'can_bus':np.zeros(18,dtype=np.float64),
                     'intersection_loc':r.get('intersection_loc',''),'type':role}
                rec.update(annotation(load('label_camera_std_path')))
                records.append(rec)
            groups=collections.defaultdict(list)
            for rec in records:groups[rec['sequence_id']].append(rec)
            for group in groups.values():
                for j,rec in enumerate(group):
                    rec['frame_idx']=j;rec['prev']=group[j-1]['token'] if j else ''
                    rec['next']=group[j+1]['token'] if j+1<len(group) else ''
            add_velocity(records)
            all_records[role]=records;indices[role]={r['frame_id']:r for r in records}
            print('Converted',role,len(records),flush=True)
        coop=[]
        for p in read('cooperative/data_info.json'):
            v=indices['vehicle'][p['vehicle_frame']];i=indices['infrastructure'][p['infrastructure_frame']]
            i2v=np.linalg.inv(v['lidar2global'])@i['lidar2global']
            # Official Coord_transformation_spd applies the supplied offset in
            # vehicle LiDAR coordinates AFTER the full transformation.
            offset=p['system_error_offset']
            i2v[:2,3]+=np.array([float(offset['delta_x'] or 0),float(offset['delta_y'] or 0)])
            rec={'token':'cooperative:'+p['vehicle_frame'],'frame_id':p['vehicle_frame'],
                 'sequence_id':p['vehicle_sequence'],'scene_token':'cooperative:'+p['vehicle_sequence'],
                 'vehicle_token':v['token'],'infrastructure_token':i['token'],
                 'timestamp':v['timestamp'],'label_timestamp_us':v['lidar_timestamp_us'],
                 'vehicle_image_timestamp_us':v['timestamp'],'infrastructure_image_timestamp_us':i['timestamp'],
                 'partition':partition[p['vehicle_sequence']],'lidar2global':v['lidar2global'],
                 'inf2veh':i2v,'veh2inf':np.linalg.inv(i2v),'system_error_offset':offset}
            rec.update(annotation(read('cooperative/label/'+p['vehicle_frame']+'.json')))
            coop.append(rec)
        coop.sort(key=lambda r:(r['sequence_id'],r['timestamp']))
        add_velocity(coop,clock='label_timestamp_us');all_records['cooperative']=coop
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for role,records in all_records.items():
        for part in split:
            selected=[r for r in records if r['partition']==part]
            path=out/(role+'_'+part+'.pkl')
            if not args.check_only:
                with path.open('wb') as f:
                    pickle.dump({'infos':selected,'metadata':{'version':'DAIR-V2X-SPD-R2','classes':CLASSES,'partition':part,'role':role}},f,protocol=4)
            manifest[role+'_'+part]={'frames':len(selected),'boxes':sum(r['gt_num'] for r in selected),
                'velocity_valid_boxes':sum(int(r['velocity_valid'].sum()) for r in selected)}
    archive_hash=hashlib.sha256()
    with Path(args.archive).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):archive_hash.update(block)
    report={'classes':CLASSES,'split_sha256':hashlib.sha256(Path(args.split).read_bytes()).hexdigest(),
            'metadata_archive_sha256':archive_hash.hexdigest(),
            'numpy_version':np.__version__,'data_root':str(target_root),'records':manifest,
            'timestamp_unit':'microseconds','training_annotations':'camera-time labels in local LiDAR coordinates',
            'cooperative_annotations':'official vehicle-LiDAR-time labels, native camera offsets retained',
            'geometry_offset':'official SPD: add delta_x, delta_y after infrastructure-to-vehicle transformation',
            'camera_calibration':'preserve published nonorthonormal camera matrices; use full inverse',
            'velocity':'offline centered or one-sided within-track world-coordinate differences, maximum adjacent gap 0.5 s',
            'images_required_before_training':True,'check_only':args.check_only}
    (out/'conversion_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['archive','split','data-root','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--check-only',action='store_true')
    build(p.parse_args())
