"""Cache completed detector outputs with causal sequential state and provenance.

Ground-truth boxes are never written into this cache or used by inference.
Test-cache extraction requires a separately frozen evaluation-protocol marker.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(args):
    import torch
    import numpy as np
    import mmcv
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint,wrap_fp16_model
    from mmdet.datasets import build_dataloader
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    root=Path(os.environ['AIONET_ROOT'])
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    config=root/'configs'/(args.name+'.py');cfg=mmcv.Config.fromfile(str(config))
    if args.partition=='test' and cfg.get('paper_checkpoint_policy')!='final_epoch':
        from freeze_evaluation_choices import verify_frozen
        verify_frozen()
    completion=json.loads((root/'status'/(args.name+'_training_complete.json')).read_text())
    work=root/'runs/detectors'/args.name
    matches=[Path(completion['checkpoint'])] if completion.get('checkpoint') else list(work.glob('best_mAP_0.25_*.pth'))
    if len(matches)!=1:raise RuntimeError('Expected one development-selected best checkpoint, found '+str(matches))
    checkpoint=matches[0];checkpoint_sha=sha(checkpoint)
    role='vehicle' if args.name.endswith('_vehicle') else 'infrastructure'
    data_cfg=copy.deepcopy(cfg.data.val);data_cfg.ann_file=str(root/'infos'/(role+'_'+args.partition+'.pkl'))
    dataset=build_dataset(data_cfg)
    loader=build_dataloader(dataset,samples_per_gpu=1,workers_per_gpu=4,num_gpus=1,dist=False,shuffle=False)
    output=root/'cache/detectors'/args.name/args.partition;output.mkdir(parents=True,exist_ok=True)
    done=output/'manifest.json'
    provenance={'name':args.name,'partition':args.partition,'role':role,'checkpoint':str(checkpoint),
        'checkpoint_sha256':checkpoint_sha,'config_sha256':sha(config),'annotation_sha256':sha(data_cfg.ann_file),
        'classes':list(dataset.CLASSES),'family':'F' if args.name.startswith('bevformer_') else 'S',
        'feature_codec':'float16 NPY for BEV','box_origin':'geometric_center',
        'causal_source_history':'chronological inference; state reset on scenario change; no later frame input',
        'contains_ground_truth':False,'timing_scope':'model forward including detector head, synchronized, excludes image reading and disk writing',
        'image_preprocessing':'frozen config deterministic test pipeline',
        'cache_extraction_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'cache_extraction_code_sha256':sha(__file__)}
    if done.exists():
        old=json.loads(done.read_text())
        assert old['checkpoint_sha256']==checkpoint_sha and old['config_sha256']==sha(config)
        assert len(old['records'])==len(dataset)
        if all((output/r['detections']).is_file() and (r.get('feature') is None or (output/r['feature']).is_file()) and (not cfg.get('store_queries') or (r.get('query') and (output/r['query']).is_file())) for r in old['records']):
            print('Cache already complete',args.name,args.partition,flush=True);return
    model=build_model(cfg.model);model.init_weights()
    load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True)
    wrap_fp16_model(model);model.eval();model=MMDataParallel(model.cuda(),device_ids=[0])
    captured={}
    def capture(_module,_args,value):
        if isinstance(value,dict):
            if 'bev_embed' in value and value['bev_embed'] is not None:captured['bev']=value['bev_embed'].detach()
            if cfg.get('store_queries'):captured['xyz']=value['all_bbox_preds'][-1,...,[0,1,4]].detach()
    hook=model.module.pts_bbox_head.register_forward_hook(capture) if args.name.startswith('bevformer_') or cfg.get('store_queries') else None
    def capture_query(module,inputs):captured['query']=inputs[0].detach()
    query_hook=model.module.pts_bbox_head.cls_branches[-1].register_forward_pre_hook(capture_query) if cfg.get('store_queries') else None
    rows=[];previous_scene=None;scene_index=0
    with torch.no_grad():
        for index,batch in enumerate(loader):
            rec=dataset.data_infos[index]
            if rec['scene_token']!=previous_scene:scene_index=0;previous_scene=rec['scene_token']
            captured.clear();torch.cuda.synchronize();start=time.perf_counter()
            results=model(return_loss=False,rescale=True,**batch)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            pred=results[0]['pts_bbox'];b=pred['boxes_3d']
            boxes=torch.cat((b.gravity_center,b.tensor[:,3:7]),1).cpu().numpy()
            scores=pred['scores_3d'].cpu().numpy();labels=pred['labels_3d'].cpu().numpy()
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise ValueError('Nonfinite detector output')
            velocity=b.tensor[:,7:9].cpu().numpy() if b.tensor.shape[1]>=9 else np.full((len(b),2),np.nan,np.float32)
            file=rec['frame_id']+'.npz';tmp=output/(file+'.writing')
            with tmp.open('wb') as f:np.savez_compressed(f,boxes=boxes.astype(np.float32),scores=scores.astype(np.float32),labels=labels.astype(np.int64),velocity=velocity.astype(np.float32))
            tmp.replace(output/file)
            row={'token':rec['token'],'frame_id':rec['frame_id'],'scenario':rec['sequence_id'],
                'observation_us':rec['image_timestamp_us'],'detections':file,'detections_sha256':sha(output/file),
                'inference_seconds':elapsed,'cold_or_initial_history':scene_index<3,
                'lidar2global':rec['lidar2global'].tolist()}
            if cfg.get('store_queries'):
                q=captured['query'];xyz=captured['xyz']
                if q.ndim!=3 or q.shape[0]!=1 or xyz.shape[:2]!=q.shape[:2]:raise ValueError('Unexpected native decoder query shape')
                qfile=rec['frame_id']+'.queries.npz'
                np.savez_compressed(output/qfile,query=q[0].float().cpu().numpy(),xyz=xyz[0].float().cpu().numpy())
                row.update(query=qfile,query_sha256=sha(output/qfile))
            if args.name.startswith('bevformer_'):
                bev=captured['bev'];h=cfg.model.pts_bbox_head.bev_h;w=cfg.model.pts_bbox_head.bev_w
                if tuple(bev.shape)==(h*w,1,256):bev=bev.permute(1,0,2)
                assert tuple(bev.shape)==(1,h*w,256),tuple(bev.shape)
                grid=bev[0].reshape(h,w,256).permute(2,0,1)
                # Materialize CHW on the GPU before transfer. np.save writes
                # strided CHW arrays much more slowly even with identical bytes.
                data=grid.contiguous().cpu().numpy().astype(np.float16)
                if index<2:
                    reference=grid.cpu().numpy().astype(np.float16)
                    assert np.array_equal(data,reference),'Feature layout conversion changed values'
                feature=rec['frame_id']+'.npy';tmp=output/(feature+'.writing')
                with tmp.open('wb') as f:np.save(f,data,allow_pickle=False)
                tmp.replace(output/feature);row.update(feature=feature,feature_sha256=sha(output/feature))
            rows.append(row);scene_index+=1
            if index%200==0:print(args.name,args.partition,index+1,'/',len(dataset),flush=True)
    if hook:hook.remove()
    if query_hook:query_hook.remove()
    provenance.update(records=rows,frames=len(rows),completed_unix=time.time())
    done.write_text(json.dumps(provenance,indent=2))
    print('Completed verified inference cache',args.name,args.partition,len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--partition',choices=['perception_train','selector_fit','development','calibration','test'],required=True)
    args=p.parse_args()
    # Multiple GPU workers may help with different partitions. A completed
    # partition is validated and reused while holding its own exclusive lock.
    lock_path=Path(os.environ['AIONET_ROOT'])/'status'/('cache_'+args.name+'_'+args.partition+'.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        main(args)
