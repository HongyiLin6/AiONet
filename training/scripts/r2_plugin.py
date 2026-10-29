"""SPD adapters used only by the newly added R2 experiments."""
import copy
import mmcv
import numpy as np
import torch
from mmdet.datasets import DATASETS, PIPELINES
from projects.mmdet3d_plugin.SideModule.BEVFormer.datasets.nuscenes_dataset import CustomNuScenesDataset
from projects.mmdet3d_plugin.SideModule.PETR.datasets.nuscenes_dataset import PETRCustomNuScenesDataset

@PIPELINES.register_module()
class R2ResizeMultiview:
    def __init__(self,width=960,height=540):self.width=width;self.height=height
    def __call__(self,results):
        images=[];shapes=[];scales=[]
        results['ori_shape']=[im.shape for im in results['img']]
        for i,img in enumerate(results['img']):
            h,w=img.shape[:2];sx=self.width/w;sy=self.height/h
            images.append(mmcv.imresize(img,(self.width,self.height)))
            shapes.append(images[-1].shape);scales.append(np.array([sx,sy,sx,sy],np.float32))
            for key in ['lidar2img','intrinsics','cam_intrinsic']:
                if key in results:
                    a=results[key][i].copy();a[0,:]*=sx;a[1,:]*=sy;results[key][i]=a
        results.update(img=images,img_shape=shapes,pad_shape=shapes,scale_factor=scales)
        return results

class SPDRecords:
    def load_annotations(self,ann_file):
        data=mmcv.load(ann_file,file_format='pkl')
        self.metadata=data['metadata'];self.version=self.metadata['version']
        return sorted(data['infos'],key=lambda r:(r['scene_token'],r['timestamp']))[::self.load_interval]
    def evaluate(self,results,metric=None,logger=None,**kwargs):
        """Global-score area AP over a common forward-facing BEV ROI."""
        if len(results)!=len(self.data_infos):raise ValueError('Prediction/frame count mismatch')
        thresholds=(.25,.5);counts=np.zeros(len(self.CLASSES),int)
        events={thr:{c:[] for c in range(len(self.CLASSES))} for thr in thresholds}
        def mask(boxes):
            c=boxes.gravity_center
            return (c[:,0]>=0)&(c[:,0]<=51.2)&(c[:,1]>=-51.2)&(c[:,1]<=51.2)
        for rec,result in zip(self.data_infos,results):
            out=result['pts_bbox'];pred=out['boxes_3d'];keep=mask(pred)
            pred=pred[keep];scores=out['scores_3d'][keep].detach().cpu().numpy();labels=out['labels_3d'][keep].detach().cpu().numpy()
            gt=self.box_type_3d(rec['gt_boxes'],box_dim=7,origin=(.5,.5,.5)).convert_to(self.box_mode_3d)
            gkeep=mask(gt);gt=gt[gkeep]
            glabels=np.array([self.CLASSES.index(n) for n in rec['gt_names']])[gkeep.cpu().numpy()]
            for c in range(len(self.CLASSES)):
                pi=np.flatnonzero(labels==c);gi=np.flatnonzero(glabels==c);counts[c]+=len(gi)
                pi=pi[np.argsort(-scores[pi],kind='stable')]
                if len(pi) and len(gi):
                    overlaps=pred.overlaps(pred[pi].to('cuda'),gt[gi].to('cuda')).detach().cpu().numpy()
                else:overlaps=np.zeros((len(pi),len(gi)))
                for thr in thresholds:
                    matched=set()
                    for row,i in enumerate(pi):
                        candidates=[j for j in range(len(gi)) if j not in matched]
                        j=max(candidates,key=lambda j:overlaps[row,j]) if candidates else None
                        tp=bool(j is not None and overlaps[row,j]>thr)
                        if tp:matched.add(j)
                        events[thr][c].append((float(scores[i]),tp))
        metrics={}
        for thr in thresholds:
            aps=[]
            for c,name in enumerate(self.CLASSES):
                if counts[c]==0:continue
                rows=sorted(events[thr][c],key=lambda p:-p[0]);tp=np.cumsum([r[1] for r in rows]);fp=np.cumsum([not r[1] for r in rows])
                recall=tp/counts[c];precision=tp/np.maximum(tp+fp,1)
                r=np.r_[0.,recall,1.];p=np.r_[0.,precision,0.];p=np.maximum.accumulate(p[::-1])[::-1]
                change=np.flatnonzero(r[1:]!=r[:-1]);ap=float(np.sum((r[change+1]-r[change])*p[change+1]));aps.append(ap)
                metrics[name+'_AP_'+format(thr,'.2f')]=ap
            metrics['mAP_'+format(thr,'.2f')]=float(np.mean(aps)) if aps else 0.
        return metrics

@DATASETS.register_module()
class R2BEVDataset(SPDRecords,CustomNuScenesDataset):
    def get_data_info(self,index):
        # Parent code writes pose fields into can_bus; isolate each invocation.
        original=self.data_infos[index]
        self.data_infos[index]=copy.deepcopy(original)
        try:return super().get_data_info(index)
        finally:self.data_infos[index]=original
    def prepare_train_data(self,index):
        scene=self.data_infos[index]['scene_token'];start=index
        while start>0 and index-start<self.queue_length-1 and self.data_infos[start-1]['scene_token']==scene:start-=1
        indices=list(range(start,index+1))
        indices=[start]*(self.queue_length-len(indices))+indices
        queue=[]
        for j in indices:
            data=self.get_data_info(j);self.pre_pipeline(data);example=self.pipeline(data)
            if example is None:return None
            queue.append(example)
        if self.filter_empty_gt and not (queue[-1]['gt_labels_3d'].data!=-1).any():return None
        return self.union2one(queue)
    def __getitem__(self,index):
        if self.test_mode:return self.prepare_test_data(index)
        for _ in range(100):
            value=self.prepare_train_data(index)
            if value is not None:return value
            index=self._rand_another(index)
        raise RuntimeError('Could not draw a valid annotated training frame')

@DATASETS.register_module()
class R2PETRDataset(SPDRecords,PETRCustomNuScenesDataset):
    pass
