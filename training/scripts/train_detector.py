"""Train a camera detector with development selection and a recorded budget."""
import argparse,hashlib,json,os,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(args):
    import torch,mmcv
    from mmcv.runner import HOOKS,Hook
    from mmdet.apis import set_random_seed
    from mmdet3d.apis import train_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    root=Path(os.environ['AIONET_ROOT']);path=root/'configs'/(args.name+'.py')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Expose one CUDA GPU per training process')
    cfg=mmcv.Config.fromfile(str(path));cfg.gpu_ids=[0]
    if not 1<=args.epochs<=cfg.runner.max_epochs:raise ValueError('Budget must fit within the configured learning-rate schedule')
    @HOOKS.register_module(force=True)
    class AiONetEpochBudget(Hook):
        def after_train_epoch(self,runner):
            # max_iters retains the original schedule; only the outer epoch loop stops.
            if runner.epoch+1>=args.epochs:runner._max_epochs=runner.epoch+1
    cfg.custom_hooks=list(cfg.get('custom_hooks',[]))+[dict(type='AiONetEpochBudget',priority='LOWEST')]
    work=Path(cfg.work_dir);work.mkdir(parents=True,exist_ok=True)
    if args.resume:
        checkpoint=Path(args.resume).resolve();checkpoint.relative_to(work.resolve())
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        cfg.resume_from=str(checkpoint)
    elif any(work.glob('*.pth')):raise ValueError('Pass --resume to continue existing checkpoints')
    split=root/'manifests/split.json';conversion=json.loads((root/'infos/conversion_manifest.json').read_text())
    if conversion['split_sha256']!=sha(split):raise ValueError('Metadata was generated from a different scenario split')
    cfg.seed=17;set_random_seed(17,deterministic=False)
    dataset=build_dataset(cfg.data.train)
    if not len(dataset):raise ValueError('Training dataset is empty')
    model=build_model(cfg.model);model.init_weights();model.CLASSES=dataset.CLASSES
    stamp=time.strftime('%Y%m%d_%H%M%S')
    provenance=dict(config_sha256=sha(path),split_sha256=sha(split),annotation_sha256=sha(cfg.data.train.ann_file),
        seed=17,epochs=args.epochs,schedule_epochs=cfg.runner.max_epochs,gpu=torch.cuda.get_device_name(0),classes=list(dataset.CLASSES))
    cfg.checkpoint_config.meta={'CLASSES':dataset.CLASSES,'config':cfg.pretty_text,'provenance':provenance}
    cfg.dump(str(work/'resolved_config.py'))
    (work/('provenance_'+stamp+'.json')).write_text(json.dumps(provenance,indent=2))
    train_model(model,[dataset],cfg,distributed=False,validate=cfg.get('paper_checkpoint_policy')!='final_epoch',timestamp=stamp,meta=provenance)
    status=dict(config_sha256=sha(path),completed_unix=time.time())
    if cfg.get('paper_checkpoint_policy')=='final_epoch':
        selected=work/('epoch_%d.pth'%args.epochs)
        if not selected.is_file():raise FileNotFoundError(selected)
        status.update(checkpoint=str(selected),checkpoint_policy='fixed final epoch',checkpoint_sha256=sha(selected))
    (root/'status'/(args.name+'_training_complete.json')).write_text(json.dumps(status,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--resume');main(p.parse_args())
