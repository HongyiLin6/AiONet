"""Official EMIFF architecture fitted to the common camera-only SPD study."""
import argparse
import json
import os
import time
import numpy as np
import torch
import mmcv
from torch.utils.data import DataLoader
from experiment_io import ROOT,sha,atomic_json
from structured_fusion import Detections
from evaluation_metrics import summarize

def main(args):
    if args.smoke:
        # Smoke tests have no scientific timing output. Use available memory
        # within the four authorized devices, before CUDA is initialized.
        import subprocess
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        allowed=os.environ['AIONET_GPU_UUIDS'].split(',')
        choices=[(int(line.split(',')[1]),line.split(',')[0].strip()) for line in lines if line.split(',')[0].strip() in allowed]
        memory,uuid=max(choices)
        if memory<24000:raise RuntimeError('EMIFF software check requires 24 GB free on an authorized device')
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        print('Software smoke GPU',uuid,'free MiB',memory,flush=True)
    import r2_emiff
    from mmdet3d.models import build_model
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from emiff_data import EMIFFPairs,collate
    assert os.environ['CUDA_VISIBLE_DEVICES'] in os.environ['AIONET_GPU_UUIDS'].split(',')
    torch.manual_seed(17);np.random.seed(17)
    cfg=mmcv.Config.fromfile(str(ROOT/'configs/emiff.py'));model=build_model(cfg.model);model.init_weights();model=model.cuda()
    fixture=str(ROOT/'tmp/smoke-synthetic-pixels-bevformer_vehicle.jpg') if args.smoke else None
    training=EMIFFPairs('perception_train',fixture);validation=EMIFFPairs('development',fixture)
    if args.smoke:training.rows=training.rows[:4];validation.rows=validation.rows[:2]
    loader=DataLoader(training,batch_size=4,shuffle=True,num_workers=0 if args.smoke else 4,collate_fn=collate,pin_memory=True)
    val=DataLoader(validation,batch_size=2,shuffle=False,num_workers=0 if args.smoke else 4,collate_fn=collate,pin_memory=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12,eta_min=2e-7)
    out=ROOT/('tmp/emiff-smoke' if args.smoke else 'runs/emiff');out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((ROOT/'manifests/emiff_adaptation.json').read_text())
    provenance['training']['epochs']=12
    provenance['training']['schedule']='12-epoch cosine schedule, user-authorized fast validation plan'
    atomic_json(ROOT/'manifests/emiff_adaptation.json',provenance)
    history=[];best=-1.;best_epoch=0;start=0
    if not args.smoke and (out/'complete.json').exists():return
    if not args.smoke and (out/'latest.pt').exists():
        state=torch.load(out/'latest.pt',map_location='cpu');assert provenance==state['provenance']
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler'])
        start=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];history=state['history']
    for epoch in range(start,1 if args.smoke else 12):
        model.train();total=0.;count=0;begin=time.monotonic()
        for b in loader:
            optimizer.zero_grad(set_to_none=True)
            boxes=[LiDARInstance3DBoxes(torch.from_numpy(x).cuda(),box_dim=7,origin=(.5,.5,.5)) for x in b['boxes']]
            labels=[torch.as_tensor(x,device='cuda') for x in b['labels']]
            values=model.forward_train(b['image'].cuda(),b['meta'],boxes,labels)
            loss,_=model._parse_losses(values)
            if not torch.isfinite(loss):raise ValueError('Nonfinite EMIFF loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),35.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite EMIFF gradients')
            optimizer.step();total+=loss.item()*len(boxes);count+=len(boxes)
        schedule.step();row=dict(epoch=epoch+1,training_loss=total/count,seconds=time.monotonic()-begin)
        if args.smoke or (epoch+1)%2==0:
            model.eval();frames=[]
            with torch.no_grad():
                for b in val:
                    for result,gt,lab in zip(model.simple_test(b['image'].cuda(),b['meta']),b['boxes'],b['labels']):
                        box=result['boxes_3d'];prediction=Detections(torch.cat((box.gravity_center,box.tensor[:,3:7]),1).cpu().numpy(),result['scores_3d'].cpu().numpy(),result['labels_3d'].cpu().numpy()).roi()
                        frames.append((prediction,gt,lab))
            metrics=summarize(frames);row['development']=metrics
            if metrics['AP25']>best:
                best=metrics['AP25'];best_epoch=epoch+1
                if not args.smoke:torch.save(dict(state_dict=model.state_dict(),provenance=provenance,epoch=best_epoch),out/'best.pt')
        history.append(row);print(json.dumps(row),flush=True)
        if not args.smoke:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),
                provenance=provenance,epoch=epoch+1,best=best,best_epoch=best_epoch,history=history),out/'latest.pt')
            atomic_json(out/'history.json',history)
    if args.smoke:atomic_json(ROOT/'status/emiff_software_smoke.json',dict(kind='synthetic_software_test_only',forward_backward_and_inference=True,no_checkpoint_saved=True))
    else:atomic_json(out/'complete.json',dict(provenance=provenance,best_epoch=best_epoch,development_AP25=best,checkpoint_sha256=sha(out/'best.pt')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args())
