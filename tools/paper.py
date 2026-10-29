"""Source-only full-paper experiment launcher."""
import argparse,copy,json,os,pickle,pprint,runpy,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aionet.paper.experiments import experiments,get_experiment,payloads
from pipeline import launch

def detector_configs(work):
    old=os.environ.get('AIONET_ROOT');os.environ['AIONET_ROOT']=str(work)
    try:
        for model in ['bevformer','petr','detr3d']:
            for side in ['vehicle','infrastructure']:
                template=model+'_'+('vehicle' if model=='bevformer' and side=='vehicle' else 'infrastructure')
                cfg=runpy.run_path(str(ROOT/'training/configs'/(template+'.py')))
                cfg={k:v for k,v in cfg.items() if not k.startswith('_')}
                original='vehicle' if template.endswith('_vehicle') else 'infrastructure'
                def update(value,key=''):
                    if key in ('classes','class_names'):return ['Car']
                    if key=='num_classes':return 1
                    if isinstance(value,dict):return {k:update(v,k) for k,v in value.items()}
                    if isinstance(value,list):return [update(v) for v in value]
                    if isinstance(value,tuple):return tuple(update(v) for v in value)
                    if isinstance(value,str):return value.replace('/infos/'+original+'_','/infos/'+side+'_').replace('_development.pkl','_perception_train.pkl')
                    return value
                cfg=update(cfg);cfg['runner']['max_epochs']=24;cfg['work_dir']=str(work/'runs/detectors'/(model+'_'+side))
                cfg['paper_checkpoint_policy']='final_epoch';cfg['store_queries']=True
                cfg['evaluation']=dict(interval=24,metric='bbox')
                (work/'configs'/(model+'_'+side+'.py')).write_text('\n'.join(k+' = '+pprint.pformat(v,width=110,sort_dicts=False) for k,v in cfg.items())+'\n')
    finally:
        if old is None:os.environ.pop('AIONET_ROOT',None)
        else:os.environ['AIONET_ROOT']=old

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',type=Path,default=ROOT/'workdir_perception');p.add_argument('--gpu',default='0')
    s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('list');q.add_argument('--table')
    q=s.add_parser('init');q.add_argument('--data-root',type=Path,required=True)
    q=s.add_parser('prepare');q.add_argument('--metadata-zip',type=Path,required=True)
    q=s.add_parser('train-detector');q.add_argument('--name',required=True);q.add_argument('--resume')
    q=s.add_parser('cache');q.add_argument('--name',required=True);q.add_argument('--partition',choices=['perception_train','test'],required=True)
    q=s.add_parser('train');q.add_argument('--experiment',required=True);q.add_argument('--resume',action='store_true')
    q=s.add_parser('evaluate');q.add_argument('--experiment',required=True);q.add_argument('--nds',action='store_true');q.add_argument('--attributes',type=Path)
    q=s.add_parser('temporal');q.add_argument('--experiment',default='fusion_adaptive');q.add_argument('--mode',choices=['stochastic','sweep','detector_sweep'],default='stochastic');q.add_argument('--temposense',type=Path,required=True)
    q=s.add_parser('train-temposense');q.add_argument('--source',default='bevformer_infrastructure');q.add_argument('--epochs',type=int,default=6);q.add_argument('--resume',action='store_true')
    q=s.add_parser('run');q.add_argument('script');q.add_argument('arguments',nargs=argparse.REMAINDER)
    s.add_parser('payload')
    args=p.parse_args();work=args.workdir.resolve()
    if args.command=='list':print(json.dumps([j for j in experiments() if args.table is None or j['table']==args.table],indent=2));return
    if args.command=='payload':print(json.dumps(payloads(),indent=2));return
    if args.command=='init':
        if (work/'manifests/split.json').exists():raise ValueError('Use a new workspace to initialize a protocol')
        data=args.data_root.resolve()
        if not (data/'cooperative/data_info.json').exists():raise ValueError('Expected an extracted SPD dataset root')
        for part in ['configs','manifests','status','runs','infos','traces','weights','data','cache','src/baselines','generated','results','tmp']:(work/part).mkdir(parents=True,exist_ok=True)
        (work/'data/V2X-Seq-SPD').symlink_to(data,target_is_directory=True)
        (work/'src/metrovision').symlink_to(ROOT/'training/vendor/metrovision',target_is_directory=True)
        shutil.copyfile(ROOT/'configs/perception_split.json',work/'manifests/split.json')
        detector_configs(work)
        (work/'manifests/experiment_specs.json').write_text(json.dumps(experiments(),indent=2))
        print(work);return
    if args.command=='prepare':
        launch('convert_spd_metadata',['--archive',str(args.metadata_zip.resolve()),'--split',str(work/'manifests/split.json'),'--data-root',str(work/'data/V2X-Seq-SPD'),'--output',str(work/'infos')],work,'')
        counts={}
        for path in (work/'infos').glob('*.pkl'):
            with path.open('rb') as f:bundle=pickle.load(f)
            for row in bundle['infos']:
                mask=row['gt_names']=='Car'
                for name in ['gt_boxes','gt_names','class','gt_track_ids','valid_flag','gt_velocity','velocity_valid']:row[name]=row[name][mask]
                row['class'][:]=0;row['gt_num']=int(mask.sum())
            bundle['metadata']['classes']=['Car']
            with path.open('wb') as f:pickle.dump(bundle,f,protocol=4)
            counts[path.stem]={'frames':len(bundle['infos']),'boxes':sum(r['gt_num'] for r in bundle['infos']),
                'velocity_valid_boxes':sum(int(r['velocity_valid'].sum()) for r in bundle['infos'])}
        report=json.loads((work/'infos/conversion_manifest.json').read_text());report['classes']=['Car'];report['class_filter']='Car only after official annotation conversion';report['records']=counts
        (work/'infos/conversion_manifest.json').write_text(json.dumps(report,indent=2))
    elif args.command=='train-detector':
        command=['--name',args.name,'--epochs','24']
        if args.resume:command+=['--resume',str(Path(args.resume).resolve())]
        launch('train_detector',command,work,args.gpu)
    elif args.command=='cache':launch('extract_detector_cache',['--name',args.name,'--partition',args.partition],work,args.gpu)
    elif args.command in ('train','evaluate'):
        command=['--experiment',args.experiment]
        if args.command=='train' and args.resume:command+=['--resume']
        if args.command=='evaluate':
            if args.nds:command+=['--nds']
            if args.attributes:command+=['--attributes',str(args.attributes.resolve())]
        launch('paper_'+args.command,command,work,args.gpu)
    elif args.command=='temporal':launch('paper_temporal',['--experiment',args.experiment,'--mode',args.mode,'--temposense',str(args.temposense.resolve())],work,args.gpu)
    elif args.command=='train-temposense':
        command=['--source',args.source,'--epochs',str(args.epochs)]
        if args.resume:command+=['--resume']
        launch('paper_train_temposense',command,work,args.gpu)
    else:launch(args.script,args.arguments,work,args.gpu)

if __name__=='__main__':main()
