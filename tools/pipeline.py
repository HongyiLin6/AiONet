"""Portable launcher for the GPU perception and branch-cache workflow."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

PACKAGE=Path(__file__).resolve().parents[1]
SCRIPTS=PACKAGE/'training/scripts'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def launch(script,arguments,work,gpu):
    if not (SCRIPTS/(script+'.py')).is_file():raise ValueError('Unknown script: '+script)
    arguments=list(arguments)
    for i,flag in enumerate(arguments[:-1]):
        if flag in ('--output','--cache','--cache-manifest','--archive','--split','--data-root','--checkpoint','--resume','--model-dir'):
            arguments[i+1]=str(Path(arguments[i+1]).resolve())
    env=dict(os.environ,AIONET_ROOT=str(work),PYTHONNOUSERSITE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu,AIONET_GPU_UUIDS=gpu)
    vendor=PACKAGE/'training/vendor/metrovision'
    env['PYTHONPATH']=os.pathsep.join([str(work/'generated'),str(SCRIPTS),str(vendor),str(vendor/'mmdetection3d'),str(PACKAGE)])
    env.setdefault('OMP_NUM_THREADS','4');env.setdefault('OPENBLAS_NUM_THREADS','1')
    subprocess.run([sys.executable,str(SCRIPTS/(script+'.py'))]+arguments,cwd=str(vendor),env=env,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir',type=Path,default=PACKAGE/'workdir');p.add_argument('--gpu',default='0')
    sub=p.add_subparsers(dest='stage',required=True)
    init=sub.add_parser('init');init.add_argument('--data-root',type=Path,required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--metadata-zip',type=Path,required=True)
    train=sub.add_parser('train');train.add_argument('--name',choices=['bevformer_vehicle','bevformer_infrastructure','petr_infrastructure','detr3d_infrastructure'],required=True);train.add_argument('--resume');train.add_argument('--epochs',type=int,default=12)
    run=sub.add_parser('run');run.add_argument('script');run.add_argument('arguments',nargs=argparse.REMAINDER)
    freeze=sub.add_parser('freeze');freeze.add_argument('--selector-dir',type=Path,nargs='+',required=True)
    args=p.parse_args();work=args.workdir.resolve()
    if args.stage=='init':
        data=args.data_root.resolve()
        if not (data/'cooperative/data_info.json').is_file():raise ValueError('Expected extracted SPD cooperative/data_info.json')
        for name in ['manifests','configs','status','runs','infos','traces','weights','cache','data','src/baselines','tmp','generated']:(work/name).mkdir(parents=True,exist_ok=True)
        for src,dest in [(data,work/'data/V2X-Seq-SPD'),(PACKAGE/'training/vendor/metrovision',work/'src/metrovision')]:
            if not dest.exists():dest.symlink_to(src,target_is_directory=True)
            elif dest.resolve()!=src:raise ValueError('Existing path points elsewhere: '+str(dest))
        for src in (PACKAGE/'training/configs').glob('*.py'):
            dest=work/'configs'/src.name
            if dest.exists() and dest.read_bytes()!=src.read_bytes():raise ValueError('Config already modified: '+str(dest))
            shutil.copyfile(src,dest)
        dest=work/'manifests/split.json';split=PACKAGE/'configs/split.json'
        if dest.exists() and dest.read_bytes()!=split.read_bytes():raise ValueError('Existing scenario split differs')
        shutil.copyfile(split,dest)
        print(work)
    elif args.stage=='prepare':
        launch('convert_spd_metadata',['--archive',str(args.metadata_zip.resolve()),'--split',str(work/'manifests/split.json'),'--data-root',str(work/'data/V2X-Seq-SPD'),'--output',str(work/'infos')],work,'')
        launch('prepare_delay_traces',[str(args.metadata_zip.resolve()),str(work/'traces')],work,'')
    elif args.stage=='train':
        command=['--name',args.name,'--epochs',str(args.epochs)]
        if args.resume:command+=['--resume',args.resume]
        launch('train_detector',command,work,args.gpu)
    elif args.stage=='run':launch(args.script,args.arguments,work,args.gpu)
    elif args.stage=='freeze':
        for name in ['bevformer_vehicle','bevformer_infrastructure','petr_infrastructure','detr3d_infrastructure']:
            if not (work/'status'/(name+'_training_complete.json')).is_file():raise ValueError('Complete detector training before freezing: '+name)
            if len(list((work/'runs/detectors'/name).glob('best_mAP_0.25_*.pth')))!=1:raise ValueError('Expected one development-selected detector checkpoint: '+name)
        for name in ['registered_fusion','temposense']:
            if not (work/'runs'/name/'complete.json').is_file():raise ValueError('Complete module fitting before freezing: '+name)
        files=[work/'manifests/split.json',work/'traces/trace_manifest.json']+list((work/'configs').glob('*.py'))+list((work/'runs/choices').glob('*.json'))+list((work/'runs').rglob('best.pt'))+list((work/'runs/detectors').rglob('best_mAP*.pth'))
        if not files or not (work/'runs/choices/F_execution.json').exists():raise ValueError('Build development choices before freezing')
        for folder in args.selector_dir:
            folder=folder.resolve()
            for name in ['model.joblib','calibration.json','development.json','provenance.json']:
                path=folder/name
                if not path.is_file():raise ValueError('Fit and calibrate the selector first: '+str(path))
                path.relative_to(work)
                files.append(path)
        path=work/'manifests/evaluation_choices_frozen.json'
        if path.exists():raise ValueError('Use a new workspace for a new evaluation protocol')
        path.write_text(json.dumps(dict(frozen_files_sha256={str(f.relative_to(work)):digest(f) for f in files},code_sha256={str(f.relative_to(PACKAGE)):digest(f) for f in [*SCRIPTS.glob('*.py'),*(PACKAGE/'aionet').glob('*.py'),*(PACKAGE/'tools').glob('*.py')]},purpose='Parameters fixed before test cache construction'),indent=2))
        print(path)

if __name__=='__main__':main()
