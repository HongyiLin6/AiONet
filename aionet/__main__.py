"""Portable cache evaluation and selector refitting command line."""
import argparse, copy, json, time
from pathlib import Path
import joblib, numpy as np
from threadpoolctl import threadpool_limits
from . import __version__
from .replay import read_json,write_json,sha,verify_manifest,verify_selector,MetricFrames,features,scores_for,choose,commit,evaluate
from .selector import fit

ROOT=Path(__file__).resolve().parents[1]


def load_rows(root,manifest,family,partition,condition=None):
    rows=read_json(root/manifest['exports'][family][partition]['rows'])
    return [r for r in rows if condition is None or r['condition']==list(condition)]


def run_evaluation(args,manifest):
    root=args.artifacts;family=args.family;condition=[args.delay,args.budget]
    rows=load_rows(root,manifest,family,'test',condition)
    if not rows:raise ValueError('Choose one of the conditions in artifacts/manifest.json')
    frames=MetricFrames(root/manifest['exports'][family]['test']['events'])
    policies=args.policies or (['A0','A1','A2','A3'] if family=='F' else ['A0','A1','A3'])
    origin=None
    if args.model_dir:
        policies=['custom'];origin=verify_selector(args.model_dir,root,family)
    output={};started=time.time()
    for policy in policies:
        model=None;threshold=0.;settings=manifest['settings'][family]
        if policy=='custom':
            cal=read_json(args.model_dir/'calibration.json');model=joblib.load(args.model_dir/'model.joblib')
            threshold=cal['chosen']['threshold'] if args.threshold is None else args.threshold;settings=cal['settings']
        elif policy not in ('A0','A1','A2','A3'):
            meta=manifest['models'][family]['R-MLP' if policy=='A4' else policy]
            model=joblib.load(root/meta['file']);threshold=0. if policy=='A4' else meta['threshold']
            if policy!='A4':settings=meta['settings']
        report,decisions=evaluate(rows,frames,policy,settings,model,threshold,args.bootstrap)
        report.update(policy=policy,family=family,delay_median_ms=args.delay,budget_ms=args.budget,threshold=threshold)
        write_json(args.output/(policy+'.json'),report)
        write_json(args.output/(policy+'.decisions.json'),decisions)
        scalar={k:v['value'] for k,v in report['metrics'].items()};output[policy]=scalar
        print(policy,json.dumps({k:scalar[k] for k in ['AP25','AP50','C_coop','H_all','H_acc']},allow_nan=False),flush=True)
    write_json(args.output/'summary.json',output)
    write_json(args.output/'run.json',dict(version=__version__,artifact_manifest_sha256=sha(root/'manifest.json'),elapsed_seconds=time.time()-started,
        family=family,condition=condition,rows=len(rows),policies=policies,bootstrap=args.bootstrap,seed=manifest['seed'],selector_provenance=origin))


def train_selector(args,manifest):
    from .calibration import calibrate
    root=args.artifacts;family=args.family
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a new output directory for each fitting run')
    rows=load_rows(root,manifest,family,'selector_fit')
    if not args.source_evidence:
        rows=[r for r in rows if r['deadline_us']>=r['decision_us']]
    dev=load_rows(root,manifest,family,'development',(234.9,100))
    cal=load_rows(root,manifest,family,'calibration',(234.9,100))
    scenes=[set(r['scenario'] for r in part) for part in [rows,dev,cal]]
    test=set(manifest['exports'][family]['test']['scenarios'])
    assert all(not (a&b) for i,a in enumerate(scenes+[test]) for b in (scenes+[test])[i+1:])
    dimensions=32 if args.source_evidence else 20
    x,positions=features(rows,dimensions)
    y=np.asarray([rows[i]['candidates'][a]['delta'] for i,a in positions],np.float32)
    ids=np.asarray([rows[i]['scenario'] for i,a in positions])
    choices=[];models={}
    # Use this run's measured development timing for cached replay.
    label='R-BH+S' if args.source_evidence else 'R-BH'
    reference=manifest.get('models',{}).get(family,{}).get(label)
    settings=copy.deepcopy(reference['settings'] if reference else manifest['fitting_settings'][family][label])
    if args.regressor=='mlp':
        from .mlp import fit_mlp
        dx,dp=features(dev,20)
        dy=np.asarray([dev[i]['candidates'][a]['delta'] for i,a in dp],np.float32)
        ds=np.asarray([dev[i]['scenario'] for i,a in dp])
        model,best=fit_mlp(x,y,ids,dx,dy,ds);choices=[best]
    else:
        for leaves in ([args.leaves] if args.leaves else [7,15]):
            model=fit(x,y,ids,leaves=leaves,without_age_horizon=args.without_age_horizon,without_source=args.without_source)
            models[leaves]=model
            for penalty in ([args.penalty] if args.penalty is not None else [1.,2.,4.]):
                model['penalty']=penalty;scores=scores_for(model,dev);losses={};increments=[]
                for i,r in enumerate(dev):
                    a=commit(r,choose(r,'selector',settings,scores,i,0.),settings)
                    loss=r['empty_loss'] if a is None else r['ego_loss'] if a=='0' else r['candidates'][a]['loss']
                    losses.setdefault(r['scenario'],[]).append(loss)
                    increments.append(max(0.,r['candidates'][a]['delta']) if a in ('O','P') else 0.)
                choices.append(dict(leaves=leaves,penalty=penalty,loss=float(np.mean([np.mean(v) for v in losses.values()])),positive_increment=float(np.mean(increments))))
        best=min(choices,key=lambda r:(r['loss'],r['positive_increment'],r['leaves'],r['penalty']))
        model=models[best['leaves']];model['penalty']=best['penalty']
    timing=None
    if reference is None:
        from .selector import score
        xx,_=features(dev,dimensions);rng=np.random.default_rng(17);samples=[]
        batches=[xx[rng.integers(0,len(xx),2)] for _ in range(138)]
        for i,batch in enumerate(batches):
            start=time.perf_counter();score(model,batch);elapsed=time.perf_counter()-start
            if i>=10:samples.append(elapsed)
        settings['selector_cost_p99_seconds']=float(np.quantile(samples,.99))
        timing=dict(scope='CPU one thread; two candidate rows; complete score call',warmup=10,samples=samples)
    frames=MetricFrames(root/manifest['exports'][family]['calibration']['events'])
    calibrated=calibrate(model,dev,cal,frames,settings,manifest['settings'][family])
    args.output.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,args.output/'model.joblib',compress=3)
    write_json(args.output/'calibration.json',calibrated)
    write_json(args.output/'development.json',dict(chosen=best,candidates=choices))
    inputs={manifest['exports'][family][part][key] for part in ['selector_fit','development','calibration']
            for key in ['rows','events'] if key in manifest['exports'][family][part]}
    write_json(args.output/'provenance.json',dict(artifact_manifest_sha256=sha(root/'manifest.json'),family=family,seed=17,dimensions=dimensions,
        partitions={k:manifest['exports'][family][k]['scenarios'] for k in ['selector_fit','development','calibration']},
        model_sha256=sha(args.output/'model.joblib'),calibration_sha256=sha(args.output/'calibration.json'),selector_timing=timing,
        fitting_input_sha256={path:sha(root/path) for path in sorted(inputs)},
        timing_source='included development benchmark' if reference else 'local development benchmark'))
    print(json.dumps(dict(development=best,calibration=calibrated['chosen'],mode=calibrated['mode']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',action='version',version=__version__)
    parser.add_argument('--artifacts',type=Path,required=True)
    subs=parser.add_subparsers(dest='command',required=True)
    subs.add_parser('verify',help='Verify artifact SHA256 values')
    run=subs.add_parser('evaluate',help='Regenerate metrics from frame-level cached detection events')
    run.add_argument('--family',choices=['F','S'],default='F');run.add_argument('--delay',type=float,default=234.9);run.add_argument('--budget',type=int,default=100)
    run.add_argument('--policies',nargs='+');run.add_argument('--output',type=Path,default=Path('runs/evaluation'))
    run.add_argument('--threshold',type=float);run.add_argument('--bootstrap',type=int,default=0);run.add_argument('--model-dir',type=Path)
    train=subs.add_parser('fit',help='Fit heads on fitting scenarios, select on development, calibrate on calibration')
    train.add_argument('--family',choices=['F','S'],default='F');train.add_argument('--source-evidence',action='store_true')
    train.add_argument('--without-age-horizon',action='store_true');train.add_argument('--without-source',action='store_true')
    train.add_argument('--regressor',choices=['tree','mlp'],default='tree');train.add_argument('--leaves',type=int,choices=[7,15]);train.add_argument('--penalty',type=float,choices=[1.,2.,4.])
    train.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if getattr(args,'bootstrap',0)<0:parser.error('bootstrap must be nonnegative')
    if getattr(args,'threshold',None) is not None and args.model_dir is None:parser.error('--threshold requires --model-dir')
    if args.command=='fit' and args.without_source and not args.source_evidence:parser.error('--without-source requires --source-evidence')
    if args.command=='fit' and args.regressor=='mlp' and (args.source_evidence or args.without_age_horizon):parser.error('The MLP control uses the complete 20-descriptor input')
    manifest=verify_manifest(args.artifacts)
    with threadpool_limits(limits=1):
        if args.command=='verify':print(json.dumps(dict(run=manifest['run'],verified_files=len(manifest['sha256']))))
        elif args.command=='evaluate':run_evaluation(args,manifest)
        else:train_selector(args,manifest)

if __name__=='__main__':main()
