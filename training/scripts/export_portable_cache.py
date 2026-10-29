"""Export real branch predictions and matching events to a portable CPU bundle."""
import argparse,copy,gzip,json,math,time
from pathlib import Path
import numpy as np
from experiment_io import ROOT,DetectorCache,records,atomic_json,sha
from evaluate_policy_cache import CompiledMetrics

CLASSES=['Car','Truck','Van','Bus','Motorcyclist','Cyclist','Pedestrian']

def packet_summary(scores,labels):
    scores=np.asarray(scores,float);labels=np.asarray(labels,int)
    ordered=np.sort(scores)[::-1];top=ordered[:10]
    hist=np.bincount(labels[scores>=.1],minlength=7).astype(float)
    hist=hist/hist.sum() if hist.sum() else hist
    entropy=-np.sum(hist[hist>0]*np.log(hist[hist>0]))/np.log(7)
    return np.asarray([np.sum(scores>=.1),np.sum(scores>=.3),np.sum(scores>=.5),
        float(ordered[0]) if len(ordered) else 0.,float(top.mean()) if len(top) else 0.,
        float(top.std()) if len(top) else 0.,*list(np.quantile(scores,[.5,.9,.99]) if len(scores) else np.zeros(3)),
        entropy,float(len(scores)==0),float(ordered[0]-ordered[1]) if len(ordered)>1 else 0.],np.float32)

def metadata_timing(family):
    path=ROOT/'runs/choices'/(family+'_source_metadata_timing.json')
    if path.exists():return json.loads(path.read_text())
    cache=DetectorCache('bevformer_infrastructure','development');arrays=[]
    for i in np.random.default_rng(17).integers(0,len(cache.rows),138):
        with np.load(cache.root/cache.rows[i]['detections'],allow_pickle=False) as d:arrays.append((d['scores'],d['labels']))
    generation=[];unpack=[]
    for i,(s,l) in enumerate(arrays):
        start=time.perf_counter();payload=packet_summary(s,l).tobytes();elapsed=time.perf_counter()-start
        start=time.perf_counter();np.frombuffer(payload,dtype=np.float32).copy();read=time.perf_counter()-start
        if i>=10:generation.append(elapsed);unpack.append(read)
    value=dict(generation_p99_seconds=float(np.quantile(generation,.99)),unpack_p99_seconds=float(np.quantile(unpack,.99)),
        partition='development',summary_bytes_per_packet=48,samples=128,warmup=10)
    atomic_json(path,value);return value

def compact(family,partition,destination,timing):
    names=['bevformer_infrastructure'] if family=='F' else (['detr3d_infrastructure'] if partition=='test' else ['bevformer_infrastructure','petr_infrastructure'])
    rows=[];frames=[];indices={};source_hashes={}
    ground_truth={r['frame_id']:r for r in records('cooperative',partition)} if partition in ('calibration','test') else None
    for name in names:
        folder=ROOT/'cache/branches'/family/name/partition
        meta=json.loads((folder/'manifest.json').read_text());source_hashes[name]=sha(folder/'manifest.json')
        compiler=CompiledMetrics(meta['score_threshold']);cache=DetectorCache(name,partition);statistics={}
        for line in (folder/'frames.jsonl').open():
            row=json.loads(line)
            r={k:row[k] for k in ['scenario','frame_id','budget_ms','frame_us','decision_us','deadline_us','descriptor_seconds','ego_loss','empty_loss','trace']}
            r.update(condition=[float(row['trace'].split('_median')[1].split('.npz')[0].replace('p','.')),row['budget_ms']],source=name,candidates={})
            ready={item['packet_id']:item['usable_us'] for item in row['history_records']}
            for action,b in row['candidates'].items():
                token=b['real_history'][-1]
                if token not in statistics:
                    with np.load(cache.root/cache.by_token[token]['detections'],allow_pickle=False) as d:statistics[token]=packet_summary(d['scores'],d['labels']).tolist()
                assert len(b['x'])==20
                candidate={k:b[k] for k in ['action','prediction_steps','seconds','loss','delta','observation_us']}
                candidate.update(x=b['x']+statistics[token],_metadata_usable_us=ready[token]+math.ceil(timing['generation_p99_seconds']*1e6))
                r['candidates'][action]=candidate
            if ground_truth is not None:
                r['metric_ids']={}
                for action in ['0',None]+list(row['candidates']):
                    file=None if action is None else Path(row['ego_file']) if action=='0' else folder/row['candidates'][action]['file']
                    key=(str(file),row['frame_id'])
                    if key not in indices:
                        indices[key]=len(frames);frames.append(compiler.frame(file,ground_truth[row['frame_id']]))
                    r['metric_ids'][str(action)]=indices[key]
            rows.append(r)
        print(family,name,partition,len(rows),flush=True)
    target=destination/family/(partition+'.json.gz');target.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(target,'wt',encoding='utf-8') as stream:json.dump(rows,stream,separators=(',',':'),allow_nan=False)
    result=dict(rows=str(target.relative_to(destination)),count=len(rows),scenarios=sorted(set(r['scenario'] for r in rows)),source_manifest_sha256=source_hashes)
    if frames:
        target=destination/family/(partition+'_events.npz')
        arrays={k:np.asarray([r[k] for r in frames]) for k in ['counts','tp','fp','fn','loss']}
        arrays.update(offsets=np.r_[0,np.cumsum([len(r['scores']) for r in frames])],scores=np.concatenate([r['scores'] for r in frames]),
            labels=np.concatenate([r['labels'] for r in frames]),match25=np.concatenate([r[.25] for r in frames]),match50=np.concatenate([r[.5] for r in frames]))
        np.savez_compressed(target,**arrays);result.update(events=str(target.relative_to(destination)),event_frames=len(frames))
    return result,sorted(set(tuple(r['condition']) for r in rows))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',choices=['F','S'],required=True)
    p.add_argument('--partitions',nargs='+',choices=['selector_fit','development','calibration','test'],default=['selector_fit','development','calibration'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--run-name',default='spd_local_run');args=p.parse_args()
    output=args.output.resolve();path=output/'manifest.json';family=args.family
    if 'test' in args.partitions:
        from freeze_evaluation_choices import verify_frozen
        verify_frozen()
    split=json.loads((ROOT/'manifests/split.json').read_text())
    report=json.loads(path.read_text()) if path.exists() else dict(format_version=1,run=args.run_name,seed=17,classes=CLASSES,exports={},models={},settings={},fitting_settings={},sha256={},expected={},conditions=[],source_checkpoints={})
    if report['run']!=args.run_name:raise ValueError('The output belongs to another named run')
    report['exports'].setdefault(family,{'test':{'scenarios':split['test']}})
    settings=json.loads((ROOT/'runs/choices'/(family+'_execution.json')).read_text())
    settings={k:settings[k] for k in ['family','branch_cost_p99_seconds','descriptor_cost_p99_seconds','selector_cost_p99_seconds','freshness']}
    timing=metadata_timing(family);plus=copy.deepcopy(settings)
    plus['uses_source_metadata']=True;plus['descriptor_cost_p99_seconds']+=timing['unpack_p99_seconds']
    report['settings'][family]=settings;report['fitting_settings'][family]={'R-BH':settings,'R-BH+S':plus}
    report.setdefault('metadata_timing',{})[family]=timing
    for partition in args.partitions:
        if 'rows' in report['exports'][family].get(partition,{}):raise ValueError('Partition already exported; use a new output for changed inputs')
        info,conditions=compact(family,partition,output,timing);report['exports'][family][partition]=info
        report['conditions']=[list(x) for x in sorted(set(map(tuple,report['conditions']))|set(conditions))]
    report['sha256']={str(f.relative_to(output)):sha(f) for f in output.rglob('*') if f.is_file() and f.name!='manifest.json'}
    report['split_sha256']=sha(ROOT/'manifests/split.json')
    atomic_json(path,report);print(path)

if __name__=='__main__':main()
