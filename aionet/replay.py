"""Policy decisions, causal availability and deadline commitment."""
import copy, gzip, hashlib, json, math
from pathlib import Path
import numpy as np
from .selector import score


def read_json(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as f:
        return json.load(f)


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_manifest(root):
    root=Path(root)
    manifest=read_json(root/'manifest.json')
    for relative,expected in manifest['sha256'].items():
        path=(root/relative).resolve()
        path.relative_to(root.resolve())
        if sha(path)!=expected:raise ValueError('Artifact checksum mismatch: '+relative)
    return manifest


def verify_selector(model_dir,root,family):
    """Match a fitted selector to its family and immutable pre-test inputs."""
    model_dir=Path(model_dir);root=Path(root)
    provenance=read_json(model_dir/'provenance.json')
    calibration=read_json(model_dir/'calibration.json')
    declared=provenance.get('family',calibration['settings'].get('family'))
    if declared!=family:raise ValueError('Selector family does not match the evaluation family')
    if sha(model_dir/'model.joblib')!=provenance['model_sha256']:raise ValueError('Selector model checksum mismatch')
    if provenance.get('calibration_sha256') and sha(model_dir/'calibration.json')!=provenance['calibration_sha256']:
        raise ValueError('Selector calibration checksum mismatch')
    for relative,expected in provenance.get('fitting_input_sha256',{}).items():
        path=(root/relative).resolve();path.relative_to(root.resolve())
        if sha(path)!=expected:raise ValueError('Selector fitting input changed: '+relative)
    return provenance


def estimated_cost(candidate, settings):
    groups=settings['branch_cost_p99_seconds'].get(candidate['action'],{})
    steps=candidate['prediction_steps']
    if not groups or steps>max(map(int,groups)):return math.inf
    key=next(k for k in sorted(map(int,groups)) if k>=steps)
    return groups[str(key)]+settings['descriptor_cost_p99_seconds']+settings['selector_cost_p99_seconds']


def feasible(row,candidate,settings):
    cost=estimated_cost(candidate,settings)
    return math.isfinite(cost) and row['decision_us']+math.ceil(cost*1e6)<=row['deadline_us']


def commit(row, action, settings, selector_seconds=None):
    if action!='0':
        candidate=row['candidates'][action]
        extra=row['descriptor_seconds']+candidate['seconds']
        extra+=settings['selector_cost_p99_seconds'] if selector_seconds is None else selector_seconds
        if row['decision_us']+math.ceil(extra*1e6)<=row['deadline_us']:return action
    return '0' if row['decision_us']<=row['deadline_us'] else None


def features(rows, dimensions=32):
    values=[];positions=[]
    for i,row in enumerate(rows):
        for action,candidate in row['candidates'].items():
            values.append([np.nan if v is None else v for v in candidate['x'][:dimensions]])
            positions.append((i,action))
    return np.asarray(values,np.float32).reshape(-1,dimensions),positions


def scores_for(model,rows):
    x,positions=features(rows,model['dimensions'])
    return dict(zip(positions,map(float,score(model,x))))


def choose(row,policy,settings,scores=None,index=0,threshold=0.):
    if policy=='A0':return '0'
    if policy in ('A1','A2'):
        action='O' if policy=='A1' else 'P'
        return action if action in row['candidates'] and feasible(row,row['candidates'][action],settings) else '0'
    ranked=[]
    for action,candidate in row['candidates'].items():
        if not feasible(row,candidate,settings):continue
        if policy=='A3':
            params=settings['freshness']
            gap=0. if action=='P' else (row['frame_us']-candidate['observation_us'])/1e6
            if gap>params['max_mismatch']:continue
            utility=-(gap+params['prediction_penalty']*(action=='P')+params['staleness_penalty']*gap)
        else:
            if settings.get('uses_source_metadata') and candidate['_metadata_usable_us']>row['decision_us']:continue
            utility=scores[(index,action)]
        ranked.append((-utility,estimated_cost(candidate,settings),action!='O',action))
    if not ranked:return '0'
    winner=min(ranked)
    return winner[-1] if policy=='A3' or -winner[0]>threshold else '0'


class MetricFrames:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as archive:
            self.data={key:archive[key] for key in archive.files}
    def frame(self,index):
        d=self.data;start,end=d['offsets'][index:index+2]
        result=dict(counts=d['counts'][index],scores=d['scores'][start:end],labels=d['labels'][start:end],
            tp=int(d['tp'][index]),fp=int(d['fp'][index]),fn=int(d['fn'][index]),loss=float(d['loss'][index]),
            **{})
        result.update({0.25:d['match25'][start:end],0.5:d['match50'][start:end]})
        return result


def evaluate(rows, frames, policy, settings, model=None, threshold=0., bootstrap=0):
    from .metrics import aggregate
    scores=scores_for(model,rows) if model is not None else {}
    output=[];outcomes=[];scenes=[];decisions=[]
    for i,row in enumerate(rows):
        attempted=choose(row,policy,settings,scores,i,threshold)
        delivered=commit(row,attempted,settings,0. if policy in ('A0','A1','A2','A3') else None)
        accepted=delivered in ('O','P')
        delta=row['candidates'][delivered]['delta'] if accepted else 0.
        output.append(frames.frame(row['metric_ids'][str(delivered)]));scenes.append(row['scenario'])
        outcomes.append(dict(accepted=accepted,harm=accepted and delta>1e-12,miss=delivered is None,
            positive_increment=max(0.,delta),late_branch=attempted in ('O','P') and delivered!=attempted))
        decisions.append(dict(scenario=row['scenario'],frame_id=row['frame_id'],attempted=attempted,committed=delivered,delta=delta))
    result=aggregate(output,outcomes,scenes,bootstrap)
    return result,decisions
