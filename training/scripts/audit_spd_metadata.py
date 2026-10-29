"""Freeze the new-study partitions and audit real SPD metadata before training.

The original manuscript's 57/10 component-study split is retained as provenance.
Only the new study subdivides its 57 non-test scenarios. No predictions or test
performance enter the allocation. This does not assert exchangeability of sites.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile

TEST = ['0000','0080','0023','0087','0054','0034','0088','0049','0063','0025']
SEED = 'aionet-r2-20261027-v1'
COUNTS = [('selector_fit',8),('development',8),('calibration',9)]

def quantiles(values):
    a=sorted(values)
    if not a: return None
    def q(p):
        x=(len(a)-1)*p; i=int(x); j=min(i+1,len(a)-1)
        return a[i]+(a[j]-a[i])*(x-i)
    return {'count':len(a),'min':a[0],'p05':q(.05),'median':q(.5),'p95':q(.95),'max':a[-1]}

def immutable_json(path, value):
    text=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise RuntimeError('Refusing to overwrite a different frozen manifest: '+str(path))
    path.write_text(text,encoding='utf-8')

def audit(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    src=Path(args.archive); digest=hashlib.sha256()
    with src.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    with zipfile.ZipFile(src) as z:
        names=set(z.namelist()); base='V2X-Seq-SPD/'
        for name in names:
            p=PurePosixPath(name.replace('\\','/'))
            if p.is_absolute() or '..' in p.parts or ':' in p.parts[0]:raise ValueError('Unsafe archive path')
        by_side={s:json.loads(z.read(base+s+'/data_info.json')) for s in ['vehicle-side','infrastructure-side','cooperative']}
        idx={s:{r['frame_id']:r for r in by_side[s]} for s in ['vehicle-side','infrastructure-side']}
        for side in idx:
            if len(idx[side])!=len(by_side[side]):raise ValueError('Duplicate frame ID: '+side)
        pairs=by_side['cooperative']; seen=set(); scenario_pairs=collections.defaultdict(list)
        for r in pairs:
            key=(r['vehicle_frame'],r['infrastructure_frame'])
            if key in seen:raise ValueError('Duplicate cooperative pair')
            seen.add(key)
            v=idx['vehicle-side'][r['vehicle_frame']]; i=idx['infrastructure-side'][r['infrastructure_frame']]
            if not (v['sequence_id']==i['sequence_id']==r['vehicle_sequence']==r['infrastructure_sequence']):raise ValueError('Cross-scenario pair')
            if base+'cooperative/label/'+r['vehicle_frame']+'.json' not in names:raise ValueError('Missing cooperative annotation')
            scenario_pairs[r['vehicle_sequence']].append(r)
        all_ids=sorted(scenario_pairs)
        if not set(TEST)<=set(all_ids):raise ValueError('Original manuscript test scenarios absent')
        ranked=sorted(set(all_ids)-set(TEST),key=lambda s:hashlib.sha256((SEED+':'+s).encode()).hexdigest())
        split={}; at=0
        for part,n in COUNTS:
            split[part]=sorted(ranked[at:at+n]); at+=n
        split['perception_train']=sorted(ranked[at:]);split['test']=sorted(TEST)
        if len(split['perception_train'])!=32:raise ValueError('Unexpected public scenario count; inspect before freezing')
        immutable_json(out/'split.json',split)
        records=[]; capture_diffs=[]; intervals={}; timing={}; class_counts={}; missing=[]
        for side,rows in by_side.items():
            if side=='cooperative':continue
            seq=collections.defaultdict(list)
            for r in rows:
                seq[r['sequence_id']].append(r)
                for key,value in r.items():
                    if (key.startswith('calib_') or key.startswith('label_')) and key.endswith('_path') and base+side+'/'+value not in names:
                        missing.append(side+'/'+value)
            gaps=[]; duplicate_times=[]
            for sid,rs in seq.items():
                ts=sorted(int(r['image_timestamp']) for r in rs)
                dt=[(b-a)/1000 for a,b in zip(ts,ts[1:])];gaps.extend(dt)
                if any(v<=0 for v in dt):duplicate_times.append(sid)
            intervals[side]={'image_interval_ms':quantiles(gaps),'nonpositive_intervals_scenarios':duplicate_times,'gaps_above_150ms':sum(x>150 for x in gaps)}
            timing[side]={'image_minus_lidar_ms':quantiles([(int(r['image_timestamp'])-int(r['pointcloud_timestamp']))/1000 for r in rows])}
        if missing:raise ValueError('Missing calibration/label files: '+str(missing[:5]))
        total_classes=collections.Counter()
        for sid in all_ids:
            ps=scenario_pairs[sid]; locs=set(); classes=collections.Counter()
            for p in ps:
                v=idx['vehicle-side'][p['vehicle_frame']];i=idx['infrastructure-side'][p['infrastructure_frame']]
                locs.add(i['intersection_loc'])
                capture_diffs.append((int(i['image_timestamp'])-int(v['image_timestamp']))/1000)
                ann=json.loads(z.read(base+'cooperative/label/'+p['vehicle_frame']+'.json'))
                for a in ann:
                    classes[a['type']]+=1
                    dims=a['3d_dimensions'];loc=a['3d_location']
                    if any(not math.isfinite(float(v)) for v in list(dims.values())+list(loc.values())+[a['rotation']]):raise ValueError('Nonfinite annotation')
                    if min(float(v) for v in dims.values())<=0:raise ValueError('Invalid box dimensions')
            total_classes.update(classes)
            records.append({'scenario':sid,'cooperative_frames':len(ps),'infrastructure_sites':sorted(locs),'cooperative_class_counts':dict(sorted(classes.items()))})
        summary={}
        for part,ids in split.items():
            summary[part]={'scenarios':len(ids),'cooperative_frames':sum(len(scenario_pairs[s]) for s in ids),
                'vehicle_images':sum(r['sequence_id'] in ids for r in by_side['vehicle-side']),
                'infrastructure_images':sum(r['sequence_id'] in ids for r in by_side['infrastructure-side']),
                'sites':sorted({loc for r in records if r['scenario'] in ids for loc in r['infrastructure_sites']})}
        original_test=sum(len(scenario_pairs[s]) for s in TEST)
        report={'metadata_sha256':digest.hexdigest(),'metadata_bytes':src.stat().st_size,
            'public_scenarios':len(all_ids),'cooperative_frames':len(pairs),
            'vehicle_images_expected':len(by_side['vehicle-side']),'infrastructure_images_expected':len(by_side['infrastructure-side']),
            'images_in_metadata_archive':sum(n.lower().endswith(('.jpg','.png','.jpeg')) for n in names),
            'original_manuscript_split':{'training_scenarios':57,'training_cooperative_frames':len(pairs)-original_test,'test_scenarios':10,'test_cooperative_frames':original_test},
            'new_study_partitions':summary,'allocation':{'seed_string':SEED,'method':'SHA-256 rank of non-test scenario IDs; first 8 selector fit, next 8 development, next 9 calibration, remaining 32 perception train'},
            'native_capture_offset_infrastructure_minus_vehicle_ms':quantiles(capture_diffs),
            'source_timing':timing,'source_intervals':intervals,
            'infrastructure_sites':sorted({loc for r in records for loc in r['infrastructure_sites']}),
            'cooperative_class_counts':dict(sorted(total_classes.items())),
            'caveats':['Public train/val metadata covers 67 of the 95 published scenarios.','Site overlap exists between partitions; scenario independence or exchangeability is not established by disjoint IDs.','Vehicle count cannot be inferred from frame IDs; do not claim collection-vehicle diversity.','Cooperative labels use vehicle LiDAR timestamps; retain and report native camera-to-label offsets.','Frame IDs are not time. Preserve actual microsecond timestamps.'],
            'scenarios':records}
        immutable_json(out/'metadata_audit.json',report)
        print(json.dumps({k:report[k] for k in ['metadata_sha256','public_scenarios','cooperative_frames','original_manuscript_split','new_study_partitions','infrastructure_sites','cooperative_class_counts','native_capture_offset_infrastructure_minus_vehicle_ms']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    audit(p.parse_args())
