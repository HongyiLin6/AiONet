"""Generate paired network draws before training; no perception scores."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from reliability_core import delay_trace

def main():
    root=Path(__file__).resolve().parent
    archive=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        rows=json.loads(z.read('V2X-Seq-SPD/infrastructure-side/data_info.json'))
    rows.sort(key=lambda r:(r['sequence_id'],int(r['image_timestamp']),r['frame_id']))
    ids=np.array([r['frame_id'] for r in rows]);scenario=np.array([r['sequence_id'] for r in rows])
    capture=np.array([int(r['image_timestamp']) for r in rows],np.int64)
    files={}
    for seed in (17,):
        for median in (187.3,234.9,258.0):
            delay_us=np.rint(delay_trace(len(rows),seed,median)*1e6).astype(np.int64)
            dest=out/('network_seed%d_median%s.npz'%(seed,str(median).replace('.','p')))
            if dest.exists():
                with np.load(dest,allow_pickle=False) as old:
                    for key,value in {'frame_id':ids,'scenario':scenario,'capture_us':capture,'network_delay_us':delay_us}.items():
                        if not np.array_equal(old[key],value):raise ValueError('Existing frozen trace differs')
            else:np.savez_compressed(dest,frame_id=ids,scenario=scenario,capture_us=capture,network_delay_us=delay_us)
            files[dest.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    report={'kind':'frozen_synthetic_network_traces_not_results','packets_per_trace':len(rows),'seeds':[17],
        'medians_ms':[187.3,234.9,258.0],'deadline_budgets_ms':[50,100,200],
        'native_capture_timestamps_preserved':True,'paired_draws_across_regimes':True,
        'delivery_rule':'source_encoding_ready_us + network_delay_us; encoding readiness must be measured separately',
        'draw_order':'sequence_id,image_timestamp,frame_id','files':files}
    report['active_conditions']={'network_seed17_median187p3.npz':[100],'network_seed17_median234p9.npz':[50,100,200],'network_seed17_median258p0.npz':[100]}
    (out/'trace_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
