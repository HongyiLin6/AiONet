"""Development-derived threshold grid and empirical operating-point choice."""
import numpy as np
from .replay import scores_for, feasible, evaluate


def choose_threshold(grid,reference,gain_retention=.9,min_coverage=.5,harm_ratio=.8):
    gain=reference['A1']['AP25']-reference['A0']['AP25']
    floor=reference['A0']['AP25']+gain_retention*max(0.,gain)
    ceiling=harm_ratio*reference['A1']['H_all']
    quality=[m for m in grid if m['AP25']>=floor and m['C_coop']>=min_coverage]
    all_targets=[m for m in quality if m['H_all']<=ceiling]
    if all_targets:
        chosen=min(all_targets,key=lambda m:(-m['AP25'],m['H_all'],-m['C_coop'],m['threshold']))
        mode='gain_coverage_harm'
    elif quality:
        chosen=min(quality,key=lambda m:(m['H_all'],-m['AP25'],-m['C_coop'],m['threshold']))
        mode='gain_coverage'
    else:
        coverage=[m for m in grid if m['C_coop']>=min_coverage]
        chosen=min(coverage or grid,key=lambda m:(-m['AP25'],m['H_all'],-m['C_coop'],m['threshold']))
        mode='coverage' if coverage else 'maximum_ap'
    return dict(chosen=chosen,mode=mode,AP25_floor=floor,harm_ceiling=ceiling,
                preferences=dict(gain_retention=gain_retention,min_coverage=min_coverage,harm_ratio=harm_ratio))


def calibrate(model,development,calibration,frames,settings,baseline_settings):
    values=scores_for(model,development)
    best=[]
    for i,row in enumerate(development):
        available=[values[(i,a)] for a,b in row['candidates'].items() if feasible(row,b,settings)]
        if available:best.append(max(available))
    grid_values=sorted(set([-1e6,0.,1e6]+[float(x) for x in np.quantile(best,np.linspace(0,1,41))])) if best else [-1e6,0.,1e6]
    scalar=lambda r:{k:v['value'] for k,v in r['metrics'].items()}
    reference={p:scalar(evaluate(calibration,frames,p,baseline_settings)[0]) for p in ['A0','A1']}
    grid=[]
    for threshold in grid_values:
        row=scalar(evaluate(calibration,frames,'selector',settings,model,threshold)[0])
        row['threshold']=threshold;grid.append(row)
    return dict(choose_threshold(grid,reference),grid=grid,reference=reference,settings=settings,
                threshold_grid_partition='development',threshold_choice_partition='calibration')
