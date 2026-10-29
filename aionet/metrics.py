"""All-point AP and paired whole-scenario aggregation."""
import numpy as np

def aggregate(frames,outcomes,scenario_ids,bootstrap):
    """Original AP/loss arithmetic, with retained paired bootstrap values."""
    scenes=sorted(set(scenario_ids));scene_index=np.array([scenes.index(s) for s in scenario_ids]);n=len(scenes)
    rng=np.random.default_rng(20261027);draw=rng.integers(0,n,(bootstrap,n))
    weights=np.array([np.bincount(r,minlength=n) for r in draw],int).reshape(bootstrap,n)
    weights=np.vstack((np.ones(n,int),weights))
    counts=np.zeros((n,7));totals=np.zeros((n,10))
    for frame,outcome,sid in zip(frames,outcomes,scene_index):
        counts[sid]+=frame['counts']
        totals[sid]+=[1,frame['tp'],frame['fp'],frame['fn'],outcome['accepted'],outcome['harm'],outcome['miss'],outcome['positive_increment'],outcome['late_branch'],frame['loss']]
    aggregate_counts=weights@counts;values={}
    for th in [.25,.5]:
        per_class=[]
        for label in range(7):
            ss=[];tt=[];ii=[]
            for frame,sid in zip(frames,scene_index):
                mask=frame['labels']==label;ss.extend(frame['scores'][mask]);tt.extend(frame[th][mask]);ii.extend([sid]*int(mask.sum()))
            scores=np.asarray(ss);tp=np.asarray(tt,float);ids=np.asarray(ii,int)
            order=np.argsort(-scores,kind='stable');tp=tp[order];ids=ids[order]
            ap=[]
            for w,total in zip(weights,aggregate_counts[:,label]):
                if total==0:ap.append(np.nan);continue
                mass=w[ids];keep=mass>0;t=np.cumsum(tp[keep]*mass[keep]);f=np.cumsum((1-tp[keep])*mass[keep])
                rr=np.r_[0.,t/total,1.];pp=np.r_[0.,t/np.maximum(1,t+f),0.];pp=np.maximum.accumulate(pp[::-1])[::-1]
                ix=np.flatnonzero(rr[1:]!=rr[:-1]);ap.append(float(((rr[ix+1]-rr[ix])*pp[ix+1]).sum()))
            per_class.append(ap)
        values['AP'+str(int(th*100))]=np.nanmean(np.array(per_class),axis=0)
    v=weights@totals;den=np.maximum(v[:,0],1);accepted=v[:,4]
    values.update(C_coop=accepted/den,H_all=v[:,5]/den,H_acc=np.divide(v[:,5],accepted,out=np.full(len(accepted),np.nan),where=accepted>0),
        output_deadline_miss=v[:,6]/den,mean_positive_increment=v[:,7]/den,late_branch_rate=v[:,8]/den,
        mean_loss=v[:,9]/den,recall25=v[:,1]/np.maximum(v[:,1]+v[:,3],1),fp_per_frame=v[:,2]/den)
    output={}
    for key,value in values.items():
        samples=value[1:];finite=samples[np.isfinite(samples)]
        output[key]=dict(value=float(value[0]) if np.isfinite(value[0]) else None,
            ci95=[float(x) for x in np.quantile(finite,[.025,.975])] if len(finite) else None,
            defined_bootstrap_samples=int(len(finite)))
    return dict(metrics=output,scenario_count=n,frame_trace_instances=len(frames),bootstrap_samples=bootstrap,
        resampling='paired whole-scenario resampling; all prespecified traces retained per scenario; AP pooled over trace instances',
        bootstrap_values={k:[float(x) if np.isfinite(x) else None for x in value[1:]] for k,value in values.items()})
