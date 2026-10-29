"""Causal temporal baselines, delay strata and payload-aware link replay."""
import numpy as np

def lognormal_delays(count,seed=17):
    rng=np.random.default_rng(seed);segments=[]
    for mean,sd in [(36.40,10.92),(71.47,19.55),(81.23,13.48)]:
        sigma=np.sqrt(np.log1p((sd/mean)**2));mu=np.log(mean)-sigma*sigma/2
        segments.append(rng.lognormal(mu,sigma,count))
    return np.column_stack(segments)

def delay_stratum(milliseconds):
    return 'Low' if milliseconds<=187.3 else 'Mid' if milliseconds<=234.9 else 'High'

def source_index(timestamps,target_us,delay_ms):
    return int(np.searchsorted(timestamps,target_us-delay_ms*1000,side='right')-1)

def propagate_boxes(boxes,velocity,age_seconds):
    b=np.array(boxes,copy=True);v=np.asarray(velocity)
    if v.shape!=(len(b),2) or not np.isfinite(v).all():raise ValueError('Constant-velocity propagation requires finite horizontal velocities')
    b[:,:2]+=v*age_seconds;return b

def serialized_arrivals(capture_us,payload_bytes,link_mbit_s,propagation_ms=0.):
    if link_mbit_s<=0:raise ValueError('Link rate must be positive')
    previous=-np.inf;out=[]
    for capture,size in zip(capture_us,np.broadcast_to(payload_bytes,(len(capture_us),))):
        previous=max(previous,float(capture))+float(size)*8/link_mbit_s
        out.append(previous+propagation_ms*1000)
    return np.asarray(out)
