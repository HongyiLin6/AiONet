"""Selector inputs derived exclusively from causal inputs and ego predictions."""
import numpy as np
from .reliability_core import FEATURE_NAMES

def make_descriptor(snapshot,action,ego,score_threshold,deadline_us,temporal_change=None,fov_overlap=None):
    if snapshot.latest is None or action not in ('O','P'):raise ValueError('No eligible candidate')
    t=snapshot.frame_time_us;o=snapshot.latest.observation_us
    v=t if action=='P' else o
    selected=ego.roi();selected=selected.select(selected.scores>=score_threshold)
    scores=selected.scores;n=len(scores)
    speed=None
    if selected.velocity is not None and n:
        valid=np.isfinite(selected.velocity).all(1)
        if valid.any():speed=np.linalg.norm(selected.velocity[valid],axis=1)
    if temporal_change is not None and (not np.isfinite(temporal_change) or temporal_change<0):raise ValueError('Invalid temporal change')
    if fov_overlap is not None and not 0<=fov_overlap<=1:raise ValueError('Invalid field-of-view overlap')
    row=np.array([(t-o)/1e6,(v-o)/1e6,(t-v)/1e6,(deadline_us-snapshot.decision_time_us)/1e6,
        n,float(scores.mean()) if n else 0.,float(scores.std()) if n else 0.,
        np.median(speed) if speed is not None else np.nan,np.quantile(speed,.9) if speed is not None else np.nan,
        fov_overlap if fov_overlap is not None else np.nan,
        temporal_change if temporal_change is not None else np.nan,len(snapshot.real),
        temporal_change is None,speed is None,fov_overlap is None,
        snapshot.family=='F',snapshot.family=='Q',snapshot.family=='S',action=='O',action=='P'],np.float64)
    assert len(row)==len(FEATURE_NAMES)==20
    return row
