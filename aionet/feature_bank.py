"""Causal Feature Bank; times are integer microseconds on a common clock.

This module stores immutable references, not detector state or ground truth.
Prediction construction consumes a snapshot and never inserts forecasts into
the real-observation history.
"""
from dataclasses import dataclass
import hashlib
import math
from typing import Tuple

@dataclass(frozen=True)
class Observation:
    packet_id:str
    scenario:str
    source:str
    family:str
    observation_us:int
    arrival_us:int
    usable_us:int
    payload_ref:str
    geometry_ref:str

    def __post_init__(self):
        if self.family not in ('F','Q','S'):raise ValueError('Unknown representation')
        if not self.observation_us<=self.arrival_us<=self.usable_us:raise ValueError('Noncausal packet')
        if not self.packet_id or not self.payload_ref or not self.geometry_ref:raise ValueError('Incomplete provenance')

@dataclass(frozen=True)
class HistorySnapshot:
    scenario:str
    source:str
    family:str
    frame_time_us:int
    decision_time_us:int
    real:Tuple[Observation,...]

    @property
    def latest(self):return self.real[-1] if self.real else None
    @property
    def fingerprint(self):
        text='|'.join([self.scenario,self.source,self.family,str(self.frame_time_us),str(self.decision_time_us)]+[r.packet_id+':'+r.payload_ref for r in self.real])
        return hashlib.sha256(text.encode()).hexdigest()
    def prediction_history(self,length=3,max_gap_us=150000,max_horizon_us=2000000,max_steps=20):
        if len(self.real)<length:return ()
        selected=self.real[-length:]
        gaps=[b.observation_us-a.observation_us for a,b in zip(selected,selected[1:])]
        horizon=self.frame_time_us-selected[-1].observation_us
        if any(g<=0 or g>max_gap_us for g in gaps) or not 0<horizon<=max_horizon_us:return ()
        cadence=(selected[-1].observation_us-selected[0].observation_us)/(length-1)
        if horizon/cadence>max_steps:return ()
        return selected

class FeatureBank:
    def __init__(self,capacity_per_source=64):
        if capacity_per_source<3:raise ValueError('At least three real history slots required')
        self.capacity=capacity_per_source;self.scenario=None;self._records={};self._seen={};self.now_us=None
    def reset(self,scenario):
        self.scenario=scenario;self._records={};self._seen={};self.now_us=None
    def advance(self,decision_us,arrivals=()):
        if self.scenario is None:raise ValueError('Reset at scenario start')
        if self.now_us is not None and decision_us<self.now_us:raise ValueError('Decision clock moved backwards')
        for row in arrivals:
            if row.scenario!=self.scenario:raise ValueError('Cross-scenario history')
            if row.usable_us>decision_us:raise ValueError('Message is not usable yet')
            if row.packet_id in self._seen:
                if self._seen[row.packet_id]!=row:raise ValueError('Packet ID reused for changed content')
                continue
            self._seen[row.packet_id]=row
            key=(row.source,row.family);values=self._records.setdefault(key,[])
            values.append(row);values.sort(key=lambda p:(p.observation_us,p.packet_id))
            del values[:-self.capacity]
        self.now_us=decision_us
    def snapshot(self,source,family,frame_time_us):
        if self.now_us is None or frame_time_us>self.now_us:raise ValueError('Ego frame not acquired at decision time')
        rows=tuple(r for r in self._records.get((source,family),()) if r.observation_us<=frame_time_us and r.usable_us<=self.now_us)
        return HistorySnapshot(self.scenario,source,family,frame_time_us,self.now_us,rows)

def serial_source_ready(capture_us,duration_us):
    """Single source-device queue; network delay begins after encoding."""
    if len(capture_us)!=len(duration_us):raise ValueError('Timing arrays differ')
    ready=[];previous=None
    for capture,duration in zip(capture_us,duration_us):
        if duration<0 or not math.isfinite(duration):raise ValueError('Invalid measured duration')
        start=max(capture,previous) if previous is not None else capture
        previous=int(math.ceil(start+duration));ready.append(previous)
    return ready
