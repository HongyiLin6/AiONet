import copy,math
import numpy as np
from aionet.calibration import choose_threshold
from aionet.metrics import aggregate
from aionet.replay import choose,commit

def test_ap_from_ranked_detections_and_harm_denominators():
    frame=dict(counts=np.array([1,0,0,0,0,0,0]),scores=np.array([.9,.8]),labels=np.array([0,0]),tp=1,fp=1,fn=0,loss=.3)
    frame[.25]=frame[.5]=np.array([True,False])
    second=copy.deepcopy(frame);second['scores']=np.array([.7,.6])
    out=aggregate([frame,second],[dict(accepted=True,harm=True,miss=False,positive_increment=.2,late_branch=False),
                               dict(accepted=False,harm=False,miss=False,positive_increment=0.,late_branch=False)],['s1','s2'],0)['metrics']
    assert np.isclose(out['AP25']['value'],5/6)
    assert out['C_coop']['value']==.5
    assert out['H_all']['value']==.5
    assert out['H_acc']['value']==1.

def test_calibration_respects_gain_and_coverage_before_harm_fallback():
    ref={'A0':{'AP25':.1},'A1':{'AP25':.2,'H_all':.1}}
    rows=[dict(AP25=.1,C_coop=0.,H_all=0.,threshold=3.),dict(AP25=.195,C_coop=.8,H_all=.09,threshold=2.),
          dict(AP25=.196,C_coop=.9,H_all=.1,threshold=1.)]
    result=choose_threshold(rows,ref)
    assert result['chosen']['threshold']==2.
    assert result['mode']=='gain_coverage'

def test_late_cooperation_preserves_early_ego():
    row=dict(decision_us=40000,deadline_us=100000,descriptor_seconds=.001,
        candidates={'O':dict(action='O',prediction_steps=0,seconds=.08,observation_us=0,x=[0]*20)})
    settings=dict(branch_cost_p99_seconds={'O':{'0':.04}},descriptor_cost_p99_seconds=.001,selector_cost_p99_seconds=.001)
    assert choose(row,'selector',settings,{(0,'O'):.5},0,.1)=='O'
    assert commit(row,'O',settings)=='0'
    row['decision_us']=100001
    assert commit(row,'0',settings) is None

def test_source_statistics_cannot_arrive_after_decision():
    row=dict(decision_us=40000,deadline_us=100000,descriptor_seconds=.001,
        candidates={'O':dict(action='O',prediction_steps=0,seconds=.01,_metadata_usable_us=40001)})
    settings=dict(branch_cost_p99_seconds={'O':{'0':.01}},descriptor_cost_p99_seconds=.001,selector_cost_p99_seconds=.001,uses_source_metadata=True)
    assert choose(row,'selector',settings,{(0,'O'):.5},0,.1)=='0'
