import numpy as np
from aionet.paper.experiments import experiments,payloads
from aionet.paper.temporal import propagate_boxes,serialized_arrivals,delay_stratum,source_index

def test_configuration_grid_and_protocol_are_explicit():
    jobs=experiments();grid=[j for j in jobs if j['table']=='2']
    assert len(grid)==36
    assert len({(j['vehicle'],j['source'],j['representation']) for j in grid})==36
    assert all(j['classes']==['Car'] and j['epochs']==24 for j in grid)
    assert all(j['checkpoint_policy']=='final_epoch' for j in jobs)

def test_payload_and_serial_queue_use_bits_and_microseconds():
    assert payloads()['registered_bev']['bytes_per_frame']==11520000
    assert payloads()['bev_query']['bytes_per_frame']==128*128*64+900*256*2
    np.testing.assert_allclose(serialized_arrivals([0,100000],1000000,10),[800000,1600000])

def test_temporal_lookup_and_motion_do_not_change_inputs():
    assert source_index([0,100000,200000],250000,100)==1
    b=np.array([[1.,2,0,4,2,1,0]]);before=b.copy()
    np.testing.assert_allclose(propagate_boxes(b,[[2.,-1]],.5)[0,:2],[2,1.5]);np.testing.assert_array_equal(b,before)
    assert [delay_stratum(x) for x in [187.3,234.9,250]]==['Low','Mid','High']
