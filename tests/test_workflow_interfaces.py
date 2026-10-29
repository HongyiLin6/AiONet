"""Exercise public workflow commands using small, explicitly synthetic inputs."""
import copy,json,os,pickle,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import pytest
from aionet.replay import sha,write_json,verify_selector

ROOT=Path(__file__).resolve().parents[1]


def command(arguments,success=True):
    env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    value=subprocess.run([sys.executable]+list(map(str,arguments)),cwd=ROOT,env=env,text=True,capture_output=True)
    if success:assert value.returncode==0,value.stdout+'\n'+value.stderr
    else:assert value.returncode!=0
    return value


def toy_spd(tmp):
    data=tmp/'dataset';archive=tmp/'metadata.zip';members={};cooperative=[]
    rigid={'rotation':np.eye(3).tolist(),'translation':[0.,0.,0.]}
    for side,role in [('vehicle-side','vehicle'),('infrastructure-side','infrastructure')]:
        info=[];members[side+'/calib/rigid.json']=rigid
        members[side+'/calib/intrinsic.json']={'cam_K':[10.,0.,4.,0.,10.,4.,0.,0.,1.]}
        for scene in range(1,6):
            for j in range(2):
                frame='%04d%02d'%(scene,j);time=scene*1000000+j*100000
                row=dict(frame_id=frame,sequence_id='%04d'%scene,image_timestamp=str(time),pointcloud_timestamp=str(time-1000),
                    image_path='image/'+frame+'.ppm',pointcloud_path='velodyne/'+frame+'.pcd',label_camera_std_path='label/'+frame+'.json',calib_camera_intrinsic_path='calib/intrinsic.json')
                keys=['calib_lidar_to_camera_path','calib_lidar_to_novatel_path','calib_novatel_to_world_path'] if role=='vehicle' else ['calib_virtuallidar_to_camera_path','calib_virtuallidar_to_world_path']
                row.update({k:'calib/rigid.json' for k in keys});info.append(row)
                labels=[dict(type=cls,track_id=idx,rotation=0.,**{'3d_location':dict(x=10.+j,y=float(idx),z=1.),'3d_dimensions':dict(l=4.,w=2.,h=2.)}) for idx,cls in enumerate(['Car','Cyclist'])]
                members[side+'/label/'+frame+'.json']=labels
                image=data/side/row['image_path'];image.parent.mkdir(parents=True,exist_ok=True);image.write_bytes(b'P6\n8 8\n255\n'+bytes([128])*8*8*3)
                if role=='vehicle':
                    members['cooperative/label/'+frame+'.json']=labels
                    cooperative.append(dict(vehicle_frame=frame,infrastructure_frame=frame,vehicle_sequence='%04d'%scene,system_error_offset=dict(delta_x=.3,delta_y=-.1)))
        members[side+'/data_info.json']=info
    members['cooperative/data_info.json']=cooperative
    (data/'cooperative').mkdir();(data/'cooperative/data_info.json').write_text(json.dumps(cooperative))
    with zipfile.ZipFile(archive,'w') as z:
        for key,value in members.items():z.writestr('V2X-Seq-SPD/'+key,json.dumps(value))
    return data,archive


@pytest.mark.skipif(os.name=='nt',reason='Full dataset launchers target Linux')
def test_perception_and_selection_prepare_commands_share_geometry_not_class_filters(tmp_path):
    data,archive=toy_spd(tmp_path)
    for launcher,name,split,classes in [
        ('paper.py','perception',{'perception_train':['0001','0002','0003','0004'],'test':['0005']},['Car']),
        ('pipeline.py','selection',{p:['%04d'%(i+1)] for i,p in enumerate(['perception_train','selector_fit','development','calibration','test'])},['Car','Truck','Van','Bus','Motorcyclist','Cyclist','Pedestrian'])]:
        work=tmp_path/name
        command(['tools/'+launcher,'--workdir',work,'init','--data-root',data])
        write_json(work/'manifests/split.json',split)
        command(['tools/'+launcher,'--workdir',work,'prepare','--metadata-zip',archive])
        manifest=json.loads((work/'infos/conversion_manifest.json').read_text())
        assert manifest['classes']==classes
        assert manifest['split_sha256']==sha(work/'manifests/split.json')
        with (work/'infos/cooperative_test.pkl').open('rb') as f:bundle=pickle.load(f)
        assert len(bundle['infos'])==2
        for row in bundle['infos']:
            assert list(row['gt_names'])==(['Car'] if name=='perception' else ['Car','Cyclist'])
            np.testing.assert_allclose(row['inf2veh'][:2,3],[.3,-.1])
            assert row['velocity_valid'].all()
        assert manifest['records']['cooperative_test']['boxes']==2*len(bundle['infos'][0]['gt_names'])
        if name=='selection':assert len(json.loads((work/'traces/trace_manifest.json').read_text())['files'])==3


def toy_bundle(root):
    """Construct declared synthetic matching events, never manuscript fixtures."""
    settings=dict(family='F',branch_cost_p99_seconds={'O':{'0':.001}},descriptor_cost_p99_seconds=.0001,
        selector_cost_p99_seconds=.0001,freshness=dict(max_mismatch=2.,prediction_penalty=0.,staleness_penalty=0.))
    exports={};root.mkdir()
    for part in ['selector_fit','development','calibration','test']:
        rows=[]
        for i in range(16):
            x=np.zeros(32);x[0]=.1;x[2]=i/16;x[15]=1;x[18]=1
            loss=.0 if i%4 else .5
            rows.append(dict(scenario=part+str(i%2),frame_id=str(i),condition=[234.9,100],source='toy',frame_us=i*100000,
                decision_us=i*100000+10000,deadline_us=i*100000+100000,descriptor_seconds=.0001,ego_loss=.25,empty_loss=1.,
                candidates={'O':dict(action='O',prediction_steps=0,seconds=.001,loss=loss,delta=loss-.25,observation_us=i*100000-100000,
                    x=x.tolist(),_metadata_usable_us=i*100000)},metric_ids={'0':0,'O':1 if i%4 else 0,'None':2}))
        path=root/(part+'.json');write_json(path,rows)
        exports[part]=dict(rows=path.name,scenarios=[part+'0',part+'1'])
        if part in ('calibration','test'):
            path=root/(part+'_events.npz')
            np.savez_compressed(path,counts=np.array([[1,0,0,0,0,0,0]]*3),tp=[0,1,0],fp=[1,0,0],fn=[1,0,1],loss=[.25,0.,1.],
                offsets=[0,1,2,2],scores=[.9,.9],labels=[0,0],match25=[False,True],match50=[False,True])
            exports[part]['events']=path.name
    plus=copy.deepcopy(settings);plus['uses_source_metadata']=True
    manifest=dict(format_version=1,run='synthetic_interface_test',seed=17,settings={'F':settings},models={},exports={'F':exports},
        fitting_settings={'F':{'R-BH':settings,'R-BH+S':plus}},sha256={p.name:sha(p) for p in root.iterdir()})
    write_json(root/'manifest.json',manifest)


def test_selector_fit_calibrate_evaluate_and_input_binding(tmp_path):
    root=tmp_path/'bundle';toy_bundle(root);model=tmp_path/'selector';output=tmp_path/'evaluated'
    command(['-m','aionet','--artifacts',root,'fit','--family','F','--leaves','7','--penalty','1','--output',model])
    command(['-m','aionet','--artifacts',root,'evaluate','--family','F','--model-dir',model,'--output',output])
    report=json.loads((output/'custom.json').read_text())
    assert report['frame_trace_instances']==16
    scalar={k:v['value'] for k,v in report['metrics'].items()}
    if scalar['C_coop']>0:assert scalar['H_all']==pytest.approx(scalar['H_acc']*scalar['C_coop'])
    verify_selector(model,root,'F')
    with pytest.raises(ValueError,match='family'):verify_selector(model,root,'S')
    path=model/'calibration.json';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='calibration checksum'):verify_selector(model,root,'F')
