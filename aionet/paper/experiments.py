"""Experiment definitions derived from the manuscript, without result values."""
from itertools import product

MODELS=('bevformer','petr','detr3d')
OPERATORS=('query_concat','box_mlp','bev_concat','bev_attention','bev_query','bev_conv','adaptive')
REPRESENTATIONS={'boxes':'box_mlp','bev':'bev_query','query':'query_concat','bev_query':'adaptive'}

def experiments():
    jobs=[]
    def add(name,table,vehicle='bevformer',source='bevformer',operator='adaptive',representation='bev_query',**kwargs):
        jobs.append(dict(name=name,table=table,vehicle=vehicle,source=source,operator=operator,representation=representation,
            classes=['Car'],seed=17,epochs=24,batch_size=4,learning_rate=1e-4,delay_ms=0.,checkpoint_policy='final_epoch',**kwargs))
    for op in ('vehicle_only','infrastructure_only')+OPERATORS:add('fusion_'+op,'1a',operator=op)
    for model in ('bevformer','petr'):
        add('detector_vehicle_'+model,'1b',vehicle=model,operator='vehicle_only')
        add('detector_infrastructure_'+model,'1b',source=model,operator='infrastructure_only')
    for vehicle,source in [('bevformer','bevformer'),('petr','petr'),('petr','bevformer')]:
        add('reference_'+vehicle+'_'+source,'1b',vehicle,source,operator='bev_concat',baseline_definition='explicit reconstruction: concatenate aligned BEV fields')
    add('aionet_petr_bevformer','1b','petr','bevformer')
    for vehicle,source,representation in product(MODELS,MODELS,REPRESENTATIONS):
        add('grid_'+vehicle+'_'+source+'_'+representation,'2',vehicle,source,REPRESENTATIONS[representation],representation)
    for strategy in ['vehicle_only','infrastructure_only','observed','fixed_delay','queue','constant_velocity','temposense']:
        add('temporal_'+strategy,'3b',temporal=strategy)
        if strategy=='fixed_delay':jobs[-1]['delay_ms']=36.4+71.47+81.23
    return jobs

def get_experiment(name):
    found=[j for j in experiments() if j['name']==name]
    if len(found)!=1:raise ValueError('Unknown experiment: '+name)
    return found[0]

def payloads():
    examples={'boxes':[(100,25,16)],'query':[(900,256,16)],'bev_int8':[(128,128,64,8)],
              'bev_fp16':[(128,128,64,16)],'bev_query':[(128,128,64,8),(900,256,16)],'registered_bev':[(150,150,256,16)]}
    result={}
    for name,parts in examples.items():
        size=0
        for p in parts:
            count=1
            for d in p[:-1]:count*=d
            size+=count*p[-1]/8
        result[name]=dict(bytes_per_frame=size,mebibytes=size/2**20,mbit_s_at_10hz=8*10*size/1e6)
    return result
