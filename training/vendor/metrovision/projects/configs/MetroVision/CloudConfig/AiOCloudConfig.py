from .VehicleConfig.BEVFormerConfig import data as vehicleDataset
from .VehicleConfig.BEVFormerConfig import model as vehicleModel
from .InfrastructureConfig.BEVFormerConfig import data as infrastructureDataset
from .InfrastructureConfig.BEVFormerConfig import model as infrastructureModel

from projects.configs.MetroVision.base_cfg import point_cloud_range

dataset_type = 'AiOCloudDataset'
latency = 1
withPAP = True
withVeh = False
queueLength = 3

model = dict(
    type='AiOCloudNet',
    infrastructureSide=infrastructureModel,
    vehicleSide=vehicleModel,
    BEVFuser=dict(
        type='BEVFuser',
        bevDim=256,
        fusionType='concat',
        point_cloud_range=point_cloud_range),
    queryFuser=dict(
        type='QueryFuser',
        queryDim=256),
    bboxFuser=dict(
        type='BboxFuser',
        iouThreshold=0.5),
    vehiclePthFile='/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_full_veh_epoch_13.pth',
    infrastructurePthFile='/home/omnisky/Desktop/WJN/metrovision/pth/bevformer_small_dair_full_inf_epoch_14.pth',
    # vehiclePthFile='/home/omnisky/Desktop/WJN/metrovision/pth/petr_dair_full_veh_epoch_24.pth',
    # infrastructurePthFile='/home/omnisky/Desktop/WJN/metrovision/pth/petr_dair_full_inf_epoch_21.pth',
    withPAP=withPAP,
    withVeh=withVeh,
    PAP=dict(
        type='PAP',
        latency=latency))

data = dict(
    train=dict(
        type=dataset_type,
        vehicleDataset=vehicleDataset['train'],
        infrastructureDataset=infrastructureDataset['train'],
        latency=latency,
        withPAP=withPAP,
        withVeh=withVeh,
        queueLength=queueLength),
    val=dict(
        type=dataset_type,
        vehicleDataset=vehicleDataset['val'],
        infrastructureDataset=infrastructureDataset['val'],
        latency=latency,
        withPAP=withPAP,
        withVeh=withVeh,
        queueLength=queueLength),
    test=dict(
        type=dataset_type,
        vehicleDataset=vehicleDataset['test'],
        infrastructureDataset=infrastructureDataset['test'],
        latency=latency,
        withPAP=withPAP,
        withVeh=withVeh,
        queueLength=queueLength))
