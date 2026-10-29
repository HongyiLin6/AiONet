from mmcv.runner import BaseModule, load_checkpoint
from mmdet3d.models import DETECTORS
from mmdet3d.models import build_model, build_fusion_layer
import torch


@DETECTORS.register_module()
class AiOCloudNet(BaseModule):
    def __init__(self,
                 vehicleSide,
                 infrastructureSide,
                 BEVFuser,
                 queryFuser,
                 bboxFuser,
                 vehiclePthFile=None,
                 infrastructurePthFile=None,
                 withPAP=False,
                 withVeh=False,
                 PAP=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.withVeh = withVeh
        if withVeh:
            self.vehicleSide = build_model(vehicleSide)
            if vehiclePthFile is not None:
                load_checkpoint(self.vehicleSide, vehiclePthFile, map_location='cpu')
                for param in self.vehicleSide.parameters():
                    param.requires_grad = False
        self.infrastructureSide = build_model(infrastructureSide)
        if infrastructurePthFile is not None:
            load_checkpoint(self.infrastructureSide, infrastructurePthFile, map_location='cpu')
            for param in self.infrastructureSide.parameters():
                param.requires_grad = False
        self.BEVFuser = build_fusion_layer(BEVFuser)
        self.queryFuser = build_fusion_layer(queryFuser)
        self.bboxFuser = build_fusion_layer(bboxFuser)
        self.withPAP = withPAP
        self.history = None
        if withPAP:
            self.PAP = build_fusion_layer(PAP)
            for param in self.PAP.parameters():
                param.requires_grad = True

    def init_weights(self):
        if self.withVeh:
            self.vehicleSide.init_weights()
        self.infrastructureSide.init_weights()

    def getBEV(self,
               train=True,
               **kwargs):
        if train:
            return self.getBEV_train(**kwargs)
        else:
            return self.getBEV_test(**kwargs)

    def getQuery(self,
                 train=True,
                 **kwargs):
        if train:
            return self.getQuery_train(**kwargs)
        else:
            return self.getQuery_test(**kwargs)

    def getBbox(self,
                train=True,
                **kwargs):
        if train:
            return self.getBbox_train(**kwargs)
        else:
            return self.getBbox_test(**kwargs)

    def getBEV_train(self,
                     vehicleData,
                     infrastructureData,
                     inf2veh_rotation,
                     inf2veh_translation,
                     **kwargs):
        if self.withPAP:
            if self.withVeh:
                assert len(vehicleData) == len(infrastructureData)
                BEVBank = []
                for i in range(len(vehicleData)):
                    vehData = vehicleData[i]
                    infData = infrastructureData[i]
                    vehBEV = self.vehicleSide.getBEV(**vehData)
                    infBEV = self.infrastructureSide.getBEV(**infData)
                    curFusedBEV = self.BEVFuser(infBEV, vehBEV, inf2veh_rotation, inf2veh_translation)
                    BEVBank.append(curFusedBEV)
                return self.PAP(train=True, featureBank=BEVBank, type='BEV')
            else:
                BEVBank = []
                for i in range(len(infrastructureData)):
                    infData = infrastructureData[i]
                    infBEV = self.infrastructureSide.getBEV(**infData)
                    BEVBank.append(infBEV)
                return self.PAP(train=True, featureBank=BEVBank, type='BEV')
        else:
            if not self.withVeh:
                return self.infrastructureSide.getBEV(**infrastructureData[0])
            vehData = vehicleData[0]
            infData = infrastructureData[0]
            vehBEV = self.vehicleSide.getBEV(**vehData)
            infBEV = self.infrastructureSide.getBEV(**infData)
            return self.BEVFuser(infBEV, vehBEV, inf2veh_rotation, inf2veh_translation)

    def getBEV_test(self,
                    vehicleData,
                    infrastructureData,
                    inf2veh_rotation,
                    inf2veh_translation,
                    **kwargs):
        if self.withPAP:
            if self.withVeh:
                assert len(vehicleData) == len(infrastructureData)
                BEVBank = []
                for i in range(len(vehicleData)):
                    vehData = vehicleData[i]
                    infData = infrastructureData[i]
                    vehBEV = self.vehicleSide.getBEV(train=False, **vehData)
                    infBEV = self.infrastructureSide.getBEV(train=False, **infData)
                    curFusedBEV = self.BEVFuser(infBEV, vehBEV, inf2veh_rotation, inf2veh_translation)
                    BEVBank.append(curFusedBEV)
                return self.PAP(BEVBank, type='BEV')
            else:
                infData = infrastructureData[0]
                infBEV = self.infrastructureSide.getBEV(train=False, **infData)
                return self.PAP(train=False, feature=infBEV, type='BEV')
        else:
            if not self.withVeh:
                return self.infrastructureSide.getBEV(train=False, **infrastructureData[0])
            vehData = vehicleData[0]
            infData = infrastructureData[0]
            vehBEV = self.vehicleSide.getBEV(train=False, **vehData)
            infBEV = self.infrastructureSide.getBEV(train=False, **infData)
            return self.BEVFuser(infBEV, vehBEV, inf2veh_rotation, inf2veh_translation)

    def getQuery_train(self,
                       vehicleData,
                       infrastructureData,
                       inf2veh_rotation,
                       inf2veh_translation,
                       **kwargs):
        if self.withPAP:
            assert len(vehicleData) == len(infrastructureData)
            queryBank = []
            for i in range(len(vehicleData)):
                vehData = vehicleData[i]
                infData = infrastructureData[i]
                vehQuery = self.vehicleSide.getQuery(**vehData)
                infQuery = self.infrastructureSide.getQuery(**infData)
                fusedQuery = self.queryFuser(infQuery, vehQuery, inf2veh_rotation, inf2veh_translation)
                queryBank.append(fusedQuery)
            return self.PAP(queryBank, type='query')
        else:
            if not self.withVeh:
                return self.infrastructureSide.getQuery(**infrastructureData[0])
            vehData = vehicleData[0]
            infData = infrastructureData[0]
            vehQuery = self.vehicleSide.getQuery(**vehData)
            infQuery = self.infrastructureSide.getQuery(**infData)
            return self.queryFuser(infQuery, vehQuery, inf2veh_rotation, inf2veh_translation)

    def getQuery_test(self,
                      vehicleData,
                      infrastructureData,
                      veh2inf_rotation,
                      veh2inf_translation,
                      **kwargs):
        if self.withPAP:
            assert len(vehicleData) == len(infrastructureData)
            queryBank = []
            for i in range(len(vehicleData)):
                vehData = vehicleData[i]
                infData = infrastructureData[i]
                vehQuery = self.vehicleSide.getQuery(train=False, **vehData)
                infQuery = self.infrastructureSide.getQuery(train=False, **infData)
                fusedQuery = self.queryFuser(infQuery, vehQuery, veh2inf_rotation, veh2inf_translation)
                queryBank.append(fusedQuery)
            return self.PAP(queryBank, type='query')
        else:
            if not self.withVeh:
                return self.infrastructureSide.getQuery(train=False, **infrastructureData[0])
            vehData = vehicleData[0]
            infData = infrastructureData[0]
            vehQuery = self.vehicleSide.getQuery(train=False, **vehData)
            infQuery = self.infrastructureSide.getQuery(train=False, **infData)
            return self.queryFuser(infQuery, vehQuery, veh2inf_rotation, veh2inf_translation)

    def getBbox_train(self,
                      vehicleData,
                      infrastructureData,
                      veh2inf_rotation,
                      veh2inf_translation,
                      **kwargs):
        if self.withPAP:
            if self.withVeh:
                assert len(vehicleData) == len(infrastructureData)
                bboxBank = []
                for i in range(len(vehicleData)):
                    vehData = vehicleData[i]
                    infData = infrastructureData[i]
                    vehBbox = self.vehicleSide.getBbox(**vehData)
                    infBbox = self.infrastructureSide.getBbox(**infData)
                    curFusedBEV = self.BEVFuser(infBbox, vehBbox, veh2inf_rotation, veh2inf_translation)
                    bboxBank.append(curFusedBEV)
                return self.PAP(bboxBank, type='bbox')
            else:
                bboxBank = []
                for i in range(len(infrastructureData)):
                    infData = infrastructureData[i]
                    infBbox = self.infrastructureSide.getBbox(**infData)
                    save = infBbox
                    bboxBank.append(infBbox['all_bbox_preds'])
                bbox_pred = self.PAP(train=True, featureBank=bboxBank, type='bbox')
                save['all_bbox_preds'] = bbox_pred
                return save
        else:
            if not self.withVeh:
                return self.infrastructureSide.getBbox(**infrastructureData[0])
            vehData = vehicleData[0]
            infData = infrastructureData[0]
            vehBbox = self.vehicleSide.getBbox(**vehData)
            infBbox = self.infrastructureSide.getBbox(**infData)
            return self.bboxFuser(infBbox, vehBbox, veh2inf_rotation, veh2inf_translation)

    def getBbox_test(self,
                     vehicleData,
                     infrastructureData,
                     veh2inf_rotation,
                     veh2inf_translation,
                     **kwargs):
        if self.withPAP:
            if self.withVeh:
                assert len(vehicleData) == len(infrastructureData)
                bboxBank = []
                for i in range(len(vehicleData)):
                    vehData = vehicleData[i]
                    infData = infrastructureData[i]
                    vehBbox = self.vehicleSide.getBbox(**vehData)
                    infBbox = self.infrastructureSide.getBbox(**infData)
                    curFusedBEV = self.BEVFuser(infBbox, vehBbox, veh2inf_rotation, veh2inf_translation)
                    bboxBank.append(curFusedBEV)
                return self.PAP(bboxBank, type='bbox')
            else:
                infBbox = self.infrastructureSide.getBbox(**infData[0])
                return self.PAP(train=False, feature=infBbox, type='bbox')
        else:
            if not self.withVeh:
                return self.infrastructureSide.getBbox(train=False, **infrastructureData[0])
            vehData = vehicleData[0]
            infData = infrastructureData[0]
            vehBbox = self.vehicleSide.getBbox(train=False, **vehData)
            infBbox = self.infrastructureSide.getBbox(train=False, **infData)
            return self.bboxFuser(infBbox, vehBbox, veh2inf_rotation, veh2inf_translation)
