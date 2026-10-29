import torch
import torch.nn as nn
from mmdet3d.models import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class PAP(nn.Module):
    def __init__(self, latency=1):
        self.latency = latency
        super().__init__()
        self.BEVEncoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1))
        self.queryEncoder = nn.Linear(256, 256)
        self.bboxEncoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256))

        self.BEVFuser = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1)
        )
        self.queryFuser = nn.Linear(512, 256)
        self.bboxFuser = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10))

        self.bev_predictor = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1))
        self.query_predictor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256))
        self.bbox_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10))
        self.history = None

    def init_weights(self, pretrained=None):
        pass

    def forward(self, train=True, *args, **kwargs):
        if train:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def forward_train(self, featureBank, type=None):
        if type == 'BEV':
            B, hw, C = featureBank[0].shape
            h = w = int(hw**0.5)
            for i in range(len(featureBank)):
                feature = featureBank[i].reshape(B, h, w, C)
                featureBank[i] = feature.permute(0, 3, 1, 2)
        queueLength = len(featureBank)
        history = None
        for i in range(queueLength - 1):
            history = self.forward_hist(history, featureBank[i], type)  # -1, 0 -> 0
        # pred = self.forward_pred(history, featureBank[-1], type) # at least predict one frame -1, 0 -> 1
        # history = self.forward_hist(history, featureBank[-1], type)
        pred = featureBank[-1]
        for i in range(self.latency):
            new_pred = self.forward_pred(history, pred, type)
            history = self.forward_hist(history, pred, type)
            pred = new_pred
        if type == 'BEV':
            pred = pred.permute(0, 2, 3, 1).reshape(B, hw, C)
        return pred
    
    def forward_test(self, feature, type=None):
        if type == 'BEV':
            B, hw, C = feature.shape
            h = w = int(hw**0.5)
            feature = feature.reshape(B, h, w, C)
            feature = feature.permute(0, 3, 1, 2)
        pred = feature
        for i in range(self.latency):
            new_pred = self.forward_pred(self.history, pred, type)
            self.history = self.forward_hist(self.history, pred, type)
            pred = new_pred
        if type == 'BEV':
            pred = pred.permute(0, 2, 3, 1).reshape(B, hw, C)
        return pred
        

    def forward_hist(self, history, feature, type):
        if history is None:
            if type == 'BEV':
                return self.BEVEncoder(feature)
            elif type == 'query':
                return self.queryEncoder(feature)
            elif type == 'bbox':
                return self.bboxEncoder(feature)
            else:
                raise NotImplementedError
        if type == 'BEV':
            feature = self.BEVEncoder(feature)
            history = torch.cat([history, feature], dim=1)
            return self.BEVFuser(history)
        elif type == 'query':
            feature = self.queryEncoder(feature)
            history = torch.cat([history, feature], dim=-1)
            return self.queryFuser(history)
        elif type == 'bbox':
            feature = self.bboxEncoder(feature)
            history = torch.cat([history, feature], dim=-1)
            return self.bboxFuser(history)
        else:
            raise NotImplementedError

    def forward_pred(self, history, feature, type):
        if history is None:
            if type == 'BEV':
                return self.BEVEncoder(feature)
            elif type == 'query':
                return self.queryEncoder(feature)
            elif type == 'bbox':
                return self.bboxEncoder(feature)
            else:
                raise NotImplementedError
        if type == 'BEV':
            feature = self.BEVEncoder(feature)
            history = torch.cat([history, feature], dim=1)
            return self.bev_predictor(history)
        elif type == 'query':
            feature = self.queryEncoder(feature)
            history = torch.cat([history, feature], dim=-1)
            return self.query_predictor(history)
        elif type == 'bbox':
            feature = self.bboxEncoder(feature)
            history = torch.cat([history, feature], dim=-1)
            return self.bbox_predictor(history)
        else:
            raise NotImplementedError
