from torch.utils.data import Dataset
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets import DATASETS
import numpy as np


@DATASETS.register_module()
class AiOCloudDataset(Dataset):
    def __init__(self,
                 vehicleDataset,
                 infrastructureDataset,
                 latency=1,
                 withPAP=True,
                 withVeh=True,
                 queueLength=2,
                 test_mode=False):
        super().__init__()
        self.vehicleDataset = build_dataset(vehicleDataset, dict(test_mode=test_mode))
        self.infrastructureDataset = build_dataset(infrastructureDataset, dict(test_mode=test_mode))
        self.latency = latency
        self.withPAP = withPAP
        self.withVeh = withVeh
        self.test_mode = test_mode
        self.queueLength = queueLength
        assert len(self.vehicleDataset) == len(self.infrastructureDataset)
        self.CLASSES = self.vehicleDataset.CLASSES
        if not test_mode:
            self._set_group_flag()

    def __getitem__(self, idx):
        if self.withVeh:
            return self.prepare_veh_inf_data(idx)
        else:
            return self.prepare_inf_only_data(idx)

    def prepare_inf_only_data(self, idx):
        if idx - self.latency < 0:
            return None
        infrastructureData = []
        infData = self.infrastructureDataset[idx - self.latency]
        if infData is None:
            return None
        infrastructureData.insert(0, infData)
        if self.test_mode:
            return dict(vehicleData=dict(),
                        infrastructureData=infrastructureData)
        if self.withPAP:
            for i in range(1, self.queueLength):
                if idx - i - self.latency < 0:
                    return None
                infData = self.infrastructureDataset[idx - i - self.latency]
                if infData is None:
                    return None
                infrastructureData.insert(0, infData)

        return dict(vehicleData=dict(),
                    infrastructureData=infrastructureData)

    def prepare_veh_inf_data(self, idx):
        if idx - self.latency < 0:
            return None
        vehicleData = []
        infrastructureData = []
        vehData = self.vehicleDataset[idx - self.latency]
        infData = self.infrastructureDataset[idx - self.latency]
        if vehData is None or infData is None:
            return None

        vehicleData.insert(0, vehData)
        infrastructureData.insert(0, infData)
        if self.withPAP:
            for i in range(1, self.queueLength):
                if idx - i - self.latency < 0:
                    break
                vehData = self.vehicleDataset[idx - i - self.latency]
                infData = self.infrastructureDataset[idx - i - self.latency]
                if vehData is None or infData is None:
                    break
                vehicleData.insert(0, vehData)
                infrastructureData.insert(0, infData)
        return dict(vehicleData=vehicleData,
                    infrastructureData=infrastructureData)

    def __len__(self):
        return len(self.vehicleDataset)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self, results, logger, **kwargs):
        return None
