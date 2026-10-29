import mmcv
from torch.utils.data import Dataset
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets import DATASETS
import numpy as np
from mmdet3d.core.bbox.structures.utils import get_box_type
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
import torch


@DATASETS.register_module()
class AiODataset(Dataset):
    def __init__(self,
                 cloudDataset,
                 vehicleDataset,
                 anno_file,
                 point_cloud_range,
                 model_delay=0,
                 test_mode=False):
        super().__init__()
        self.cloudDataset = build_dataset(cloudDataset, dict(test_mode=test_mode))
        self.vehicleDataset = build_dataset(vehicleDataset, dict(test_mode=test_mode))
        self.cop_infos_file = anno_file
        self.model_delay = model_delay
        self.point_cloud_range = np.array(point_cloud_range)
        self.point_cloud_range[0] = 0
        self.test_mode = test_mode
        assert len(self.cloudDataset) == len(self.vehicleDataset)
        self.CLASSES = self.vehicleDataset.CLASSES
        if not self.test_mode:
            self._set_group_flag()
        self.box_type_3d, self.box_mode_3d = get_box_type('lidar')
        data = mmcv.load(self.cop_infos_file, file_format='pkl')
        data_infos = list(sorted(data, key=lambda x: x['timestamp']))
        self.cop_infos = data_infos


    def get_cop_annos(self, idx):
        info = self.cop_infos[idx]
        gt_bboxes_3d = info['gt_boxes']

        gt_velocity = info['gt_velocity']
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_names_3d = info['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.vehicleDataset.CLASSES:
                gt_labels_3d.append(self.vehicleDataset.CLASSES.index(cat))
        gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        mask = gt_bboxes_3d.in_range_bev(self.point_cloud_range[[0, 1, 3, 4]])
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)

        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)
        return gt_bboxes_3d, gt_labels_3d, DC(torch.tensor(info['veh2inf']), cpu_only=True)

    def get_cop_veh2inf(self, idx):
        info = self.cop_infos[idx]
        return DC(torch.tensor(info['veh2inf']), cpu_only=True)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            # print(idx)
            data = self.prepare_train_data(idx)
            if data is None:
                print("index error", idx)
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_data(self, idx):
        # idx = 3331
        # print(idx)
        cloudData = self.cloudDataset[idx]  # dict + list
        if cloudData is None:
            return None
        vehicleData = self.vehicleDataset[idx]
        if vehicleData is None:
            return None
        img_metas = dict(img=[])

        gt_bboxes_3d, gt_labels_3d, veh2inf = self.get_cop_annos(idx)
        # print(idx, gt_bboxes_3d, gt_labels_3d)
        if ~(gt_labels_3d._data != -1).any():
            return None
        return dict(
            cloudData=cloudData,
            vehicleData=vehicleData,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            veh2inf=veh2inf,
            img_metas=img_metas)

    def prepare_test_data(self, idx):
        cloudData = self.cloudDataset[idx]  # dict + list
        vehicleData = self.vehicleDataset[idx]
        img_metas = dict(img=[])
        return dict(
            cloudData=cloudData if cloudData is not None else {},
            vehicleData=vehicleData,
            veh2inf=self.get_cop_veh2inf(idx),
            img_metas=img_metas)

    def __len__(self):
        return len(self.vehicleDataset)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self, results, logger=None, **kwargs):
        # print()
        # print("evaluate inf side")
        # result0 = self.cloudDataset.infrastructureDataset.evaluate(results, logger=logger, 
        #                                                            point_cloud_range=self.point_cloud_range, **kwargs)
        print()
        # Use the following evaluator for vehicle-side detections.
        print("evaluate veh side")
        result1 = self.vehicleDataset.evaluate(results, logger=logger, **kwargs)

        print()
        # Cloud-wide evaluation uses an application-specific evaluator.
        print("evaluate cloud side")
        result2 = self.evaluate_cloud(results, logger=logger, **kwargs)

        # import pickle
        # with open('display/result_veh_only.pkl', 'wb') as f:
        #     pickle.dump(results, f)

        return result2

    def evaluate_cloud(self,
                      results,
                      metric=None,
                      iou_thr=(0.25, 0.5, 0.7),
                      logger=None,
                      point_cloud_range=None,
                      show=False,
                      out_dir=None,
                      pipeline=None,
                      **kwargs):
        from .eval import indoor_eval
        # from mmdet3d.core.evaluation.indoor_eval import indoor_eval
        assert isinstance(results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.cop_infos)
        assert isinstance(results[0], dict), f'Expect elements in results to be dict, got {type(results[0])}.'
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(
            self.cop_infos,
            results,
            iou_thr,
            label2cat,
            model_delay=self.model_delay,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            point_cloud_range=point_cloud_range)
        return ret_dict
