import os.path as osp
import random
import mmcv
import numpy as np
import json

random.seed(42)

max_sweeps = 5
# dairRoot = "E:\\300BUAA\\380PyCode\\data\\V2X-Seq-SPD-Example\\"
dairRoot = "/media/omnisky/C8F41D29F41D1AF01/data_wjn/V2X-Seq-SPD/"
# dairRoot = "/media/omnisky/C8F41D29F41D1AF01/data_wjn/V2X-Seq-SPD-Example/"

infoPrefix = 'BEVFormer'
CLASSES = ['Car', 'Truck', 'Van', 'Pedestrian', 'Cyclist', 'Motocyclist']

class_dict = {}


def create_all_index():
    cop_label_path = osp.join(dairRoot, 'cooperative/data_info.json')
    with open(cop_label_path, 'r') as f:
        cop_label = json.load(f)

    inf_label_path = osp.join(dairRoot, 'infrastructure-side/data_info.json')
    with open(inf_label_path, 'r') as f:
        inf_label = json.load(f)

    veh_label_path = osp.join(dairRoot, 'vehicle-side/data_info.json')
    with open(veh_label_path, 'r') as f:
        veh_label = json.load(f)

    print(len(cop_label), len(inf_label), len(veh_label))

    all_index = {}
    veh_index = {}
    inf_index = {}
    cop_infos = []

    for veh in veh_label:
        veh_fm = veh['frame_id']
        assert veh_fm not in veh_index.keys()
        veh_index[veh_fm] = veh

    for inf in inf_label:
        inf_fm = inf['frame_id']
        assert inf_fm not in inf_index.keys()
        inf_index[inf_fm] = inf

    print('Building the global temporal data index...')
    for cop in mmcv.track_iter_progress(cop_label):
        veh_sq = cop['vehicle_sequence']
        veh_fm = cop['vehicle_frame']
        inf_sq = cop['infrastructure_sequence']
        inf_fm = cop['infrastructure_frame']
        assert inf_sq == veh_sq
        if veh_sq not in all_index.keys():
            all_index[veh_sq] = {}
        # assert veh_fm not in all_index[veh_sq].keys()
        all_index[veh_sq][veh_fm] = {}

        veh_info = {}

        veh_info['frame_id'] = veh_fm
        veh_info['timestamp'] = veh_index[veh_fm]['image_timestamp']
        image_info = {}
        image_info['path'] = 'vehicle-side/' + veh_index[veh_fm]['image_path']
        assert osp.exists(osp.join(dairRoot, image_info['path']))

        calib = {}

        cam_intrin_path = osp.join(dairRoot, 'vehicle-side', veh_index[veh_fm]['calib_camera_intrinsic_path'])
        with open(cam_intrin_path, 'r') as f:
            cam_intrin = json.load(f)
        calib['cam_intrin'] = cam_intrin

        l2c_path = osp.join(dairRoot, 'vehicle-side', veh_index[veh_fm]['calib_lidar_to_camera_path'])
        with open(l2c_path, 'r') as f:
            l2c = json.load(f)
        calib['l2c'] = l2c

        e2g_path = osp.join(dairRoot, 'vehicle-side', veh_index[veh_fm]['calib_novatel_to_world_path'])
        with open(e2g_path, 'r') as f:
            e2g = json.load(f)
        calib['e2g'] = e2g

        image_info['calib'] = calib

        anno_path = osp.join(dairRoot, 'vehicle-side', veh_index[veh_fm]['label_camera_std_path'])
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        image_info['anno'] = anno

        lidar_info = {}

        lidar_info['path'] = 'vehicle-side/' + veh_index[veh_fm]['pointcloud_path']

        calib = {}

        l2e_path = osp.join(dairRoot, 'vehicle-side', veh_index[veh_fm]['calib_lidar_to_novatel_path'])
        with open(l2e_path, 'r') as f:
            l2e = json.load(f)
        calib['l2e'] = l2e['transform']
        calib['e2g'] = e2g

        lidar_info['calib'] = calib

        anno_path = osp.join(dairRoot, 'vehicle-side', veh_index[veh_fm]['label_lidar_std_path'])
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        lidar_info['anno'] = anno

        veh_info['image'] = image_info
        veh_info['lidar'] = lidar_info

        all_index[veh_sq][veh_fm]['veh_info'] = veh_info

        inf_info = {}
        inf_info['frame_id'] = inf_fm
        inf_info['timestamp'] = inf_index[inf_fm]['image_timestamp']
        image_info = {}
        image_info['path'] = 'infrastructure-side/' + inf_index[inf_fm]['image_path']
        assert osp.exists(osp.join(dairRoot, image_info['path']))
        calib = {}
        cam_intrin_path = osp.join(dairRoot, 'infrastructure-side', inf_index[inf_fm]['calib_camera_intrinsic_path'])
        with open(cam_intrin_path, 'r') as f:
            cam_intrin = json.load(f)
        calib['cam_intrin'] = cam_intrin
        l2c_path = osp.join(dairRoot, 'infrastructure-side', inf_index[inf_fm]['calib_virtuallidar_to_camera_path'])
        with open(l2c_path, 'r') as f:
            l2c = json.load(f)
        calib['l2c'] = l2c
        e2g_path = osp.join(dairRoot, 'infrastructure-side', inf_index[inf_fm]['calib_virtuallidar_to_world_path'])
        with open(e2g_path, 'r') as f:
            e2g = json.load(f)
        calib['e2g'] = e2g
        image_info['calib'] = calib
        anno_path = osp.join(dairRoot, 'infrastructure-side', inf_index[inf_fm]['label_camera_std_path'])
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        image_info['anno'] = anno
        lidar_info = {}
        lidar_info['path'] = 'infrastructure-side/' + inf_index[inf_fm]['pointcloud_path']

        calib = {}
        calib['l2e'] = {
            'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'translation': [[0], [0], [0]]
        }
        calib['e2g'] = e2g
        lidar_info['calib'] = calib
        anno_path = osp.join(dairRoot, 'infrastructure-side', inf_index[inf_fm]['label_lidar_std_path'])
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        lidar_info['anno'] = anno
        inf_info['image'] = image_info
        inf_info['lidar'] = lidar_info
        all_index[veh_sq][veh_fm]['inf_info'] = inf_info

        cop_info_path = osp.join(dairRoot, 'cooperative', 'label', veh_fm + ".json")
        with open(cop_info_path, 'r') as f:
            cop_info = json.load(f)
        gt_boxes_lidar = np.array([
            [
                ann['3d_location']['x'],
                ann['3d_location']['y'],
                ann['3d_location']['z'],
                ann['3d_dimensions']['l'],
                ann['3d_dimensions']['w'],
                ann['3d_dimensions']['h'],
                ann['rotation']
            ] for ann in cop_info
        ])
        gt_labels_lidar = np.array([ann['type'] for ann in cop_info])

        gt_labels_wash = []
        gt_boxes_wash = []
        for i in range(len(gt_labels_lidar)):
            if gt_labels_lidar[i] in CLASSES:
                gt_labels_wash.append(gt_labels_lidar[i])
                gt_boxes_wash.append(gt_boxes_lidar[i])
            if gt_labels_lidar[i] not in class_dict:
                class_dict[gt_labels_lidar[i]] = 0
            class_dict[gt_labels_lidar[i]] += 1

        if gt_labels_wash == []:
            gt_labels_wash = np.zeros((0))
            gt_boxes_wash = np.zeros((0, 7))
        else:
            gt_labels_wash = np.array(gt_labels_wash)
            gt_boxes_wash = np.array(gt_boxes_wash)

        gt_labels_no = []
        for cat in gt_labels_wash:
            if cat in CLASSES:
                gt_labels_no.append(CLASSES.index(cat))
        gt_labels_no = np.array(gt_labels_no, dtype=np.int64)

        all_index[veh_sq][veh_fm]['cop_info'] = {
            'gt_boxes': gt_boxes_wash,
            'gt_names': gt_labels_wash,
            'gt_num': len(gt_labels_wash),
            'class': gt_labels_no,
            'gt_velocity': np.zeros((gt_labels_wash.shape[0], 2))  ##
        }
    return all_index


def split_trainval(all_data, is_shuffle=False):
    all_keys = list(all_data.keys())
    if is_shuffle:
        random.shuffle(all_keys)
        split_index = int(len(all_keys) * 0.8)
        train_keys = all_keys[:split_index]
        val_keys = all_keys[split_index:]
    else:
        # val_keys = ['0000', '0080', '0023', '0087',
        #             '0054', '0034', '0088', '0049',
        #             '0063', '0025']
        val_keys = ['0035']
        train_keys = [item for item in all_keys if item not in val_keys]
    return train_keys, val_keys


def obtain_sensor2top(now_info, his_info):
    sweep = {}
    sweep['token'] = his_info['frame_id']
    l2e_n = np.eye(4)
    l2e_n[:3, :3] = np.array(now_info['lidar']['calib']['l2e']['rotation'])
    l2e_n[:3, -1] = np.array(now_info['lidar']['calib']['l2e']['translation']).reshape([-1])

    e2g_n = np.eye(4)
    e2g_n[:3, :3] = np.array(now_info['lidar']['calib']['e2g']['rotation'])
    e2g_n[:3, -1] = np.array(now_info['lidar']['calib']['e2g']['translation']).reshape([-1])

    l2e_o = np.eye(4)
    l2e_o[:3, :3] = np.array(his_info['lidar']['calib']['l2e']['rotation'])
    l2e_o[:3, -1] = np.array(his_info['lidar']['calib']['l2e']['translation']).reshape([-1])

    e2g_o = np.eye(4)
    e2g_o[:3, :3] = np.array(his_info['lidar']['calib']['e2g']['rotation'])
    e2g_o[:3, -1] = np.array(his_info['lidar']['calib']['e2g']['translation']).reshape([-1])

    lo2ln = np.linalg.inv(l2e_n) @ np.linalg.inv(e2g_n) @ e2g_o @ l2e_o
    sweep['sensor2lidar'] = lo2ln

    return sweep


def _gen_sweeps(single_info, seq_info, key_info, device):
    keys_in_seq = sorted(list(seq_info.keys()))
    cur_id = keys_in_seq.index(key_info)

    if cur_id == 0:
        prev_id = ''
        next_id = device[:4] + seq_info[keys_in_seq[cur_id + 1]][device]['frame_id']
    elif cur_id == len(keys_in_seq) - 1:
        prev_id = device[:4] + seq_info[keys_in_seq[cur_id - 1]][device]['frame_id']
        next_id = ''
    else:
        prev_id = device[:4] + seq_info[keys_in_seq[cur_id - 1]][device]['frame_id']
        next_id = device[:4] + seq_info[keys_in_seq[cur_id + 1]][device]['frame_id']

    sweeps = []
    his_id = cur_id - 1
    while len(sweeps) <= max_sweeps:
        if (not prev_id == '') and (his_id >= 0):
            sweep = obtain_sensor2top(single_info, seq_info[keys_in_seq[his_id]][device])
            sweeps.append(sweep)
            his_id -= 1
        else:
            break

    return prev_id, next_id, sweeps


def rotation_matrix_to_quaternion(matrix):
    matrix = np.array(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]

    trace = np.trace(matrix)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (matrix[2, 1] - matrix[1, 2]) / S
        qy = (matrix[0, 2] - matrix[2, 0]) / S
        qz = (matrix[1, 0] - matrix[0, 1]) / S
    elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
        S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        qw = (matrix[2, 1] - matrix[1, 2]) / S
        qx = 0.25 * S
        qy = (matrix[0, 1] + matrix[1, 0]) / S
        qz = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        qw = (matrix[0, 2] - matrix[2, 0]) / S
        qx = (matrix[0, 1] + matrix[1, 0]) / S
        qy = 0.25 * S
        qz = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        qw = (matrix[1, 0] - matrix[0, 1]) / S
        qx = (matrix[0, 2] + matrix[2, 0]) / S
        qy = (matrix[1, 2] + matrix[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def _gen_single_info(single_info, seq_info, key_info, device):
    ori_frame_id = single_info['frame_id']
    prev_frame, next_frame, sweeps = _gen_sweeps(single_info, seq_info, key_info, device)

    lidar_path = single_info['lidar']['path']
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = np.array(single_info['lidar']['calib']['l2e']['rotation'])
    lidar2ego[:3, -1] = np.array(single_info['lidar']['calib']['l2e']['translation']).reshape([-1])

    ego2global = np.eye(4)
    ego2global[:3, :3] = np.array(single_info['lidar']['calib']['e2g']['rotation'])
    ego2global[:3, -1] = np.array(single_info['lidar']['calib']['e2g']['translation']).reshape([-1])

    lidar2global = ego2global @ lidar2ego

    cam = {}
    cam['data_path'] = dairRoot + single_info['image']['path']
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = np.array(single_info['image']['calib']['l2c']['rotation'])
    lidar2cam[:3, -1] = np.array(single_info['image']['calib']['l2c']['translation']).reshape([-1])
    # cam2ego = lidar2ego @ np.linalg.inv(lidar2cam)
    # cam['cam2ego'] = cam2ego
    cam['cam_intrinsic'] = np.array(single_info['image']['calib']['cam_intrin']['cam_K']).reshape((3, 3))
    cam['lidar2cam_r'] = lidar2cam[:3, :3]
    cam['lidar2cam_t'] = lidar2cam[:3, -1]
    sensor2lidar = np.linalg.inv(lidar2cam)
    cam['sensor2lidar_rotation'] = sensor2lidar[:3, :3]
    cam['sensor2lidar_translation'] = sensor2lidar[:3, 3]

    ann_info_cam = single_info['image']['anno']
    gt_boxes_cam = np.array([
        [
            ann['3d_location']['x'],
            ann['3d_location']['y'],
            ann['3d_location']['z'],
            ann['3d_dimensions']['l'],
            ann['3d_dimensions']['w'],
            ann['3d_dimensions']['h'],
            ann['rotation']
        ]
        for ann in ann_info_cam
    ])
    gt_labels_cam = np.array([ann['type'] for ann in ann_info_cam])
    gt_inds_cam = np.array([ann['track_id'] for ann in ann_info_cam])

    ann_info_lidar = single_info['lidar']['anno']
    gt_boxes_lidar = np.array([
        [
            ann['3d_location']['x'],
            ann['3d_location']['y'],
            ann['3d_location']['z'],
            ann['3d_dimensions']['l'],
            ann['3d_dimensions']['w'],
            ann['3d_dimensions']['h'],
            ann['rotation']
        ]
        for ann in ann_info_lidar
    ])
    gt_labels_lidar = np.array([ann['type'] for ann in ann_info_lidar])

    gt_labels_wash = []
    gt_boxes_wash = []
    for i in range(len(gt_labels_cam)):
        if gt_labels_cam[i] in CLASSES:
            gt_labels_wash.append(gt_labels_cam[i])
            gt_boxes_wash.append(gt_boxes_cam[i])
    if gt_labels_wash == []:
        gt_labels_wash = np.zeros((0))
        gt_boxes_wash = np.zeros((0, 7))
    else:
        gt_labels_wash = np.array(gt_labels_wash)
        gt_boxes_wash = np.array(gt_boxes_wash)

    gt_labels_no = []
    for cat in gt_labels_wash:
        if cat in CLASSES:
            gt_labels_no.append(CLASSES.index(cat))
    gt_labels_no = np.array(gt_labels_no, dtype=np.int64)

    info = {
        'type': device,
        'lidar_path': dairRoot + lidar_path,
        'token': device[:4] + ori_frame_id,
        'prev': prev_frame,
        'next': next_frame,
        'sweeps': sweeps,
        'ego2global_rotation': rotation_matrix_to_quaternion(ego2global[:3, :3]),
        'ego2global_translation': ego2global[:3, -1],
        'lidar2ego_rotation': rotation_matrix_to_quaternion(lidar2ego[:3, :3]),
        'lidar2ego_translation': lidar2ego[:3, -1],
        'lidar2global': lidar2global,
        'cams': {'CAM_FRONT': cam},
        'can_bus': np.zeros(18),
        
        'valid_flag': np.array(range(gt_labels_wash.shape[0]), dtype=np.int32),
        'gt_boxes': gt_boxes_wash,
        'gt_names': gt_labels_wash,
        'gt_num': len(gt_labels_wash),
        'class': gt_labels_no,
        'gt_velocity': np.zeros((gt_labels_wash.shape[0], 2))
    }

    return info


def _fill_trainval_infos(all_data, train_index):
    veh_train_infos = []
    veh_val_infos = []
    inf_train_infos = []
    inf_val_infos = []
    cop_train_annos = []
    cop_val_annos = []

    timeStamp = 0.0
    for seq in mmcv.track_iter_progress(all_data.keys()):
        sequence_data = all_data[seq]
        all_frame_keys = list(sequence_data.keys())
        assert all_frame_keys == sorted(all_frame_keys)

        for frame_id, frame_key in enumerate(all_frame_keys):
            sample = sequence_data[frame_key]

            veh_info = _gen_single_info(sample['veh_info'], sequence_data, frame_key, 'veh_info')
            inf_info = _gen_single_info(sample['inf_info'], sequence_data, frame_key, 'inf_info')
            
            sample['cop_info']['veh2inf'] = np.linalg.inv(inf_info['lidar2global']) @ veh_info['lidar2global']
            # print(sample['cop_info']['veh2inf'])

            veh_info['frame_idx'] = frame_id  #
            veh_info['scene_token'] = seq  #
            veh_info['timestamp'] = timeStamp  #
            inf_info['frame_idx'] = frame_id  #
            inf_info['scene_token'] = seq  #
            inf_info['timestamp'] = timeStamp  #

            sample['cop_info']['timestamp'] = timeStamp
            timeStamp += 1

            if seq in train_index:
                veh_train_infos.append(veh_info)
                inf_train_infos.append(inf_info)
                cop_train_annos.append(sample['cop_info'])
            else:
                veh_val_infos.append(veh_info)
                inf_val_infos.append(inf_info)
                cop_val_annos.append(sample['cop_info'])

    return veh_train_infos, veh_val_infos, inf_train_infos, inf_val_infos, cop_train_annos, cop_val_annos


def shuffle(all_data, train_index, val_index):
    train = []
    val = []
    for i in train_index:
        train.append(all_data[i])
    for i in val_index:
        val.append(all_data[i])
    return train, val


if __name__ == '__main__':
    all_index = create_all_index()
    print(class_dict)
    train_index, val_index = split_trainval(all_index)

    print('Creating training and validation splits...')
    veh_train_infos, veh_val_infos, inf_train_infos, inf_val_infos, cop_train_annos, cop_val_annos = _fill_trainval_infos(
        all_index, train_index)
    
    # veh_val_infos = veh_train_infos
    # inf_val_infos = inf_train_infos
    # cop_val_annos = cop_train_annos


    # index = [i for i in range(len(veh_train_infos))]
    # random.shuffle(index)
    # split_index = int(len(index) * 0.804)
    # train_keys = index[:split_index]
    # val_keys = index[split_index:]

    # veh_train_infos, veh_val_infos = shuffle(veh_train_infos, train_keys, val_keys)
    # inf_train_infos, inf_val_infos = shuffle(inf_train_infos, train_keys, val_keys)
    # cop_train_annos, cop_val_annos = shuffle(cop_train_annos, train_keys, val_keys)

    metadata = dict(version='DAIR-V2X')
    print('Training samples: {}, validation samples: {}'.format(len(veh_train_infos), len(veh_val_infos)))
    veh_data = dict(infos=veh_train_infos, metadata=metadata)
    info_path = osp.join(dairRoot, '{}/veh_temporal_infos_train.pkl'.format(infoPrefix))
    mmcv.dump(veh_data, info_path)
    veh_data['infos'] = veh_val_infos
    info_val_path = osp.join(dairRoot, '{}/veh_temporal_infos_val.pkl'.format(infoPrefix))
    mmcv.dump(veh_data, info_val_path)

    inf_data = dict(infos=inf_train_infos, metadata=metadata)
    info_path = osp.join(dairRoot, '{}/inf_temporal_infos_train.pkl'.format(infoPrefix))
    mmcv.dump(inf_data, info_path)
    inf_data['infos'] = inf_val_infos
    info_val_path = osp.join(dairRoot, '{}/inf_temporal_infos_val.pkl'.format(infoPrefix))
    mmcv.dump(inf_data, info_val_path)

    info_path = osp.join(dairRoot, 'COOPERATION/cop_train_annos.pkl'.format(infoPrefix))
    mmcv.dump(cop_train_annos, info_path)
    info_val_path = osp.join(dairRoot, 'COOPERATION/cop_val_annos.pkl'.format(infoPrefix))
    mmcv.dump(cop_val_annos, info_val_path)
