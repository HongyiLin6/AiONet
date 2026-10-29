# data_root = r'E:\300BUAA\380PyCode\data\V2X-Seq-SPD-Example\\'
# data_root = "/media/omnisky/C8F41D29F41D1AF01/data_wjn/V2X-Seq-SPD-Example/"
data_root = "/media/omnisky/C8F41D29F41D1AF01/data_wjn/V2X-Seq-SPD/"

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = ['Car', 'Truck', 'Van', 'Bus', 'trailer', 'Unknown_unmovable',
               'Motorcyclist', 'Cyclist', 'Pedestrian', 'TrafficCone']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 150
bev_w_ = 150
queue_length = 3

num_query = 900