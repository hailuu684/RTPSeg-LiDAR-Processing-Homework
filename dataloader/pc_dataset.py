import os
import yaml
# import cv2
import numpy as np
#import open3d as o3d
from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils import data
import copy
REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))




@register_dataset
class PIR(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        #self.learning_map_2 = semkittiyaml['learning_map_2']
        self.imageset = imageset

        if imageset == 'train':

            #a = semkittiyaml['split']['train'][0][0]
            #b = semkittiyaml['split']['train'][0][1]
            #split = [x for x in range(a, b+1)]
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            #a = semkittiyaml['split']['valid'][0][0]
            #b = semkittiyaml['split']['valid'][0][1]
            #split = [x for x in range(a, b+1)]
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'lidar']), num_vote)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)



    @staticmethod
    def imu_rectify_points(points, imu_dict):
        points_copy = copy.deepcopy(points)
        R = imu_dict['R']
        R_LI = imu_dict['R_LI']
        R_out_inv = imu_dict['R_out_inv']
        R_end_inv = imu_dict['R_end_inv']
        T_LI = imu_dict['T_LI']
        T_out = imu_dict['T_out']
        T_end = imu_dict['T_end']

        for i in range(len(points)):
            point = points[i]
            points_copy[i] = R_LI.T.dot(R_out_inv.dot(R_end_inv.dot(R_LI.dot((R.T.dot(point.T)).T) + T_LI) + T_end - T_out) - T_LI)
        return points_copy

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out





    def __getitem__(self, index):

        raw_data = np.load(self.im_idx[index])  

        origin_len = raw_data.shape[0]
        points = raw_data[:, :3]




        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)

        else:
            annotated_data = np.load(self.im_idx[index].replace('lidar', 'labels')[:-3] + 'npz')['arr_0']#.reshape((-1, 1))
            annotated_data = annotated_data[:,1]
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)


        others_mask = annotated_data!=0
        annotated_data = annotated_data[others_mask] - 1
        points = points[others_mask]

        image_file = self.im_idx[index].replace('lidar', 'camera').replace('.npy', '.png')
        image = Image.open(image_file)
        infra_file = self.im_idx[index].replace('lidar', 'infra').replace('.npy', '.png')
        infra = Image.open(infra_file)

        img_imu_path = self.im_idx[index].replace('lidar', 'imu_pos_rgb').replace('.npy', '.pkl')
        infra_imu_path = self.im_idx[index].replace('lidar', 'imu_pos_infra').replace('.npy', '.pkl')
        import pickle
        with open(img_imu_path, "rb") as f:
            img_imu_dict = pickle.load(f)
        with open(infra_imu_path, "rb") as f:
            infra_imu_dict = pickle.load(f)




        img_imu_points = self.imu_rectify_points(points, img_imu_dict)
        infra_imu_points = self.imu_rectify_points(points, infra_imu_dict)


        data_dict = {}
        data_dict['xyz'] = points
        data_dict['img_imu_points'] = img_imu_points
        data_dict['infra_imu_points'] = infra_imu_points

        data_dict['labels'] = annotated_data.astype(np.uint8)
        #data_dict['instance_label'] = instance_label
        #data_dict['signal'] = raw_data[:, 3:4]
        data_dict['signal'] = raw_data[others_mask, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['infra'] = infra
        #data_dict['proj_matrix'] = proj_matrix
        data_dict['imageset'] = self.imageset

        return data_dict, self.im_idx[index]



def _blur3x3(x):
    """3x3 box blur, equivalent to cv2.blur(x, (3,3)) (BORDER_REFLECT_101)."""
    pad = [(1, 1), (1, 1)] + [(0, 0)] * (x.ndim - 2)
    p = np.pad(x, pad, mode='symmetric')          # symmetric == BORDER_REFLECT_101
    out = np.zeros_like(x, dtype=np.float32)
    for dv in range(3):
        for du in range(3):
            out += p[dv:dv + x.shape[0], du:du + x.shape[1]]
    return out * (1.0 / 9.0)


def render_virtual_view(xyz, rgb, cam_height, fx, W, H, splat_size, max_splat_radius, depth_tol, hole_fill_iters):
    """
    Pinhole camera at (0, 0, cam_height) looking along +x of the local frame (x forward, y left, z up).
    Colored points are splatted into a z-buffer (footprint shrinks with depth), small holes are filled.
    :return: image (H, W, 3) uint8, img_rc (N, 2) float32 (row, col) per point, visible (N,) bool
             visible = point is in the image AND is the front-most surface at its pixel
    """
    N = len(xyz)
    X, Y, Z = -xyz[:, 1], cam_height - xyz[:, 2], xyz[:, 0]
    img_rc = np.zeros((N, 2), dtype=np.float32)
    front = Z > 0.3
    img_rc[front, 0] = fx * Y[front] / Z[front] + H / 2
    img_rc[front, 1] = fx * X[front] / Z[front] + W / 2
    inside = front & (img_rc[:, 0] >= 0) & (img_rc[:, 0] < H) & (img_rc[:, 1] >= 0) & (img_rc[:, 1] < W)
    idx = np.nonzero(inside)[0]
    if len(idx) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8), img_rc, np.zeros(N, dtype=bool)
    vi, ui, z = img_rc[idx, 0].astype(np.int64), img_rc[idx, 1].astype(np.int64), Z[idx]

    # splat candidates: every point covers a (2r+1)^2 window, r ~ projected footprint
    rad = np.clip(np.round(fx * splat_size / z / 2), 0, max_splat_radius).astype(np.int64)
    pix, dep, src = [], [], []
    for r in np.unique(rad):
        sel = np.nonzero(rad == r)[0]
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                vv, uu = vi[sel] + dv, ui[sel] + du
                ok = (vv >= 0) & (vv < H) & (uu >= 0) & (uu < W)
                pix.append(vv[ok] * W + uu[ok])
                dep.append(z[sel][ok])
                src.append(sel[ok])
    pix, dep, src = np.concatenate(pix), np.concatenate(dep), np.concatenate(src)

    # z-buffer: sort by (pixel, depth in mm), keep the nearest candidate per pixel
    key = pix * (1 << 24) + np.minimum(dep * 1000, (1 << 24) - 1).astype(np.int64)
    order = np.argsort(key)
    first = np.r_[True, pix[order][1:] != pix[order][:-1]]
    win = order[first]
    depth = np.full(H * W, np.inf, dtype=np.float32)
    depth[pix[win]] = dep[win]
    image = np.zeros((H * W, 3), dtype=np.float32)
    image[pix[win]] = rgb[idx[src[win]]]

    # fill pinholes between splats (sky / empty space far away stays black)
    image, valid = image.reshape(H, W, 3), np.isfinite(depth).reshape(H, W).astype(np.float32)
    for _ in range(hole_fill_iters):
        s, n = _blur3x3(image * valid[..., None], (3, 3)), _blur3x3(valid, (3, 3))
        new = (valid == 0) & (n > 0)
        image[new] = s[new] / n[new][:, None]
        valid[new] = 1

    tol_abs, tol_rel = depth_tol
    visible = np.zeros(N, dtype=bool)
    visible[idx] = z <= depth[vi * W + ui] * (1 + tol_rel) + tol_abs
    return image.astype(np.uint8), img_rc, visible


@register_dataset
class Toronto3D(data.Dataset):
    """
    Toronto-3D tiles preprocessed by tools/prepare_toronto3d.py. One sample = one LiDAR-like "frame":
    all points within frame_radius of a vehicle pose on the estimated scan trajectory, expressed in a local
    frame (x = heading, z = up from ground), plus a virtual RGB image rendered from the colored points.
    """
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            toronto_yaml = yaml.safe_load(stream)
        learning_map = toronto_yaml['learning_map']
        self.label_lut = np.zeros(max(learning_map) + 1, dtype=np.uint8)
        for k, v in learning_map.items():
            self.label_lut[k] = v

        p = config['dataset_params']['toronto3d']
        self.imageset = imageset
        self.radius = p['frame_radius']
        self.cam_height = p['cam_height']
        self.W, self.H = p['img_size']
        self.fx = self.W / 2 / np.tan(np.deg2rad(p['hfov']) / 2)
        self.splat_size = p['splat_size']
        self.max_splat_radius = p['max_splat_radius']
        self.depth_tol = p['depth_tol']
        self.hole_fill_iters = p['hole_fill_iters']
        self.pose_jitter = p['pose_jitter'] if imageset == 'train' else None

        if imageset == 'train':
            split = list(toronto_yaml['split']['train'])
            if config['train_params'].get('trainval', False):
                split += toronto_yaml['split']['valid']
            pose_stride = p['train_pose_stride']
        elif imageset == 'val':
            split = toronto_yaml['split']['valid']
            pose_stride = p['val_pose_stride']
        elif imageset == 'test':
            split = toronto_yaml['split']['test']
            pose_stride = p['val_pose_stride']
        else:
            raise Exception('Split must be train/val/test')

        self.tiles = {}
        self.frames = []
        for name in split:
            d = np.load(Path(data_path) / f'{name}.npz')
            order = np.argsort(d['xyz'][:, 0], kind='stable')  # x-sorted -> frame crop is a searchsorted slice
            tile = {k: d[k][order] for k in ('xyz', 'rgb', 'intensity', 'label')}
            tile['poses'] = d['poses'][::pose_stride]
            if imageset != 'train':
                # every point is scored in exactly one frame: the one with the nearest pose
                tile['owner'] = cKDTree(tile['poses'][:, :2]).query(tile['xyz'][:, :2])[1].astype(np.int32)
            self.tiles[name] = tile
            self.frames += [(name, i) for i in range(len(tile['poses'])) for _ in range(num_vote)]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.frames)

    def __getitem__(self, index):
        name, i = self.frames[index]
        tile = self.tiles[name]
        x0, y0, z0, yaw = tile['poses'][i].astype(np.float64)
        if self.pose_jitter is not None:
            d_along, d_lat, d_yaw = self.pose_jitter
            a, l = np.random.uniform(-d_along, d_along), np.random.uniform(-d_lat, d_lat)
            x0 += a * np.cos(yaw) - l * np.sin(yaw)
            y0 += a * np.sin(yaw) + l * np.cos(yaw)
            yaw += np.deg2rad(np.random.uniform(-d_yaw, d_yaw)) + np.pi * (np.random.rand() < 0.5)

        lo, hi = np.searchsorted(tile['xyz'][:, 0], [x0 - self.radius, x0 + self.radius])
        d2 = (tile['xyz'][lo:hi, 0] - x0) ** 2 + (tile['xyz'][lo:hi, 1] - y0) ** 2
        idx = lo + np.nonzero(d2 < self.radius ** 2)[0]
        labels = self.label_lut[tile['label'][idx]]
        others_mask = labels != 0  # drop unclassified points, as in the RTPSeg loader
        idx, labels = idx[others_mask], labels[others_mask] - 1
        if self.imageset != 'train':
            labels[tile['owner'][idx] != i] = 255

        c, s = np.cos(yaw), np.sin(yaw)
        p = tile['xyz'][idx].astype(np.float64) - [x0, y0, z0]
        xyz = np.stack([c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1], p[:, 2]], 1).astype(np.float32)

        image, img_rc, img_visible = render_virtual_view(
            xyz, tile['rgb'][idx], self.cam_height, self.fx, self.W, self.H,
            self.splat_size, self.max_splat_radius, self.depth_tol, self.hole_fill_iters)

        data_dict = {}
        data_dict['xyz'] = xyz
        data_dict['labels'] = labels.astype(np.uint8)
        data_dict['signal'] = tile['intensity'][idx].reshape(-1, 1).astype(np.float32)
        data_dict['rgb'] = tile['rgb'][idx]
        data_dict['img'] = Image.fromarray(image)
        data_dict['img_rc'] = img_rc
        data_dict['img_visible'] = img_visible
        data_dict['origin_len'] = len(xyz)
        data_dict['pose'] = np.array([x0, y0, z0, yaw])
        data_dict['imageset'] = self.imageset

        return data_dict, f'{name}/{i:05d}'


@register_dataset
class RTPSegRGB(data.Dataset):
    """
    RTPSeg LiDAR + RGB camera, no TIR. Same output as Toronto3D, so both run through RGBPC_dataset.
    Point -> pixel projections (IMU-rectified, camera calibration) are precomputed by tools/prepare_rtpseg.py.
    """
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            rtpseg_yaml = yaml.safe_load(stream)
        learning_map = rtpseg_yaml['learning_map']
        self.label_lut = np.zeros(max(learning_map) + 1, dtype=np.uint8)
        for k, v in learning_map.items():
            self.label_lut[k] = v
        self.imageset = imageset
        self.proj_path = Path(config['dataset_params']['rtpseg']['proj_path'])

        if imageset == 'train':
            split = list(rtpseg_yaml['split']['train'])
            if config['train_params'].get('trainval', False):
                split += rtpseg_yaml['split']['valid']
        elif imageset in ('val', 'test'):
            split = rtpseg_yaml['split']['valid']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for seq in split:
            for f in sorted((Path(data_path) / str(seq).zfill(2) / 'lidar').glob('*.npy')):
                self.im_idx += [f] * num_vote

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        lidar_file = self.im_idx[index]
        seq_dir, seq = lidar_file.parent.parent, lidar_file.parent.parent.name
        raw_data = np.load(lidar_file)
        labels = np.load(seq_dir / 'labels' / f'{lidar_file.stem}.npz')['arr_0'][:, 1]
        labels = self.label_lut[labels.astype(np.int64)]

        proj = np.load(self.proj_path / seq / f'{lidar_file.stem}.npz')
        img_rc = np.zeros((len(raw_data), 2), dtype=np.float32)
        img_rc[proj['idx']] = proj['rc']
        img_visible = np.zeros(len(raw_data), dtype=bool)
        img_visible[proj['idx']] = True

        others_mask = labels != 0  # drop "Other", as in the upstream loader
        data_dict = {}
        data_dict['xyz'] = raw_data[others_mask, :3].astype(np.float32)
        data_dict['labels'] = (labels[others_mask] - 1).astype(np.uint8)
        data_dict['signal'] = raw_data[others_mask, 3:4].astype(np.float32)
        data_dict['img'] = Image.open(seq_dir / 'camera' / f'{lidar_file.stem}.png').convert('RGB')
        data_dict['img_rc'] = img_rc[others_mask]
        data_dict['img_visible'] = img_visible[others_mask]
        data_dict['origin_len'] = len(raw_data)
        data_dict['imageset'] = self.imageset

        return data_dict, f'{seq}/{lidar_file.stem}'


def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name,semkittiyaml['learning_map']


get_label_name = get_SemKITTI_label_name  # utils/metric_util.py imports this name



#from data_util import sa_create
#from data_util import data_prepare
#import SharedArray as SA

