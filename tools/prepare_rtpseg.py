"""
RTPSeg -> RTPSegNet input (point cloud + RGB, no TIR).
1) find the sequence folders (<seq>/lidar, labels, camera, imu_pos_rgb) under <data-root>/raw, link them as <data-root>/sequences
2) per frame: IMU-rectify the LiDAR points to the RGB exposure time and project them with the camera calibration
   (json/answer_camera.json for seq < 63, answer_camera_02.json otherwise), exactly as the upstream loader, vectorized
3) save <data-root>/processed_rgb_proj/<seq>/<frame>.npz: idx (points in front of the camera and inside the image), rc (row, col)
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

NUM_LABELS = 19


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--json-dir', type=Path, default=Path('json'))
    parser.add_argument('--min-depth', type=float, default=1.0, help='upstream drops points with camera z < 1 m')
    return parser.parse_args()


class NumpyUnpickler(pickle.Unpickler):
    """imu .pkl files hold numpy arrays only; refuse anything else (unpickling can execute code)"""
    ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
               ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
               ('numpy', 'ndarray'), ('numpy', 'dtype'), ('collections', 'OrderedDict')}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name} in imu pickle')


def load_calib(path):
    c = json.loads(path.read_text())
    return {'K': np.array(c['camera_matrix'], dtype=np.float64), 'R': np.array(c['rotation_matrix'], dtype=np.float64),
            't': np.array(c['translation_vector'], dtype=np.float64).reshape(3)}


def imu_rectify(points, d):
    """vectorized PIR.imu_rectify_points: R_LI^T (R_out_inv (R_end_inv (R_LI R^T p + T_LI) + T_end - T_out) - T_LI)"""
    R, R_LI, R_out_inv, R_end_inv = [np.asarray(d[k], dtype=np.float64) for k in ('R', 'R_LI', 'R_out_inv', 'R_end_inv')]
    T_LI, T_out, T_end = [np.asarray(d[k], dtype=np.float64).reshape(3) for k in ('T_LI', 'T_out', 'T_end')]
    w = points @ R @ R_LI.T
    z = (w + T_LI) @ R_end_inv.T
    y = (z + T_end - T_out) @ R_out_inv.T
    return (y - T_LI) @ R_LI


def project(points, calib, W, H, min_depth):
    cam = points @ calib['R'].T + calib['t']
    front = np.nonzero(cam[:, 2] >= min_depth)[0]
    uvw = cam[front] @ calib['K'].T
    uv = uvw[:, :2] / uvw[:, 2:3]
    inside = (uv[:, 0] > 0) & (uv[:, 1] > 0) & (uv[:, 0] < W) & (uv[:, 1] < H)
    return front[inside].astype(np.int32), uv[inside][:, ::-1].astype(np.float32)  # (row, col)


def main():
    args = parse_args()
    raw_dir = args.data_root / 'raw'
    lidar_dirs = sorted(p for p in raw_dir.rglob('lidar') if p.is_dir())
    assert lidar_dirs, f'no <seq>/lidar folders under {raw_dir}; run the unzip step first'
    seq_root = lidar_dirs[0].parent.parent
    assert all(p.parent.parent == seq_root for p in lidar_dirs), f'sequences in several folders: {lidar_dirs[:5]}'
    link = args.data_root / 'sequences'
    if not link.exists():
        link.symlink_to(seq_root.resolve(), target_is_directory=True)
    print(f'sequences: {seq_root} ({len(lidar_dirs)}) -> {link}')

    calibs = {False: load_calib(args.json_dir / 'answer_camera.json'), True: load_calib(args.json_dir / 'answer_camera_02.json')}
    out_root = args.data_root / 'processed_rgb_proj'
    label_hist = np.zeros(NUM_LABELS, dtype=np.int64)
    for lidar_dir in lidar_dirs:
        seq_dir = lidar_dir.parent
        calib = calibs[int(seq_dir.name) >= 63]
        (out_root / seq_dir.name).mkdir(parents=True, exist_ok=True)
        n_points, n_inside, missing = [], [], 0
        for f in sorted(lidar_dir.glob('*.npy')):
            image_file = seq_dir / 'camera' / f'{f.stem}.png'
            imu_file = seq_dir / 'imu_pos_rgb' / f'{f.stem}.pkl'
            label_file = seq_dir / 'labels' / f'{f.stem}.npz'
            if not (image_file.exists() and imu_file.exists() and label_file.exists()):
                missing += 1
                continue
            points = np.load(f)[:, :3].astype(np.float64)
            with open(imu_file, 'rb') as fh:
                imu = NumpyUnpickler(fh).load()
            W, H = Image.open(image_file).size
            idx, rc = project(imu_rectify(points, imu), calib, W, H, args.min_depth)
            np.savez(out_root / seq_dir.name / f'{f.stem}.npz', idx=idx, rc=rc)
            label_hist += np.bincount(np.load(label_file)['arr_0'][:, 1].astype(np.int64), minlength=NUM_LABELS)[:NUM_LABELS]
            n_points.append(len(points))
            n_inside.append(len(idx))
        print(f'seq {seq_dir.name}: {len(n_points)} frames ({missing} skipped, missing files), '
              f'{np.mean(n_points):.0f} pts/frame, {100 * np.sum(n_inside) / max(1, np.sum(n_points)):.1f}% in RGB image')
    print('label histogram (0 = Other, dropped): ' + ', '.join(f'{i}={h}' for i, h in enumerate(label_hist)))


if __name__ == '__main__':
    main()
