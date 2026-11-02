"""
Toronto-3D -> RTPSegNet input (expects the unzipped tiles under <data-root>/raw).
1) per tile L00x.ply: subtract UTM offset, keep one random original point per grid cell (exact labels / colors)
2) estimate the vehicle trajectory from GPS time (median of the ground returns of each short time bin),
   resample it every --pose-spacing m -> poses (x, y, z_ground, yaw) used as LiDAR-frame centres / virtual cameras
3) save <out-dir>/<tile>.npz
"""
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData

UTM_OFFSET = np.array([627285.0, 4841948.0, 0.0])  # offset suggested by the Toronto-3D authors; keeps float32 precise
LABEL_NAMES = ['unclassified', 'road', 'road_marking', 'natural', 'building', 'utility_line', 'pole', 'car', 'fence']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--grid-size', type=float, default=0.05)
    parser.add_argument('--pose-spacing', type=float, default=1.0)
    parser.add_argument('--bin-dt', type=float, default=0.2, help='GPS-time bin (s) for one trajectory sample')
    parser.add_argument('--ground-band', type=float, default=0.7, help='m above the bin minimum counted as ground')
    parser.add_argument('--min-bin-points', type=int, default=200)
    parser.add_argument('--smooth-bins', type=int, default=9)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def field(vertex, *candidates):
    names = {n.lower(): n for n in vertex.data.dtype.names}
    for c in candidates:
        if c.lower() in names:
            return np.asarray(vertex[names[c.lower()]])
    raise KeyError(f'none of {candidates} in PLY fields {vertex.data.dtype.names}')


def read_tile(path):
    vertex = PlyData.read(str(path))['vertex']
    print(f'{path.name}: {vertex.count} points, fields {vertex.data.dtype}')
    xyz = np.stack([field(vertex, 'x'), field(vertex, 'y'), field(vertex, 'z')], 1).astype(np.float64)
    if np.abs(xyz[:, :2]).mean() > 1e5:  # raw UTM coordinates
        xyz -= UTM_OFFSET
    rgb = np.stack([field(vertex, 'red'), field(vertex, 'green'), field(vertex, 'blue')], 1)
    if rgb.max() > 255:  # 16-bit colors
        rgb = rgb / 257.0
    gps_time = field(vertex, 'scalar_GPSTime', 'gps_time', 'gpstime').astype(np.float64)
    print(f'  gps_time: {gps_time.min():.3f} .. {gps_time.max():.3f} s, {len(np.unique(gps_time))} unique values')
    return (xyz, rgb.astype(np.uint8), field(vertex, 'scalar_Intensity', 'intensity').astype(np.float32),
            gps_time, field(vertex, 'scalar_Label', 'label', 'class').astype(np.uint8))


def grid_subsample(xyz, grid_size, rng):
    """indices of one random original point per grid cell"""
    k = np.floor((xyz - xyz.min(0)) / grid_size).astype(np.int64)
    dims = k.max(0) + 1
    key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    perm = rng.permutation(len(key))
    _, first = np.unique(key[perm], return_index=True)
    return np.sort(perm[first])


def estimate_trajectory(xyz, gps_time, bin_dt, ground_band, min_bin_points, smooth_bins, spacing):
    b = np.floor((gps_time - gps_time.min()) / bin_dt).astype(np.int64)
    order = np.argsort(b, kind='stable')
    bs = b[order]
    starts = np.r_[0, np.nonzero(np.diff(bs))[0] + 1]
    ends = np.r_[starts[1:], len(bs)]
    centres = []
    for s, e in zip(starts, ends):
        if e - s < min_bin_points:
            continue
        p = xyz[order[s:e]]
        g = p[p[:, 2] < np.percentile(p[:, 2], 2) + ground_band]
        centres.append([bs[s] * bin_dt, *np.median(g, 0)])
    c = np.asarray(centres)  # t, x, y, z_ground

    # continuous segments: split at time gaps or spatial jumps
    jump = (np.diff(c[:, 0]) > 2.0) | (np.linalg.norm(np.diff(c[:, 1:3], axis=0), axis=1) > 10.0)
    poses, segment = [], []
    for sid, ids in enumerate(np.split(np.arange(len(c)), np.nonzero(jump)[0] + 1)):
        if len(ids) < smooth_bins:
            continue
        kernel = np.ones(smooth_bins) / smooth_bins
        pts = np.stack([np.convolve(np.pad(c[ids, j], smooth_bins // 2, mode='edge'), kernel, mode='valid')
                        for j in (1, 2, 3)], 1)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))]
        keep = np.r_[True, np.diff(arc) > 1e-3]  # drop stationary duplicates (np.interp needs increasing xp)
        pts, arc = pts[keep], arc[keep]
        if arc[-1] < 10.0:
            continue
        s = np.arange(0, arc[-1], spacing)
        x, y, z = [np.interp(s, arc, pts[:, j]) for j in range(3)]
        ahead = [np.interp(np.minimum(s + 2.0, arc[-1]), arc, pts[:, j]) for j in (0, 1)]
        behind = [np.interp(np.maximum(s - 2.0, 0), arc, pts[:, j]) for j in (0, 1)]
        yaw = np.arctan2(ahead[1] - behind[1], ahead[0] - behind[0])
        poses.append(np.stack([x, y, z, yaw], 1))
        segment.append(np.full(len(s), sid))
        print(f'  segment {sid}: {arc[-1]:.1f} m, {len(s)} poses, t = {c[ids[0], 0]:.1f} .. {c[ids[-1], 0]:.1f} s')
    return np.concatenate(poses).astype(np.float32), np.concatenate(segment).astype(np.int32), c


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    raw_dir = args.data_root / 'raw'
    ply_files = sorted(raw_dir.rglob('L00*.ply'))
    assert ply_files, f'no L00*.ply found under {raw_dir}; run the unzip step first'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ply in ply_files:
        xyz, rgb, intensity, gps_time, label = read_tile(ply)
        poses, segment, centres = estimate_trajectory(xyz, gps_time, args.bin_dt, args.ground_band,
                                                      args.min_bin_points, args.smooth_bins, args.pose_spacing)
        keep = grid_subsample(xyz, args.grid_size, rng)
        hist = np.bincount(label[keep], minlength=len(LABEL_NAMES))
        print(f'  subsampled {len(xyz)} -> {len(keep)} points (grid {args.grid_size} m), {len(poses)} poses')
        print('  labels: ' + ', '.join(f'{n}={h}' for n, h in zip(LABEL_NAMES, hist)))
        np.savez(args.out_dir / f'{ply.stem}.npz',
                 xyz=xyz[keep].astype(np.float32), rgb=rgb[keep], intensity=intensity[keep], label=label[keep],
                 poses=poses, pose_segment=segment, traj_centres=centres.astype(np.float32))


if __name__ == '__main__':
    main()
