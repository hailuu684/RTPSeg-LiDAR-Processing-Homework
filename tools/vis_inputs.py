"""
Visual check of the dataset -> RTPSegNet inputs before training (Toronto3D or RTPSegRGB config).
  tiles_bev.png        (Toronto3D) colored points + estimated trajectory / poses per tile
  frame_<split>_k.png  RGB image, projected labels, BEV labels (+ virtual camera frustum), BEV visibility
  model_input_k.png    what the network gets: cropped/jittered/flipped image + img_indices, augmented voxels
  stats.txt            frame sizes, visibility, loader timing (+ Toronto3D val coverage)
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from dataloader.dataset import get_model_class
from dataloader.pc_dataset import get_pc_model_class
from utils.engine import load_config, class_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--num-frames', type=int, default=4)
    parser.add_argument('--num-timing', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def draw_frustum(ax, hfov, length=25.0):
    for sign in (-1, 1):
        a = sign * np.deg2rad(hfov / 2)
        ax.plot([0, length * np.cos(a)], [0, length * np.sin(a)], 'k--', lw=1)
    ax.plot(0, 0, 'k^', ms=8)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config_path)
    names, palette = class_info(config)
    dp = config.dataset_params
    toronto = 'toronto3d' in dp
    pc_cls = get_pc_model_class(dp.pc_dataset_type)
    pc_sets = {'train': pc_cls(config, dp.train_data_loader.data_path, imageset='train'),
               'val': pc_cls(config, dp.val_data_loader.data_path, imageset='val')}
    stats = [f'frames: train {len(pc_sets["train"])}, val {len(pc_sets["val"])}']

    # ---- Toronto3D tiles + trajectory ----
    if toronto:
        tiles = {**pc_sets['train'].tiles, **pc_sets['val'].tiles}
        fig, axes = plt.subplots(1, len(tiles), figsize=(7 * len(tiles), 7))
        for ax, (name, tile) in zip(np.atleast_1d(axes), sorted(tiles.items())):
            sub = slice(None, None, max(1, len(tile['xyz']) // 300000))
            ax.scatter(tile['xyz'][sub, 0], tile['xyz'][sub, 1], c=tile['rgb'][sub] / 255., s=0.2, linewidths=0)
            poses = tile['poses']
            ax.plot(poses[:, 0], poses[:, 1], 'r.-', ms=2, lw=0.8, label=f'poses ({len(poses)})')
            q = poses[::max(1, len(poses) // 25)]
            ax.quiver(q[:, 0], q[:, 1], np.cos(q[:, 3]), np.sin(q[:, 3]), color='r', scale=30)
            ax.set_title(f'{name}: {len(tile["xyz"])} pts')
            ax.set_aspect('equal')
            ax.legend(loc='upper right')
            stats.append(f'{name}: {len(tile["xyz"])} points, {len(poses)} poses, '
                         f'label hist (unclassified first) {np.bincount(tile["label"], minlength=9).tolist()}')
        fig.tight_layout()
        fig.savefig(args.out_dir / 'tiles_bev.png', dpi=110)
        plt.close(fig)

    # ---- raw frames: image + point-pixel correspondences ----
    handles = [plt.Line2D([], [], ls='', marker='o', color=palette[c] / 255., label=n) for c, n in enumerate(names)]
    for split, ds in pc_sets.items():
        for k, index in enumerate(np.random.choice(len(ds), args.num_frames, replace=False)):
            data, root = ds[index]
            xyz, labels, vis = data['xyz'], data['labels'], data['img_visible']
            rc = data['img_rc'][vis]
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            axes[0, 0].imshow(data['img'])
            axes[0, 0].set_title(f'{split} {root}: RGB image {data["img"].size}')
            axes[0, 1].imshow(data['img'])
            axes[0, 1].scatter(rc[:, 1], rc[:, 0], c=palette[labels[vis]] / 255., s=0.3, linewidths=0)
            axes[0, 1].set_title(f'points in image by label ({vis.sum()} / {len(xyz)})')
            o = np.argsort(xyz[:, 2])
            axes[1, 0].scatter(xyz[o, 0], xyz[o, 1], c=palette[labels[o]] / 255., s=0.2, linewidths=0)
            axes[1, 0].set_title('BEV labels' + (' (x = heading; gray = scored in another val frame)' if toronto else ''))
            axes[1, 0].legend(handles=handles, loc='upper left', fontsize=8, markerscale=1.5)
            axes[1, 1].scatter(xyz[~vis, 0], xyz[~vis, 1], c='lightgray', s=0.2, linewidths=0)
            axes[1, 1].scatter(xyz[vis, 0], xyz[vis, 1], c='r', s=0.2, linewidths=0)
            axes[1, 1].set_title('BEV: red = has a pixel in the RGB image')
            for ax in axes[1]:
                ax.set_aspect('equal')
                if toronto:
                    draw_frustum(ax, dp.toronto3d.hfov)
            fig.tight_layout()
            fig.savefig(args.out_dir / f'frame_{split}_{k}.png', dpi=100)
            plt.close(fig)
            stats.append(f'{split} {root}: {len(xyz)} points, {vis.sum()} in image ({100 * vis.mean():.1f}%), '
                         f'scored {(labels != 255).sum()}')

    # ---- model inputs after the RGBPC wrapper (train augmentation on) ----
    wrapper = get_model_class(dp.dataset_type)(pc_sets['train'], config, dp.train_data_loader)
    mean, std = dp.image_normalizer
    for k, index in enumerate(np.random.choice(len(wrapper), args.num_frames, replace=False)):
        d = wrapper[index]
        img = np.clip(d['img'] * np.array(std) + np.array(mean), 0, 1)
        voxels, targets = d['lidar'].F, d['targets'].F.reshape(-1)
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        axes[0].imshow(img)
        axes[0].scatter(d['img_indices'][:, 1], d['img_indices'][:, 0], c=palette[d['img_label'].reshape(-1)] / 255.,
                        s=1, linewidths=0)
        axes[0].set_title(f'2D input {img.shape[1]}x{img.shape[0]}: {len(d["img_indices"])} point-pixel pairs')
        axes[1].scatter(voxels[:, 0], voxels[:, 1], c=palette[targets] / 255., s=0.2, linewidths=0)
        axes[1].set_title(f'3D input after aug: {d["num_voxel"]} voxels / {d["point_num"]} points')
        axes[1].set_aspect('equal')
        fig.tight_layout()
        fig.savefig(args.out_dir / f'model_input_{k}.png', dpi=100)
        plt.close(fig)
        stats.append(f'model input {d["root"]}: voxels {d["num_voxel"]}, points {d["point_num"]}, '
                     f'pairs {len(d["img_indices"])}, img {img.shape}, feat {voxels.shape}')

    # ---- timing ----
    t = time.time()
    for index in np.random.choice(len(wrapper), args.num_timing, replace=False):
        wrapper[index]
    stats.append(f'loader: {(time.time() - t) / args.num_timing:.2f} s / train sample (single process)')

    # ---- Toronto3D val coverage: labeled points farther than frame_radius from every val pose are never scored ----
    if toronto:
        for name, tile in pc_sets['val'].tiles.items():
            labeled = tile['label'] != 0
            dist = cKDTree(tile['poses'][:, :2]).query(tile['xyz'][labeled, :2])[0]
            stats.append(f'val {name}: {100 * (dist < dp.toronto3d.frame_radius).mean():.2f}% of labeled points scored '
                         f'(max distance to trajectory {dist.max():.1f} m)')

    (args.out_dir / 'stats.txt').write_text('\n'.join(stats) + '\n')
    print('\n'.join(stats))


if __name__ == '__main__':
    main()
