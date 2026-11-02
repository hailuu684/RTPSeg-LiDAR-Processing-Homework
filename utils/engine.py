"""Shared pieces of train.py / evaluate.py (plain PyTorch): config, loaders, device transfer,
point-level evaluation and Weights & Biases visualizations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from easydict import EasyDict

from dataloader.dataset import get_model_class, get_collate_class
from dataloader.pc_dataset import get_pc_model_class, get_label_name


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    config.setdefault('model_3d', 'SPVCNN')
    config['debug'] = False
    return EasyDict(config)


def class_info(config):
    """class names and RGB palette (uint8) of the num_class training labels (label file keys 1..num_class)"""
    label_name, _ = get_label_name(config.dataset_params.label_mapping)
    with open(config.dataset_params.label_mapping, 'r') as f:
        color_map = yaml.safe_load(f)['color_map']
    C = config.model_params.num_class
    names = [label_name[k] for k in range(1, C + 1)]
    palette = np.full((256, 3), 160, dtype=np.uint8)  # ignore / unknown -> gray
    palette[:C] = [color_map[k][::-1] for k in range(1, C + 1)]  # bgr -> rgb
    return names, palette


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)  # fresh numpy augmentation stream per worker and epoch


def build_loader(config, imageset, batch_size, num_workers, pin_memory, num_vote=1):
    params = config.dataset_params
    loader_cfg = dict(params.train_data_loader if imageset == 'train' else params.val_data_loader)
    if num_vote > 1:  # test-time augmentation: one batch = num_vote augmented copies of one frame
        loader_cfg.update(rotate_aug=True, transform_aug=True)
        batch_size = num_vote
    pc_dataset = get_pc_model_class(params.pc_dataset_type)(config, data_path=loader_cfg['data_path'],
                                                             imageset=imageset, num_vote=num_vote)
    dataset = get_model_class(params.dataset_type)(pc_dataset, config, loader_cfg, num_vote=num_vote)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=imageset == 'train',
                                       num_workers=num_workers, pin_memory=pin_memory, drop_last=imageset == 'train',
                                       collate_fn=get_collate_class(params.collate_type), worker_init_fn=worker_init_fn)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            out[k] = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in v]
        elif hasattr(v, 'to') and not isinstance(v, np.ndarray):  # torch.Tensor or SparseTensor
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def point_outputs(batch):
    """per-frame (logits, labels, xyz) at the original points: voxel logits gathered through inverse_map"""
    lidar_b = batch['lidar'].C[:, -1]
    invs, labels, points = batch['inverse_map'], batch['targets_mapped'], batch['points']
    frames = []
    for b in range(len(batch['root'])):
        inv = invs.F[invs.C[:, -1] == b].long().reshape(-1)
        frames.append((batch['sparse_logits'][lidar_b == b][inv],
                       labels.F[labels.C[:, -1] == b].reshape(-1).long(),
                       points.F[points.C[:, -1] == b][:, :3]))
    return frames


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_label, num_vote=1, on_frame=None):
    """confusion matrix (rows = ground truth) over all scored points, mean loss; on_frame(root, xyz, label, pred)"""
    model.eval()
    hist = torch.zeros(num_classes ** 2, dtype=torch.long, device=device)
    loss_sum = 0.0
    for i, batch in enumerate(loader):
        batch = to_device(batch, device)
        batch['sparse_label'] = batch['targets'].F.long().reshape(-1)
        batch = model(batch)
        loss_sum += batch['loss'].item()
        frames, roots = point_outputs(batch), batch['root']
        if num_vote > 1:  # copies share points and order: sum the logits
            frames, roots = [(sum(f[0] for f in frames), frames[0][1], frames[0][2])], roots[:1]
        for (logits, label, xyz), root in zip(frames, roots):
            pred = logits.argmax(1)
            m = label != ignore_label
            hist += torch.bincount(label[m] * num_classes + pred[m], minlength=num_classes ** 2)
            if on_frame is not None:
                on_frame(root, xyz, label, pred)
        if (i + 1) % 50 == 0:
            print(f'  eval {i + 1}/{len(loader)}')
    return hist.view(num_classes, num_classes).cpu().numpy(), loss_sum / max(len(loader), 1)


def iou_from_hist(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(0) + hist.sum(1) - np.diag(hist))


def metric_logs(prefix, hist, names, loss):
    iou = iou_from_hist(hist)
    logs = {f'{prefix}/mIoU': float(np.nanmean(iou)), f'{prefix}/acc': float(np.diag(hist).sum() / max(hist.sum(), 1)),
            f'{prefix}/loss': loss}
    logs.update({f'{prefix}/IoU/{n}': float(v) for n, v in zip(names, iou)})
    return logs


def iou_table(hist, names):
    iou = iou_from_hist(hist)
    return '\n'.join([f'{n:>28s}: {100 * v:6.2f}' for n, v in zip(names, iou)] + [f'{"mIoU":>28s}: {100 * np.nanmean(iou):6.2f}'])


def frame_collector(frames, num_vis):
    def on_frame(root, xyz, label, pred):
        if len(frames) < num_vis:
            frames.append((root, xyz.cpu().numpy(), label.cpu().numpy(), pred.cpu().numpy()))
    return on_frame


def bev_figure(xyz, label, pred, palette, ignore_label, title, max_points=200000):
    sel = np.random.default_rng(0).permutation(len(xyz))[:max_points]
    sel = sel[np.argsort(xyz[sel, 2])]  # draw high points last
    xyz, label, pred = xyz[sel], label[sel], pred[sel]
    err = np.where((label == pred)[:, None], [[0.85, 0.85, 0.85]], [[0.9, 0.1, 0.1]])
    err[label == ignore_label] = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, colors, name in zip(axes, [palette[label] / 255., palette[pred] / 255., err], ['ground truth', 'prediction', 'errors (red)']):
        ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=0.3, linewidths=0)
        ax.set_aspect('equal')
        ax.set_title(name)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def to_wandb_image(fig, caption=None):
    image = wandb.Image(fig, caption=caption)
    plt.close(fig)
    return image


def point_cloud(xyz, label, palette, max_points=100000):
    sel = np.random.default_rng(0).permutation(len(xyz))[:max_points]
    return wandb.Object3D(np.concatenate([xyz[sel], palette[label[sel]].astype(np.float32)], 1))


def frame_logs(prefix, frames, palette, ignore_label):
    if not frames:
        return {}
    logs = {f'{prefix}/bev': [], f'{prefix}/cloud_gt': [], f'{prefix}/cloud_pred': []}
    for root, xyz, label, pred in frames:
        logs[f'{prefix}/bev'].append(to_wandb_image(bev_figure(xyz, label, pred, palette, ignore_label, root), caption=root))
        logs[f'{prefix}/cloud_gt'].append(point_cloud(xyz, label, palette))
        logs[f'{prefix}/cloud_pred'].append(point_cloud(xyz, pred, palette))
    return logs


def confusion_figure(hist, names):
    norm = hist / np.maximum(hist.sum(1, keepdims=True), 1)
    C = len(names)
    fig, ax = plt.subplots(figsize=(0.6 * C + 3, 0.6 * C + 2))
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(C):
        for j in range(C):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center', fontsize=7, color='w' if norm[i, j] > 0.5 else 'k')
    ax.set_xticks(range(C))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(C))
    ax.set_yticklabels(names)
    ax.set_xlabel('prediction')
    ax.set_ylabel('ground truth (row-normalized)')
    fig.tight_layout()
    return fig


def input_2d_figure(batch, palette, image_normalizer):
    """first sample of a collated train batch: network image + point-pixel pairs colored by label"""
    mean, std = image_normalizer
    img = batch['img'][0].permute(1, 2, 0).cpu().numpy() * np.array(std) + np.array(mean)
    rc = batch['img_indices'][0]
    label = batch['img_label'][:len(rc)].cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(np.clip(img, 0, 1))
    ax.scatter(rc[:, 1], rc[:, 0], c=palette[label] / 255., s=1, linewidths=0)
    ax.set_title(f'{batch["root"][0]}: {len(rc)} point-pixel pairs')
    ax.axis('off')
    fig.tight_layout()
    return fig
