"""Evaluate a RTPSegNet checkpoint (point-level mIoU, optional test-time voting), save metrics / predictions, log to W&B."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import wandb

from utils.engine import (load_config, class_info, build_loader, evaluate, metric_logs, iou_table, iou_from_hist,
                          frame_collector, frame_logs, confusion_figure, to_wandb_image)

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1, help='ignored when --num-vote > 1 (batch = votes)')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--no-pin-memory', action='store_true')
    parser.add_argument('--num-vote', type=int, default=1, help='>1: sum logits over rotated / translated copies')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--save-pred', action='store_true', help='save <out-dir>/pred/<frame>.npy: xyz, label, pred')
    parser.add_argument('--num-vis', type=int, default=8)
    parser.add_argument('--wandb-project', default='rtpsegnet')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--run-name', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_path)
    device = torch.device(args.device)
    names, palette = class_info(config)
    num_classes, ignore_label = config.model_params.num_class, config.dataset_params.ignore_label

    loader = build_loader(config, args.split, args.batch_size, args.num_workers, not args.no_pin_memory, num_vote=args.num_vote)
    model = importlib.import_module('network.' + config.model_params.model_architecture).get_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f'{args.checkpoint}: epoch {ckpt["epoch"]}, best val mIoU during training {ckpt["best_miou"]:.4f}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, mode=args.wandb_mode,
               dir=str(args.out_dir), job_type='eval', config={**config, 'cli': {k: str(v) for k, v in vars(args).items()}})

    frames = []
    collect = frame_collector(frames, args.num_vis)

    def on_frame(root, xyz, label, pred):
        collect(root, xyz, label, pred)
        if args.save_pred:
            out = args.out_dir / 'pred' / f'{root}.npy'
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, np.concatenate([xyz.cpu().numpy(), label.cpu().numpy()[:, None], pred.cpu().numpy()[:, None]], 1))

    hist, loss = evaluate(model, loader, device, num_classes, ignore_label, num_vote=args.num_vote, on_frame=on_frame)
    logs = metric_logs(args.split, hist, names, loss)
    print(iou_table(hist, names))

    np.save(args.out_dir / 'confusion.npy', hist)
    (args.out_dir / 'metrics.json').write_text(json.dumps({k: v for k, v in logs.items()}, indent=2))
    table = wandb.Table(columns=['class', 'IoU'], data=[[n, float(v)] for n, v in zip(names, iou_from_hist(hist))])
    logs[f'{args.split}/iou_table'] = wandb.plot.bar(table, 'class', 'IoU', title=f'{args.split} per-class IoU')
    logs[f'{args.split}/confusion'] = to_wandb_image(confusion_figure(hist, names))
    logs.update(frame_logs(args.split, frames, palette, ignore_label))
    wandb.log(logs)
    wandb.summary.update({k: v for k, v in logs.items() if isinstance(v, float)})
    wandb.finish()


if __name__ == '__main__':
    main()
