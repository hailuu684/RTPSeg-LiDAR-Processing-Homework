"""Train RTPSegNet (point cloud + RGB) with plain PyTorch; metrics and visualizations go to Weights & Biases."""
import argparse
import importlib
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb

from utils.engine import (load_config, class_info, build_loader, to_device, evaluate, metric_logs, iou_table,
                          frame_collector, frame_logs, confusion_figure, input_2d_figure, to_wandb_image)
from utils.schedulers import cosine_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=None, help='default: train_params.max_num_epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='default: train_data_loader.batch_size')
    parser.add_argument('--val-batch-size', type=int, default=None, help='default: val_data_loader.batch_size')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--no-pin-memory', action='store_true')
    parser.add_argument('--ckpt-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='continue from <ckpt-dir>/last.pt if it exists')
    parser.add_argument('--val-every', type=int, default=1, help='epochs')
    parser.add_argument('--log-every', type=int, default=20, help='iterations')
    parser.add_argument('--num-vis', type=int, default=4, help='val frames logged as BEV images / 3D clouds')
    parser.add_argument('--wandb-project', default='rtpsegnet')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--run-name', default=None)
    return parser.parse_args()


def build_optimizer(config, params, epochs, batch_size):
    tp = config.train_params
    if tp.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=tp.learning_rate)
    elif tp.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=tp.learning_rate, momentum=tp.momentum,
                                    weight_decay=tp.weight_decay, nesterov=tp.nesterov)
    else:
        raise NotImplementedError(tp.optimizer)

    if tp.lr_scheduler == 'CosineAnnealingWarmRestarts':  # upstream name for the per-step cosine decay
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=partial(
            cosine_schedule_with_warmup, num_epochs=epochs, batch_size=batch_size,
            dataset_size=config.dataset_params.training_size, num_gpu=1))
        return optimizer, scheduler, True
    if tp.lr_scheduler == 'CosineAnnealingLR':
        return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 4, eta_min=1e-5), False
    if tp.lr_scheduler == 'StepLR':
        return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, step_size=tp.decay_step, gamma=tp.decay_rate), False
    raise NotImplementedError(tp.lr_scheduler)


def main():
    args = parse_args()
    config = load_config(args.config_path)
    dp = config.dataset_params
    epochs = args.epochs or config.train_params.max_num_epochs
    batch_size = args.batch_size or dp.train_data_loader.batch_size
    val_batch_size = args.val_batch_size or dp.val_data_loader.batch_size
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    names, palette = class_info(config)
    num_classes, ignore_label = config.model_params.num_class, dp.ignore_label

    train_loader = build_loader(config, 'train', batch_size, args.num_workers, not args.no_pin_memory)
    val_loader = build_loader(config, 'val', val_batch_size, args.num_workers, not args.no_pin_memory)
    if dp.training_size is None:
        dp.training_size = len(train_loader.dataset)
    print(f'train frames {len(train_loader.dataset)}, val frames {len(val_loader.dataset)}')

    model = importlib.import_module('network.' + config.model_params.model_architecture).get_model(config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'trainable params {sum(p.numel() for p in params) / 1e6:.2f}M / total {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    optimizer, scheduler, per_step = build_optimizer(config, params, epochs, batch_size)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, step, best_miou, wandb_id = 0, 0, 0.0, None
    if args.resume and (args.ckpt_dir / 'last.pt').exists():
        ckpt = torch.load(args.ckpt_dir / 'last.pt', map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch, step, best_miou, wandb_id = ckpt['epoch'] + 1, ckpt['step'], ckpt['best_miou'], ckpt['wandb_id']
        print(f'resumed from epoch {ckpt["epoch"]}, best mIoU {best_miou:.4f}')

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, id=wandb_id, resume='allow',
               mode=args.wandb_mode, dir=str(args.ckpt_dir), job_type='train',
               config={**config, 'cli': {k: str(v) for k, v in vars(args).items()}})

    def save(path, epoch):
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'epoch': epoch, 'step': step, 'best_miou': best_miou, 'wandb_id': wandb.run.id if wandb.run else None,
                    'config_path': str(args.config_path)}, path)

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            if it == 0:
                wandb.log({'train/input_2d': to_wandb_image(input_2d_figure(batch, palette, dp.image_normalizer))}, step=step)
            batch = to_device(batch, device)
            batch['sparse_label'] = batch['targets'].F.long().reshape(-1)
            batch = model(batch)
            loss = batch['loss']

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)  # upstream: gradient_clip_val=1
            optimizer.step()
            if per_step:
                scheduler.step()
            step += 1

            if step % args.log_every == 0:
                valid = batch['sparse_label'] != ignore_label
                acc = (batch['sparse_logits'].argmax(1)[valid] == batch['sparse_label'][valid]).float().mean().item()
                logs = {'train/loss': loss.item(), 'train/loss_3d_ce': batch['loss_main_ce'].item(),
                        'train/loss_3d_lovasz': batch['loss_main_lovasz'].item(), 'train/acc': acc,
                        'train/lr': optimizer.param_groups[0]['lr'], 'epoch': epoch}
                wandb.log(logs, step=step)
                print(f'epoch {epoch} it {it + 1}/{len(train_loader)} loss {logs["train/loss"]:.4f} '
                      f'acc {acc:.3f} lr {logs["train/lr"]:.5f} ({(time.time() - t0) / (it + 1):.2f} s/it)')
        if not per_step:
            scheduler.step()
        print(f'epoch {epoch} done in {(time.time() - t0) / 60:.1f} min')

        if (epoch + 1) % args.val_every == 0 or epoch + 1 == epochs:
            frames = []
            hist, val_loss = evaluate(model, val_loader, device, num_classes, ignore_label,
                                      on_frame=frame_collector(frames, args.num_vis))
            logs = metric_logs('val', hist, names, val_loss)
            if logs['val/mIoU'] > best_miou:
                best_miou = logs['val/mIoU']
                save(args.ckpt_dir / 'best.pt', epoch)
            logs['val/best_mIoU'] = best_miou
            logs['val/confusion'] = to_wandb_image(confusion_figure(hist, names))
            logs.update(frame_logs('val', frames, palette, ignore_label))
            wandb.log(logs, step=step)
            print(f'epoch {epoch} val:\n{iou_table(hist, names)}\nbest mIoU {100 * best_miou:.2f}')
        save(args.ckpt_dir / 'last.pt', epoch)

    wandb.finish()


if __name__ == '__main__':
    main()
