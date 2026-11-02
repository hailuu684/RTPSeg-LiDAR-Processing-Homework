"""One train step (forward / backward / optimizer) + one eval batch on real data: catches setup errors before a long run."""
import argparse
import importlib
import time
from pathlib import Path

import torch

from utils.engine import load_config, build_loader, to_device, point_outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_path)
    config.dataset_params.training_size = 1000
    device = torch.device(args.device)
    model = importlib.import_module('network.' + config.model_params.model_architecture).get_model(config).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    batch = next(iter(build_loader(config, 'train', args.batch_size, args.num_workers, True)))
    print(f'train batch: lidar {tuple(batch["lidar"].F.shape)}, img {tuple(batch["img"].shape)}, '
          f'pairs {[len(i) for i in batch["img_indices"]]}')
    torch.cuda.reset_peak_memory_stats(device)
    t = time.time()
    model.train()
    batch = to_device(batch, device)
    batch['sparse_label'] = batch['targets'].F.long().reshape(-1)
    batch = model(batch)
    optimizer.zero_grad()
    batch['loss'].backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    print(f'train step ok: loss {batch["loss"].item():.4f}, layer_1/2/3 {[tuple(batch[f"layer_{i}"].shape) for i in (1, 2, 3)]}, '
          f'img_scale2 {tuple(batch["img_scale2"].shape)}, {time.time() - t:.2f} s, '
          f'peak mem {torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f} GiB')

    model.eval()
    with torch.no_grad():
        batch = to_device(next(iter(build_loader(config, 'val', args.batch_size, args.num_workers, True))), device)
        batch['sparse_label'] = batch['targets'].F.long().reshape(-1)
        batch = model(batch)
        for (logits, label, xyz), root in zip(point_outputs(batch), batch['root']):
            print(f'eval {root}: {len(xyz)} points, logits {tuple(logits.shape)}, labels {tuple(label.shape)}')
    print('smoke test passed')


if __name__ == '__main__':
    main()
