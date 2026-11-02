"""
Convert the authors' RTPSegNet Lightning checkpoint (torchsparse SPVCNN, trained with RGB + TIR) to this repo's spconv
model. Only model_3d is converted: RTPSegNet's 2D branches / fusion are used for training only, inference runs model_3d.

Kernel layouts (both libraries index kernels as "input = output (* stride) + offset", like torch Conv3d):
  torchsparse v1.4  kernel (K, in, out); odd k: offsets {-1,0,1}, x fastest (z outermost); even k: offsets {0,1}, x outermost
  spconv 2.x        weight KRSC (out, kx, ky, kz, in) or RSKC (kx, ky, kz, out, in) for the Native algorithm
"""
import argparse
import importlib
import re
from pathlib import Path

import numpy as np
import torch

from utils.engine import load_config

DOWN_UP = re.compile(r'^(model_3d\.(?:stage\d|up\d)\.0)\.(\d+\.\w+)$')  # ours: stage1.0.0.weight, theirs: stage1.0.net.0.kernel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, required=True, help='authors\' Lightning .ckpt')
    parser.add_argument('--config-path', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def load_lightning_ckpt(path):
    """weights_only load (no code execution); classes we do not have (e.g. pytorch_lightning callback keys) become inert stubs"""
    try:
        from easydict import EasyDict
        torch.serialization.add_safe_globals([EasyDict])
    except ImportError:
        pass
    for _ in range(50):
        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except Exception as e:
            m = re.search(r'GLOBAL ([\w\.]+) was not an allowed global', str(e))
            if m is None:
                raise
            module, name = m.group(1).rsplit('.', 1)
            stub = type(name, (), {'__module__': module, '__qualname__': name,
                                   '__new__': lambda cls, *a, **k: object.__new__(cls),
                                   '__setstate__': lambda self, state: self.__dict__.update(state=state)})
            torch.serialization.add_safe_globals([stub])
            print(f'  stub for {m.group(1)}')
    raise RuntimeError(f'{path}: too many unknown globals')


def ts_kernel_to_spconv(kernel, target_shape):
    K, cin, cout = kernel.shape
    k = round(K ** (1 / 3))
    assert k ** 3 == K, kernel.shape
    w = kernel.reshape(k, k, k, cin, cout)
    if k % 2 == 1:
        w = w.permute(2, 1, 0, 3, 4)  # [z, y, x] -> [x, y, z]
    # w[kx, ky, kz, in, out]
    if tuple(target_shape) == (cout, k, k, k, cin):
        return w.permute(4, 0, 1, 2, 3).contiguous()  # KRSC
    if tuple(target_shape) == (k, k, k, cout, cin):
        return w.permute(0, 1, 2, 4, 3).contiguous()  # RSKC
    raise ValueError(f'unexpected spconv weight shape {tuple(target_shape)} for torchsparse kernel {tuple(kernel.shape)}')


def main():
    args = parse_args()
    config = load_config(args.config_path)
    assert config.baseline_only, 'use a config with baseline_only: True (3D branch only)'
    model = importlib.import_module('network.' + config.model_params.model_architecture).get_model(config)
    target = model.state_dict()

    ckpt = load_lightning_ckpt(args.src)
    src = ckpt.get('state_dict', ckpt)
    print(f'{args.src}: epoch {ckpt.get("epoch")}, global_step {ckpt.get("global_step")}, '
          f'Lightning {ckpt.get("pytorch-lightning_version")}, {len(src)} tensors')
    hparams = ckpt.get('hyper_parameters')
    hp = hparams.get('config') if isinstance(hparams, dict) else None
    if isinstance(hp, dict):
        print(f'  authors\' config: image_only {hp.get("image_only")}, infra_only {hp.get("infra_only")}, '
              f'image_and_infra {hp.get("image_and_infra")}')
        for group, key in [('model_params', 'num_class'), ('model_params', 'input_dims'), ('model_params', 'voxel_size'),
                           ('model_params', 'layer_num'), ('model_params', 'cr'), ('dataset_params', 'max_volume_space'),
                           ('dataset_params', 'min_volume_space')]:
            theirs, ours = hp[group][key], config[group][key]
            same = np.allclose(np.atleast_1d(np.asarray(theirs, dtype=float)), np.atleast_1d(np.asarray(ours, dtype=float)))
            print(f'  {group}.{key}: {theirs}' + ('' if same else f'  <-- MISMATCH, ours {ours}'))

    converted, used = {}, set()
    for key, value in target.items():
        src_key = DOWN_UP.sub(r'\1.net.\2', key)
        if value.dim() == 5:  # spconv conv weight
            src_key = src_key[:-len('weight')] + 'kernel'
        elif key.endswith('.downsample.0.weight'):  # torchsparse 1x1x1 conv kernel (in, out) -> nn.Linear (out, in)
            src_key = src_key[:-len('weight')] + 'kernel'
        if src_key not in src:
            raise KeyError(f'{src_key} (for {key}) not in checkpoint; first keys: {list(src)[:10]}')
        w = src[src_key]
        if value.dim() == 5:
            w = ts_kernel_to_spconv(w, value.shape)
        elif key.endswith('.downsample.0.weight'):
            w = w.t().contiguous()
        assert w.shape == value.shape, (key, tuple(w.shape), tuple(value.shape))
        converted[key] = w.to(value.dtype)
        used.add(src_key)
    model.load_state_dict(converted, strict=True)

    unused = sorted({k.split('.')[0] for k in src if k not in used})
    print(f'converted {len(converted)} model_3d tensors; not used (training-only branches): {unused}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'epoch': ckpt.get('epoch', -1), 'step': ckpt.get('global_step', -1),
                'best_miou': float('nan'), 'source': str(args.src), 'config_path': str(args.config_path)}, args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
