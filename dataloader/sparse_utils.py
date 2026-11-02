"""
Pure numpy/torch re-implementation of the torchsparse v1.4 data utilities used by the loaders
(SparseTensor container, sparse_quantize, sparse_collate_fn), same semantics as torchsparse.
SparseTensor here is only a (feats, coords) container; coords = (x, y, z, batch) after collate.
"""
import numpy as np
import torch


class SparseTensor:
    def __init__(self, feats, coords, stride=1):
        self.feats = feats
        self.coords = coords
        self.stride = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
        self.cmaps = {}
        self.kmaps = {}

    @property
    def F(self):
        return self.feats

    @F.setter
    def F(self, feats):
        self.feats = feats

    @property
    def C(self):
        return self.coords

    @C.setter
    def C(self, coords):
        self.coords = coords

    @property
    def s(self):
        return self.stride

    def to(self, device, non_blocking=False):
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')


def ravel_hash(x):
    assert x.ndim == 2, x.shape
    x = (x - np.min(x, axis=0)).astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1
    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(coords, voxel_size=1, *, return_index=False, return_inverse=False):
    voxel_size = np.array((voxel_size,) * 3 if isinstance(voxel_size, (int, float)) else voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)
    _, indices, inverse_indices = np.unique(ravel_hash(coords), return_index=True, return_inverse=True)
    outputs = [coords[indices]]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs


def sparse_collate(inputs):
    """concatenate samples; batch index appended as the last coords column"""
    coords, feats = [], []
    for k, x in enumerate(inputs):
        c = torch.as_tensor(x.coords)
        coords.append(torch.cat((c, torch.full((c.shape[0], 1), k, dtype=c.dtype)), dim=1))
        feats.append(torch.as_tensor(x.feats))
    return SparseTensor(torch.cat(feats, 0), torch.cat(coords, 0), stride=inputs[0].stride)


def sparse_collate_fn(inputs):
    if not isinstance(inputs[0], dict):
        return inputs
    output = {}
    for name, value in inputs[0].items():
        if isinstance(value, dict):
            output[name] = sparse_collate_fn([d[name] for d in inputs])
        elif isinstance(value, np.ndarray):
            output[name] = torch.stack([torch.tensor(d[name]) for d in inputs], dim=0)
        elif isinstance(value, torch.Tensor):
            output[name] = torch.stack([d[name] for d in inputs], dim=0)
        elif isinstance(value, SparseTensor):
            output[name] = sparse_collate([d[name] for d in inputs])
        else:
            output[name] = [d[name] for d in inputs]
    return output
