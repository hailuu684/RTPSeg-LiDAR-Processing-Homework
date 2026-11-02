"""
spconv (2.x) port of the torchsparse pieces SPVCNN uses:
- blocks: torchsparse Conv3d(stride 1) -> SubMConv3d, Conv3d(k2, s2) -> SparseConv3d, transposed -> SparseInverseConv3d
- point <-> voxel ops (torchsparse point_to_voxel / voxel_to_point) in plain torch, by coordinate hashing.
Points are the input voxels (integer coords at stride 1); a voxel tensor at stride s has indices floor(p / s).
"""
import itertools

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F


def down_block(inc, outc, level):
    """torchsparse BasicConvolutionBlock(ks=2, stride=2): level-1 -> level"""
    return spconv.SparseSequential(
        spconv.SparseConv3d(inc, outc, kernel_size=2, stride=2, bias=False, indice_key=f'down{level}'),
        nn.BatchNorm1d(outc), nn.ReLU(True))


def up_block(inc, outc, level):
    """torchsparse BasicDeconvolutionBlock(ks=2, stride=2): level -> level-1, restores exactly the level-1 voxels"""
    return spconv.SparseSequential(
        spconv.SparseInverseConv3d(inc, outc, kernel_size=2, bias=False, indice_key=f'down{level}'),
        nn.BatchNorm1d(outc), nn.ReLU(True))


class ResidualBlock(spconv.SparseModule):
    """torchsparse ResidualBlock(ks=3, stride=1); the 1x1x1 shortcut conv is a per-voxel linear layer"""
    def __init__(self, inc, outc, level):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inc, outc, kernel_size=3, bias=False, indice_key=f'subm{level}'),
            nn.BatchNorm1d(outc), nn.ReLU(True),
            spconv.SubMConv3d(outc, outc, kernel_size=3, bias=False, indice_key=f'subm{level}'),
            nn.BatchNorm1d(outc))
        self.downsample = nn.Identity() if inc == outc else nn.Sequential(nn.Linear(inc, outc, bias=False), nn.BatchNorm1d(outc))

    def forward(self, x):
        out = self.net(x)
        return out.replace_feature(F.relu(out.features + self.downsample(x.features)))


def cat_features(x, skip):
    """torchsparse.cat: both tensors live on the same voxels in the same order (inverse conv output vs. skip)"""
    return x.replace_feature(torch.cat([x.features, skip.features], 1))


class PointVoxelMapper:
    """point_to_voxel (scatter mean) and voxel_to_point (trilinear, torchsparse calc_ti_weights) for one batch"""
    def __init__(self, batch_idx, coords):
        self.b, self.p = batch_idx.long(), coords.long()
        self.dims = self.p.max(0).values + 2
        self.cache = {}

    def _hash(self, b, xyz):
        return ((b * self.dims[0] + xyz[:, 0]) * self.dims[1] + xyz[:, 1]) * self.dims[2] + xyz[:, 2]

    def _lookup(self, x, query_xyz):
        """row of x (SparseConvTensor) at each query cell of the points, -1 if the voxel does not exist"""
        idx = x.indices.long()
        keys, order = torch.sort(self._hash(idx[:, 0], idx[:, 1:]))
        q = self._hash(self.b, query_xyz)
        pos = torch.searchsorted(keys, q).clamp(max=len(keys) - 1)
        return torch.where(keys[pos] == q, order[pos], torch.full_like(pos, -1))

    def point_to_voxel(self, x, feats, stride):
        if ('p2v', stride) not in self.cache:
            self.cache[('p2v', stride)] = self._lookup(x, torch.div(self.p, stride, rounding_mode='floor'))
        idx = self.cache[('p2v', stride)]  # every point falls in an existing voxel (voxels come from the points)
        n = x.features.shape[0]
        out = torch.zeros(n, feats.shape[1], device=feats.device, dtype=feats.dtype).index_add_(0, idx, feats)
        cnt = torch.zeros(n, device=feats.device, dtype=feats.dtype).index_add_(0, idx, torch.ones_like(feats[:, 0]))
        return x.replace_feature(out / cnt.clamp(min=1)[:, None])

    def voxel_to_point(self, x, stride):
        if ('v2p', stride) not in self.cache:
            base = torch.div(self.p, stride, rounding_mode='floor')
            frac = (self.p - base * stride).float() / stride
            idx, w = [], []
            for off in itertools.product((0, 1), repeat=3):
                o = torch.tensor(off, device=self.p.device)
                idx.append(self._lookup(x, base + o))
                w.append(torch.prod(torch.where(o.bool(), frac, 1 - frac), 1))
            idx, w = torch.stack(idx, 1), torch.stack(w, 1)
            w[idx < 0] = 0
            w = w / (w.sum(1, keepdim=True) + 1e-8)
            self.cache[('v2p', stride)] = (idx.clamp(min=0), w)
        idx, w = self.cache[('v2p', stride)]
        out = 0
        for k in range(idx.shape[1]):  # accumulate the 8 corners (avoids an N x 8 x C tensor)
            out = out + x.features[idx[:, k]] * w[:, k:k + 1]
        return out
