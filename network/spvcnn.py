import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from network.spconv_utils import down_block, up_block, ResidualBlock, cat_features, PointVoxelMapper
from network.basic_block import Lovasz_loss


class get_model(nn.Module):
    """SPVCNN (same layout as the upstream torchsparse version) on spconv 2.x.
    Level l voxels have stride 2^l w.r.t. the input voxels (voxel_size); level 0 = input voxels."""
    def __init__(self, config):
        super().__init__()
        self.args = config

        cr = config.model_params.cr
        cs = config.model_params.layer_num
        cs = [int(cr * x) for x in cs]

        self.num_classes = config.model_params.num_class

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(config.model_params.input_dims, cs[0], kernel_size=3, bias=False, indice_key='subm0'),
            nn.BatchNorm1d(cs[0]), nn.ReLU(True),
            spconv.SubMConv3d(cs[0], cs[0], kernel_size=3, bias=False, indice_key='subm0'),
            nn.BatchNorm1d(cs[0]), nn.ReLU(True))

        self.stage1 = spconv.SparseSequential(down_block(cs[0], cs[0], 1), ResidualBlock(cs[0], cs[1], 1), ResidualBlock(cs[1], cs[1], 1))
        self.stage2 = spconv.SparseSequential(down_block(cs[1], cs[1], 2), ResidualBlock(cs[1], cs[2], 2), ResidualBlock(cs[2], cs[2], 2))
        self.stage3 = spconv.SparseSequential(down_block(cs[2], cs[2], 3), ResidualBlock(cs[2], cs[3], 3), ResidualBlock(cs[3], cs[3], 3))
        self.stage4 = spconv.SparseSequential(down_block(cs[3], cs[3], 4), ResidualBlock(cs[3], cs[4], 4), ResidualBlock(cs[4], cs[4], 4))

        self.up1 = nn.ModuleList([
            up_block(cs[4], cs[5], 4),
            spconv.SparseSequential(ResidualBlock(cs[5] + cs[3], cs[5], 3), ResidualBlock(cs[5], cs[5], 3))])
        self.up2 = nn.ModuleList([
            up_block(cs[5], cs[6], 3),
            spconv.SparseSequential(ResidualBlock(cs[6] + cs[2], cs[6], 2), ResidualBlock(cs[6], cs[6], 2))])
        self.up3 = nn.ModuleList([
            up_block(cs[6], cs[7], 2),
            spconv.SparseSequential(ResidualBlock(cs[7] + cs[1], cs[7], 1), ResidualBlock(cs[7], cs[7], 1))])
        self.up4 = nn.ModuleList([
            up_block(cs[7], cs[8], 1),
            spconv.SparseSequential(ResidualBlock(cs[8] + cs[0], cs[8], 0), ResidualBlock(cs[8], cs[8], 0))])

        self.classifier = nn.Sequential(nn.Linear(cs[8], self.num_classes))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.criterion = get_loss(config)
        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data_dict):
        x = data_dict['lidar']  # one row per input voxel; C = (x, y, z, batch) voxel coords
        b = x.C[:, -1].long()
        p = x.C[:, :3].long()
        spatial_shape = ((p.max(0).values + 16) // 16 * 16).tolist()  # divisible by 2^4 for the 4 stride-2 levels
        x0 = spconv.SparseConvTensor(x.F.float(), torch.cat([b[:, None], p], 1).int().contiguous(),
                                     spatial_shape, int(b.max().item()) + 1)
        mapper = PointVoxelMapper(b, p)

        # the "points" are the input voxels, so at level 0 point <-> voxel is the identity (as in torchsparse)
        x0 = self.stem(x0)
        z0 = x0.features

        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = mapper.voxel_to_point(x4, 16) + self.point_transforms[0](z0)
        data_dict['layer_1'] = z1

        y1 = mapper.point_to_voxel(x4, z1, 16)
        y1 = y1.replace_feature(self.dropout(y1.features))
        y1 = self.up1[0](y1)
        y1 = self.up1[1](cat_features(y1, x3))

        y2 = self.up2[0](y1)
        y2 = self.up2[1](cat_features(y2, x2))
        z2 = mapper.voxel_to_point(y2, 4) + self.point_transforms[1](z1)
        data_dict['layer_2'] = z2

        y3 = mapper.point_to_voxel(y2, z2, 4)
        y3 = y3.replace_feature(self.dropout(y3.features))
        y3 = self.up3[0](y3)
        y3 = self.up3[1](cat_features(y3, x1))

        y4 = self.up4[0](y3)
        y4 = self.up4[1](cat_features(y4, x0))
        z3 = y4.features + self.point_transforms[2](z2)  # y4 rows = input voxels in input order
        data_dict['layer_3'] = z3

        output = self.classifier(z3)
        data_dict['sparse_logits'] = output
        data_dict = self.criterion(data_dict)

        return data_dict


class get_loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=config['dataset_params']['ignore_label']
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )

    def forward(self, data_dict):
        lovasz_loss = self.lovasz_loss(
            F.softmax(data_dict['sparse_logits'], dim=1),
            data_dict['sparse_label']
        )
        seg_loss = self.ce_loss(data_dict['sparse_logits'], data_dict['sparse_label'])
        total_loss = lovasz_loss + seg_loss
        data_dict['loss'] = total_loss
        data_dict['loss_sparse'] = total_loss
        data_dict['loss_main_ce'] = seg_loss
        data_dict['loss_main_lovasz'] = lovasz_loss

        return data_dict
