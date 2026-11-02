

import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet34
from utils.lovasz_loss import lovasz_softmax

class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))



class ResNetFCN(nn.Module):
    def __init__(self, backbone="resnet34",pretrained=True,pretrained_load_path=None, config=None):
        super(ResNetFCN, self).__init__()

        if backbone == "resnet34":
            if pretrained:
                state_dict = torch.load(pretrained_load_path)
                net = resnet34(False)
                net.load_state_dict(state_dict)
            else: net = resnet34(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = config['model_params']['hiden_size']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        #self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )

    def forward(self, data_dict):
        x = data_dict['img']
        #x = torch.cat((data_dict['img'],data_dict['infra']),dim=1)
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)

        # Encoder
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)

        data_dict['img_scale2'] = layer1_out
        data_dict['img_scale4'] = layer2_out
        data_dict['img_scale8'] = layer3_out
        data_dict['img_scale16'] = layer4_out

        process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = data_dict['img_indices']

        temp = {k: [] for k in process_keys}

        # root_list=data_dict['root']
        # for i in range(len(root_list)):
        #     root= root_list[i]
        #     save_name= (root.split('/')[-1])[:-4]
        #     save_path = '/'+  os.path.join(*root.split('/')[1:-4])  + '/feats_vis/'+root.split('/')[-3]+'/'
        #     os.makedirs(save_path, exist_ok=True)
        #     for j in range(len(process_keys)):
        #         scale = process_keys[j]
        #         save_name_scale =save_name+'_'+scale
        #         img_feat = data_dict[scale][i,...]
        #         img_feat = img_feat.permute(1, 2, 0)
        #
        #         img_feat = img_feat.view(-1, 64)
        #         pca = PCA(n_components=3, whiten=True)
        #         pca.fit(img_feat.detach().cpu().numpy())
        #         projected_image = torch.from_numpy(pca.transform(img_feat.detach().cpu().numpy())).view(h, w, 3)
        #         projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0))
        #         import matplotlib.pyplot as plt
        #         # enjoy
        #         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        #
        #         # 显示原始RGB
        #         axes[0].imshow(projected_image)
        #         axes[0].set_title('1')
        #         axes[0].axis('off')
        #         gray_from_rgb = 0.2989 * projected_image[:, :, 0] + 0.5870 * projected_image[:, :, 1] + 0.1140 * projected_image[:, :, 2]
        #         # 显示灰度图
        #         axes[1].imshow(gray_from_rgb, cmap='gray')
        #         axes[1].set_title('2')
        #         axes[1].axis('off')
        #
        #         # 显示灰度图
        #         axes[2].imshow(x[i,...].permute(1, 2, 0).detach().cpu().numpy())
        #         axes[2].set_title('3')
        #         axes[2].axis('off')
        #
        #
        #         plt.savefig(save_path+save_name_scale+'.png')
        #         plt.close(fig)  # 关闭图形以释放内存

        for i in range(x.shape[0]):
            for k in process_keys:
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
                #center_x = img_indices[i][:, 0]
                #center_y = img_indices[i][:, 1]
                #mask_1 = np.logical_and(center_x - lenth > 0, center_x + lenth < h)
                #mask_2 = np.logical_and(center_y - lenth > 0, center_y + lenth < w)
                #mask = np.logical_and(mask_1, mask_2)
                #img_indices[i]=img_indices[i][mask]
                #data_dict['point2img_index'][i]=data_dict['point2img_index'][i][mask]
                #center_fearure=[]
                #for j in range(-lenth,lenth+1):
                #    for l in range(-lenth,lenth+1):
                #        center_fearure.append(data_dict[k].permute(0, 2, 3, 1)[i][center_x+j,center_y+l].reshape(-1,1,64))
                #for j in range(center_x.shape[0]):
                #    center_fearure.append(data_dict[k].permute(0, 2, 3, 1)[i][center_x[j] - lenth: center_x[j] + lenth + 1,center_y[j] - lenth: center_y[j] + lenth+1].reshape(1,-1,64))
                #temp[k].append(torch.cat(center_fearure, dim=1))


        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        return data_dict

class DINOFCN(nn.Module):
    """Drop-in for ResNetFCN: frozen DINOv2/v3 ViT (timm) + trainable 1x1 heads.
    Intermediate patch features are bilinearly sampled at the point pixels -> data_dict['img_scale{s}']."""
    def __init__(self, config):
        super(DINOFCN, self).__init__()
        import timm
        mp = config['model_params']
        overlay = dict(file=mp['dino_weights']) if mp.get('dino_weights') else None
        self.backbone = timm.create_model(mp['backbone_2d'], pretrained=True, num_classes=0,
                                          dynamic_img_size=True, pretrained_cfg_overlay=overlay)
        self.backbone.requires_grad_(False)
        self.layers = mp['dino_layers']
        self.scale_list = mp['scale_list']
        assert len(self.layers) == len(self.scale_list), "dino_layers must match scale_list"
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.backbone.embed_dim, mp['hiden_size'], kernel_size=1), nn.ReLU(inplace=True))
            for _ in self.layers])

    def forward(self, data_dict):
        x = data_dict['img']
        h, w = x.shape[2], x.shape[3]
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone.forward_intermediates(x, indices=self.layers, output_fmt='NCHW', intermediates_only=True)

        # pixel centres (row, col) -> grid_sample coords in [-1, 1] (align_corners=False)
        grids = []
        for rc in data_dict['img_indices']:
            rc = torch.from_numpy(rc).to(x.device).float()
            grids.append(torch.stack([(rc[:, 1] + 0.5) / w * 2 - 1, (rc[:, 0] + 0.5) / h * 2 - 1], 1).view(1, 1, -1, 2))

        for scale, head, feat in zip(self.scale_list, self.heads, feats):
            feat = head(feat)
            data_dict['img_scale{}'.format(scale)] = torch.cat(
                [F.grid_sample(feat[i:i + 1], g, mode='bilinear', align_corners=False)[0, :, 0].t() for i, g in enumerate(grids)], 0)

        return data_dict


class ResNetFCN_infra(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, pretrained_load_path=None,config=None):
        super(ResNetFCN_infra, self).__init__()


        if backbone == "resnet34":
            if pretrained:
                state_dict = torch.load(pretrained_load_path)
                net = resnet34(False)
                net.load_state_dict(state_dict)
            else: net = resnet34(pretrained)
        else:
            raise NotImplementedError("invalid backbone: {}".format(backbone))
        self.hiden_size = config['model_params']['hiden_size']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )

    def forward(self, data_dict):
        x = data_dict['infra']
        x = x.repeat(1,3,1,1)
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)

        # Encoder
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)

        data_dict['infra_scale2'] = layer1_out
        data_dict['infra_scale4'] = layer2_out
        data_dict['infra_scale8'] = layer3_out
        data_dict['infra_scale16'] = layer4_out

        process_keys = [k for k in data_dict.keys() if k.find('infra_scale') != -1]
        img_indices = data_dict['infra_indices']

        temp = {k: [] for k in process_keys}

        # root_list = data_dict['root']
        # for i in range(len(root_list)):
        #     root= root_list[i]
        #     save_name= (root.split('/')[-1])[:-4]
        #     save_path = '/'+  os.path.join(*root.split('/')[1:-4])  + '/feats_vis/'+root.split('/')[-3]+'/'
        #     os.makedirs(save_path, exist_ok=True)
        #     for j in range(len(process_keys)):
        #         scale = process_keys[j]
        #         save_name_scale = save_name + '_' + scale
        #         img_feat = data_dict[scale][i, ...]
        #         img_feat = img_feat.permute(1, 2, 0)
        #
        #         img_feat = img_feat.view(-1, 64)
        #         pca = PCA(n_components=3, whiten=True)
        #         pca.fit(img_feat.detach().cpu().numpy())
        #         projected_image = torch.from_numpy(pca.transform(img_feat.detach().cpu().numpy())).view(h, w, 3)
        #         projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0))
        #         import matplotlib.pyplot as plt
        #         # enjoy
        #         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        #
        #         # 显示原始RGB
        #         axes[0].imshow(projected_image)
        #         axes[0].set_title('1')
        #         axes[0].axis('off')
        #         gray_from_rgb = 0.2989 * projected_image[:, :, 0] + 0.5870 * projected_image[:, :,
        #                                                                      1] + 0.1140 * projected_image[:, :, 2]
        #         # 显示灰度图
        #         axes[1].imshow(gray_from_rgb, cmap='gray')
        #         axes[1].set_title('2')
        #         axes[1].axis('off')
        #
        #         # 显示灰度图
        #         axes[2].imshow(x[i, ...].permute(1, 2, 0).detach().cpu().numpy())
        #         axes[2].set_title('3')
        #         axes[2].axis('off')
        #
        #         plt.savefig(save_path+save_name_scale+'.png')
        #         plt.close(fig)  # 关闭图形以释放内存


        for i in range(x.shape[0]):
            for k in process_keys:
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        return data_dict

class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)