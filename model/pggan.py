import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import numpy as np

from .base_model import PixelNormLayer, LayerNormLayer, \
    ToRgbLayer, GBaseBlock, FromRgbLayer, DBaseBlock, \
    MinibatchStatConcatLayer, WScaleLayer, he_init


class Generator(nn.Module):

    def __init__(self, resolution=1024, output_act='linear', norm='pixelnorm', device=torch.device('cpu')):
        super(Generator, self).__init__()
        self.resolution = resolution
        self.R = int(np.log2(self.resolution))  # resolution level
        self.norm = norm.lower()
        assert resolution == 2 ** self.R and resolution >= 4

        # ==== define model ====
        self.toRgbLayers = nn.ModuleList()
        self.baseBlocks = nn.ModuleList()

        # 1x1 --> 4x4
        self.baseBlocks.append(GBaseBlock(512, 512, kernel_size=4, padding=3, upsample=False, nonlinearity='leaky_relu', param=0.2, norm=self.norm, device=device))
        self.toRgbLayers.append(ToRgbLayer(512, nonlinearity=output_act, device=device))

        # 4x4 --> 1024x1024
        for level in range(2, self.R):
            ic, oc = self.get_channel_num(level), self.get_channel_num(level + 1)
            self.baseBlocks.append(GBaseBlock(ic, oc, nonlinearity='leaky_relu', param=0.2, norm=self.norm, device=device))
            # Keep ToRgbLayer for model of each resolution
            self.toRgbLayers.append(ToRgbLayer(oc, nonlinearity=output_act, device=device))

    def get_channel_num(self, level):
        '''
        16384: 2^(10+4)
        :param level: 2~10
        :return: channel number for current layer
        '''
        return int(min(16384 / (2 ** level), 512))

    def forward(self, x, alpha=1.0):
        assert alpha >= 0.0 and alpha <= 1.0
        for i, level in enumerate(range(2, self.R)):
            x = self.baseBlocks[i](x)

        if alpha < 1.0:
            y = F.interpolate(x, (self.resolution, self.resolution), mode='bilinear')
            y = self.toRgbLayers[-2](y)
            x = self.baseBlocks[-1](x)
            x = self.toRgbLayers[-1](x)
            return (1 - alpha) * y + alpha * x
        else:
            x = self.baseBlocks[-1](x)
            x = self.toRgbLayers[-1](x)
            return x

    def load_model(self, model_file, map_location='cpu'):
        pretrained_dict = torch.load(model_file, map_location)
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            state_dict[k] = v
            # print('load: ', k)
        self.load_state_dict(state_dict)

    def save_model(self, model_file):
        torch.save(self.state_dict(), model_file)


class Discriminator(nn.Module):

    def __init__(self, resolution=1024, device=torch.device('cpu')):
        super(Discriminator, self).__init__()
        self.resolution = resolution
        self.R = int(np.log2(resolution)) # resolution level
        assert resolution == 2 ** self.R and resolution <= 1024

        # ==== define model ====
        self.fromRgbLayers = nn.ModuleList()
        self.baseBlocks = nn.ModuleList()
        self.minibatchStddevLayer = MinibatchStatConcatLayer()

        # self.fromRgbLayers.append(FromRgbLayer(self.get_channel_num(self.R)))
        # 1024x1024
        self.fromRgbLayers.insert(0, FromRgbLayer(self.get_channel_num(self.R), device=device))

        # 512x512 --> 4x4
        for level in range(self.R, 2, -1):
            ic, oc = self.get_channel_num(level), self.get_channel_num(level - 1)
            # Insert small resolution layer in the front
            self.baseBlocks.insert(0, DBaseBlock(ic, oc, device=device))
            # Keep FromRgbLayer for model of each resolution
            self.fromRgbLayers.insert(0, FromRgbLayer(oc, device=device))

        self.baseBlocks.insert(0, nn.Sequential(
            MinibatchStatConcatLayer(),
            DBaseBlock(513, 512, kernel_size=4, padding=0, downsample=False, device=device)
        ))
        self.linear = nn.Linear(512, 1)

        print(self.fromRgbLayers)

    def get_channel_num(self, level):
        '''
        :param level:
        :return:
        '''
        return int(min(16384 / (2 ** level), 512))

    def forward(self, x, alpha=1.0):
        assert alpha >= 0.0 and alpha <= 1.0

        if alpha < 1.0:
            y = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = self.fromRgbLayers[-2](y)
            x = self.fromRgbLayers[-1](x)
            x = self.baseBlocks[-1](x)
            x = (1 - alpha) * y + alpha * x
        else:
            x = self.fromRgbLayers[-1](x)
            x = self.baseBlocks[-1](x)

        for i, level in enumerate(range(self.R, 2, -1), 1):
            x = self.baseBlocks[-1-i](x)

        x = x.view((-1, 512))
        x = self.linear(x)

        return x

    def load_model(self, model_file, map_location='cpu'):
        pretrained_dict = torch.load(model_file, map_location)
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            state_dict[k] = v
            # print('load: ', k)
        self.load_state_dict(state_dict)

    def save_model(self, model_file):
        torch.save(self.state_dict(), model_file)

