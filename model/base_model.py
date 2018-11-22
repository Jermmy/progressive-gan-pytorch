import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def init_params(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """

    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


def mean(tensor, axis, **kwargs):
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        tensor = torch.mean(tensor, axis=ax, **kwargs)
    return tensor


class LayerNormLayer(nn.Module):
    """
    Layer normalization. Custom reimplementation based on the paper: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, incoming, eps=1e-4):
        super(LayerNormLayer, self).__init__()
        self.incoming = incoming
        self.eps = eps
        self.gain = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.bias = None

        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = x - mean(x, axis=range(1, len(x.size())))
        x = x * 1.0/(torch.sqrt(mean(x**2, axis=range(1, len(x.size())), keepdim=True) + self.eps))
        x = x * self.gain
        if self.bias is not None:
            x += self.bias
        return x

    def __repr__(self):
        param_str = '(incoming = %s, eps = %s)' % (self.incoming.__class__.__name__, self.eps)
        return self.__class__.__name__ + param_str


class ToRgbLayer(nn.Module):

    def __init__(self, in_channels, out_channels=3):
        super(ToRgbLayer, self).__init__()
        self.layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]
        # self.layers += [nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*self.layers)
        init_params(self.layers)

    def forward(self, x):
        return self.layers(x)


class GBaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, upsample=True):
        super(GBaseBlock, self).__init__()
        self.upsample = upsample
        self.layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        self.layers += [nn.LeakyReLU(0.2)]
        self.layers += [nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)]
        self.layers += [nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*self.layers)

        init_params(self.layers)

    def forward(self, x):
        if self.upsample:
            height, width = x.shape[2], x.shape[3]
            x = F.interpolate(x, size=(height * 2, width * 2), mode='nearest')
        return self.layers(x)


class FromRgbLayer(nn.Module):

    def __init__(self, out_channels, in_channels=3):
        super(FromRgbLayer, self).__init__()
        self.layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)]
        self.layers += [nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*self.layers)
        init_params(self.layers)

    def forward(self, x):
        return self.layers(x)


class DBaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, downsample=True):
        super(DBaseBlock, self).__init__()
        self.downsample = downsample
        self.layers = []
        self.layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1)]
        self.layers += [nn.LeakyReLU(0.2)]
        self.layers += [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        self.layers += [nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*self.layers)

        init_params(self.layers)

    def forward(self, x):
        x = self.layers(x)
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x