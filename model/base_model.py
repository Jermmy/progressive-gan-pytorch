import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_normal, calculate_gain


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

def he_init(layer, nonlinearity='conv2d', param=None):
    nonlinearity = nonlinearity.lower()
    if nonlinearity not in ['linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu', 'sigmoid', 'tanh']:
        if not hasattr(layer, 'gain') or layer.gain is None:
            gain = 0  # default
        else:
            gain = layer.gain
    elif nonlinearity == 'leaky_relu':
        assert param is not None, 'Negative_slope(param) should be given.'
        gain = calculate_gain(nonlinearity, param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal(layer.weight, a=gain)


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


class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    """
    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming
        self.scale = (torch.mean(self.incoming.weight.data ** 2)) ** 0.5
        self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        x = self.scale * x
        if self.bias is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % (self.incoming.__class__.__name__)
        return self.__class__.__name__ + param_str


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

    def __init__(self, in_channels, out_channels=3, nonlinearity='linear', param=None):
        super(ToRgbLayer, self).__init__()
        self.nonlinearity = nonlinearity.lower()
        assert self.nonlinearity == 'tanh' or self.nonlinearity == 'linear'

        self.layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]
        he_init(self.layers[-1], nonlinearity, param)
        self.layers += [WScaleLayer(self.layers[-1])]
        if self.nonlinearity == 'tanh':
            self.layers += [nn.Tanh()]
        # self.layers += [nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*self.layers)
        init_params(self.layers)

    def forward(self, x):
        return self.layers(x)


class GBaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 upsample=True, norm='pixelnorm', nonlinearity='leaky_relu', param=0.2):
        super(GBaseBlock, self).__init__()
        self.nonlinearity = nonlinearity.lower()
        assert self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu'
        self.norm = norm.lower()
        if self.norm == 'pixelnorm':
            normLayer = PixelNormLayer()
        elif self.norm == 'batchnorm':
            normLayer = nn.BatchNorm2d()
        else:
            raise NotImplementedError('Norm type %s is not supported.' % self.norm)

        self.upsample = upsample
        self.layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        he_init(self.layers[-1], nonlinearity, param)
        self.layers += [WScaleLayer(self.layers[-1])]

        if self.nonlinearity == 'leaky_relu':
            self.layers += [nn.LeakyReLU(param)]
        elif self.nonlinearity == 'relu':
            self.layers += [nn.ReLU()]

        self.layers += [normLayer]

        self.layers += [nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)]
        he_init(self.layers[-1], nonlinearity, param)
        self.layers += [WScaleLayer(self.layers[-1])]

        if self.nonlinearity == 'leaky_relu':
            self.layers += [nn.LeakyReLU(param)]
        elif self.nonlinearity == 'relu':
            self.layers += [nn.ReLU()]

        self.layers += [normLayer]

        self.layers = nn.Sequential(*self.layers)

        init_params(self.layers)

    def forward(self, x):
        if self.upsample:
            height, width = x.shape[2], x.shape[3]
            x = F.interpolate(x, size=(height * 2, width * 2), mode='nearest')
        return self.layers(x)


class FromRgbLayer(nn.Module):

    def __init__(self, out_channels, in_channels=3, nonlinearity='linear', param=None):
        super(FromRgbLayer, self).__init__()
        self.layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)]
        he_init(self.layers[-1], nonlinearity, param)
        self.layers += [WScaleLayer(self.layers[-1])]
        self.nonlinearity = nonlinearity.lower()
        if self.nonlinearity == 'leaky_relu':
            self.layers += [nn.LeakyReLU(param)]
        self.layers = nn.Sequential(*self.layers)
        init_params(self.layers)

    def forward(self, x):
        return self.layers(x)


class DBaseBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, downsample=True,
                 nonlinearity='leaky_relu', param=0.2):
        super(DBaseBlock, self).__init__()
        self.nonlinearity = nonlinearity.lower()
        assert self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu'
        self.downsample = downsample
        self.layers = []

        self.layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1)]
        he_init(self.layers[-1], self.nonlinearity, param)

        self.layers += [WScaleLayer(self.layers[-1])]

        if self.nonlinearity == 'leaky_relu':
            self.layers += [nn.LeakyReLU(param)]
        elif self.nonlinearity == 'relu':
            self.layers += [nn.ReLU()]

        self.layers += [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        he_init(self.layers[-1], self.nonlinearity, param)

        self.layers += [WScaleLayer(self.layers[-1])]

        if self.nonlinearity == 'leaky_relu':
            self.layers += [nn.LeakyReLU(param)]
        elif self.nonlinearity == 'relu':
            self.layers += [nn.ReLU()]

        self.layers = nn.Sequential(*self.layers)

        init_params(self.layers)

    def forward(self, x):
        x = self.layers(x)
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x



class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, x, y):
        return torch.cat([x, y], 1)


def mean(tensor, axis, **kwargs):
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        tensor = torch.mean(tensor, axis=ax, **kwargs)
    return tensor


class MinibatchStatConcatLayer(nn.Module):
    """Minibatch stat concatenation layer.
    - averaging tells how much averaging to use ('all', 'spatial', 'none')
    """
    def __init__(self, averaging='all'):
        super(MinibatchStatConcatLayer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8) #Tstdeps in the original implementation

    def forward(self, x):
        shape = list(x.size())
        target_shape = shape.copy()
        vals = self.adjusted_std(x, dim=0, keepdim=True)# per activation, over minibatch dim
        if self.averaging == 'all':  # average everything --> 1 value per minibatch
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)#vals = torch.mean(vals, keepdim=True)

        elif self.averaging == 'spatial':  # average spatial locations
            if len(shape) == 4:
                vals = mean(vals, axis=[2,3], keepdim=True)  # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':  # no averaging, pass on all information
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':  # EXPERIMENTAL: compute variance (func) over minibatch AND spatial locations.
            if len(shape) == 4:
                vals = mean(x, [0,2,3], keepdim=True)  # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':  # variance of ALL activations --> 1 value per minibatch
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:  # self.averaging == 'group'  # average everything over n groups of feature maps --> n values per minibatch
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1) # feature-map concatanation

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)
