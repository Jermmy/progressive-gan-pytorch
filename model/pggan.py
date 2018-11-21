import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

from .base_model import PixelNormLayer, LayerNormLayer, ToRgbLayer


class Generator(nn.Module):

    def __init__(self, target_resolution=1024):
        super(Generator, self).__init__()
        self.target_resolution = target_resolution
        self.R = int(np.log2(self.target_resolution))  # resolution level
        assert target_resolution == 2 ** self.R and target_resolution >= 4

        # ==== define model ====
        self.toRgbLayers = []




    def load_model(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save_model(self, model_file):
        torch.save(self.state_dict(), model_file)