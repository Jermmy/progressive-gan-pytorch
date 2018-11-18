import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .base_model import PixelNormLayer, LayerNormLayer


class Generator(nn.Module):

    def __init__(self, target_resolution=1024):
        super(Generator, self).__init__()


    def load_model(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save_model(self, model_file):
        torch.save(self.state_dict(), model_file)