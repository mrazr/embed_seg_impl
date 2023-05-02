# This file contains the downsampling block as introduced in the paper `ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation` (https://arxiv.org/abs/1606.02147)

import torch
from torch import nn
import torch.nn.functional as F


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 3, 2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels - in_channels, eps=1e-3)
    
    def forward(self, x):
        conv = self.conv(x)
        pooled = F.max_pool2d(x, 2, 2)
        cat = torch.cat((conv, pooled), dim=1)

        return F.relu(self.bn(cat))
        

