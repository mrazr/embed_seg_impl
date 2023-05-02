""" This file contains an implementation of the residual block with factorized
    convolutions as proposed in `ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation` (https://ieeexplore.ieee.org/document/8063438)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, dilation: int=1):
        super().__init__()

        self.conv1_v = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding='same')
        self.prelu1_v = nn.PReLU()
        self.conv1_h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu1_h = nn.PReLU()

        self.conv2_v = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding='same', dilation=dilation)
        self.prelu2_v = nn.PReLU()
        self.conv2_h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding='same', dilation=dilation)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.dropout = nn.Dropout(p=0.3)

        self.prelu_out = nn.PReLU()

    def forward(self, x):
        in_x = x

        x = self.prelu1_v(self.conv1_v(x))
        x = self.prelu1_h(self.bn1(self.conv1_h(x)))

        x = self.prelu2_v(self.conv2_v(x))
        x = self.conv2_h(x)

        x = self.dropout(self.bn2(x))

        return self.prelu_out(torch.add(x, in_x))
