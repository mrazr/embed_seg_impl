"""This file contains the implementation of the ERFNet architecture as proposed in `ERFNet: Efficient Residual Factorized ConvNet
for Real-Time Semantic Segmentation` (https://ieeexplore.ieee.org/document/8063438)"""


import torch.nn as nn

from blocks.enet_downsample_block import DownsampleBlock
from blocks.erfnet_res_block import ResBlock


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            DownsampleBlock(1, 16),
            DownsampleBlock(16, 64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            DownsampleBlock(64, 128),
            ResBlock(128, dilation=2),
            ResBlock(128, dilation=4),
            ResBlock(128, dilation=8),
            ResBlock(128, dilation=16),
            ResBlock(128, dilation=2),
            ResBlock(128, dilation=4),
            ResBlock(128, dilation=8),
            ResBlock(128, dilation=16),
        )
    
    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=3),
            ResBlock(64),
            ResBlock(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, stride=2, kernel_size=3),
            ResBlock(16),
            ResBlock(16),
            nn.ConvTranspose2d(in_channels=16, out_channels=out_channels, stride=2, kernel_size=3)
        )
    
    def forward(self, x):
        return self.block(x)


class ERFNet(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))