# This file contains the definition of the EmbedSeg model as introduced in the paper
# `Embedding-based Instance Segmentation in Microscopy` (https://arxiv.org/abs/2101.10033)
import torch
import torch.nn as nn
import torch.nn.functional as F

from erfnet.erfnet import Encoder, Decoder


class EmbedSegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.seed_branch = Decoder(1)  # for the seediness map
        self.instance_branch = Decoder(4)  # channels 0, 1 for x and y sigma, channels 2, 3 for y and x offsets
    
    def forward(self, x):
        z = self.encoder(x)

        instance_branch = self.instance_branch(z)

        offsets_y_map = F.tanh(instance_branch[:, 2:3, :, :])
        offsets_x_map = F.tanh(instance_branch[:, 3:, :, :])
        
        offset_map = torch.cat((offsets_y_map, offsets_x_map), dim=1)

        return F.sigmoid(self.seed_branch(z)), offset_map, F.sigmoid(instance_branch[:, :2, :, :])