import torch
import torch.nn as nn
import torch.nn.functional as F

from erfnet.erfnet import Encoder, Decoder


class EmbedSegModel(nn.Module):
    def __init__(self, sigma_2d: bool=True):
        super().__init__()
        self.encoder = Encoder()

        self.seed_branch = Decoder(1)
        self.instance_branch = Decoder(3 + int(sigma_2d))
    
    def forward(self, x):
        z = self.encoder(x)

        instance_branch = self.instance_branch(z)

        offset_map = instance_branch[:, :2, :, :]

        return F.sigmoid(self.seed_branch(z)), F.tanh(offset_map), F.sigmoid(instance_branch[:, 2:, :, :])