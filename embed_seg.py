import torch.nn as nn
import torch.nn.functional as F

from erfnet.erfnet import Encoder, Decoder


class EmbedSegModel(nn.Module):
    def __init__(self, num_classses: int, sigma_2d: bool=True):
        super().__init__()
        self.encoder = Encoder()

        self.seed_branch = Decoder(num_classses)
        self.seed_activation = F.softmax if num_classses > 2 else F.sigmoid
        self.instance_branch = Decoder(3 + int(sigma_2d))
    
    def forward(self, x):
        z = self.encoder(x)

        instance_branch = self.instance_branch(z)

        offset_map = instance_branch[:, :2, :, :]

        return self.seed_activation(self.seed_branch(z)), F.tanh(offset_map), F.elu(instance_branch[:, 2:, :, :])