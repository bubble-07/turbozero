
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Callable

@dataclass
class ASMRNetConfig:
    feature_maps: int
    layers: int

def reset_model_weights(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.LeakyReLU(),
            nn.Linear(features, features))
        self.leaky_relu = nn.LeakyReLU()
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class ASMRNet(nn.Module):
    def __init__(self, config: ASMRNetConfig, input_shape: torch.Size, output_shape: torch.Size) -> None:
        super().__init__()
        print(input_shape)
        self.input_features, = input_shape

        self.input_layer = nn.Linear(self.input_features, config.feature_maps)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(config.feature_maps) for _ in range(config.layers)])

        self.policy_head = nn.Sequential(
            nn.Linear(config.feature_maps, output_shape[0])
            # we use cross-entropy loss so no need for softmax
        )

        self.value_head = nn.Sequential(
            nn.Linear(config.feature_maps, 1)
        )

        self.config = config

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
