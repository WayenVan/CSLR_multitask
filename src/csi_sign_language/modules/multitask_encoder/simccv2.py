"""
more lower weigth simcc header for learning a better visual encoder rather than a better decoder
"""

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


class SimCCHeaderV2(nn.Module):
    """
                                ┌──► Linear x
    final_conv ───► flatten ────┤
                                └──► Linear y

    """

    def __init__(
        self,
        in_channels: int,
        final_layer_kernel_size: int,
        final_layer_channels: int,
        x_samples: int,
        y_samples: int,
    ):
        super().__init__()

        self.final_layer = nn.Conv2d(
            in_channels,
            final_layer_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
        )

        self.mlp_head_x = nn.Linear(in_channels, x_samples)
        self.mlp_head_y = nn.Linear(in_channels, y_samples)

    # NOTE: depreacated for now
