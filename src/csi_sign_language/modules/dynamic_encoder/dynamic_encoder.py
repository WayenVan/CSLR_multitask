import torch
from torch import nn


class WeightedDifferentiaBlock(nn.Module):
    """

    Differential ─┬─► conv ────►
                  └─► norm / factor ───► softmax * origin feature
    """

    def __init__(self, factor=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param x: (B, C, T, H, W)
        """
        x0 = x
        x1 = torch.cat([x0[:, :, 0], x0[:, :, 0:-1]], dim=2)
        delta_x = x0 - x1

        # feature-wise amplifire
        norm = torch.norm(delta_x, 2, dim=1, keepdim=True) / self.factor
        weight = torch.softmax(norm, dim=1)

        raise NotImplementedError
