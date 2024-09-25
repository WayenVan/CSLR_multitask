import torch
import torch.nn as nn
from collections import namedtuple
from mmpose.models.heads import HeatmapHead
from einops import rearrange
from einops.layers.torch import Reduce
from abc import ABC, abstractmethod
from typing import List, Tuple


if __name__ == "__main__":
    import sys

    sys.path.append("./src")
    from csi_sign_language.modules.resnet_distill.simcc import SimCCHead
else:
    from .simcc import SimCCHead


class VisualBackbone(ABC, nn.Module):
    @abstractmethod
    def get_input_size() -> Tuple[int, int]:
        """
        return the required input size of the visual backbone, (H, W)
        """
        pass

    @abstractmethod
    def get_output_feats_size() -> List[Tuple[int, int]]:
        """
        return  a list of output feats, each corrresponding to a different stage, the element is (H, W)
        """
        pass

    @abstractmethod
    def get_output_dims() -> List[int]:
        """
        @return a list of output dims, each corrresponding to a different stage
        """
        pass


class MultiTaskEncoder(nn.Module):
    """
    support differnet visual backbone for multi task processing

                                             ┌───► Heatmap header ────► heatmap
                                             ├───► Simcc Header ──────► x, y
    visualbackbone ───► dropout ───► gap ────┴───► feature

    """

    def __init__(
        self,
        backbone: VisualBackbone,
        n_keypoints_simcc: int,
        n_keypoints_heatmap: int,
        simcc_x_samples: int,
        simcc_y_samples: int,
        drop_prob=0.1,
        enable_heatmap=True,
        enable_simcc=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        input_size = backbone.get_input_size()
        self.feats_map_size = backbone.get_output_feats_size()[-1]
        self.pure_inference = False

        self.enable_heatmap = enable_heatmap
        self.enable_simcc = enable_simcc

        ## CSLR bracnh
        self.dropout = nn.Dropout3d(drop_prob)
        self.gap = Reduce("b c t h w -> b c t", reduction="mean")

        if enable_simcc:
            ## SimmCC bracnh
            # NOTE: this is formed H, W, should be revers for the next
            self.pose_header = SimCCHead(
                in_channels=self.backbone.get_output_dims()[-1],
                out_channels=n_keypoints_simcc,
                # W H
                input_size=(input_size[1], input_size[0]),
                # W H
                in_featuremap_size=(self.feats_map_size[1], self.feats_map_size[0]),
                simcc_x_samples=simcc_x_samples,
                simcc_y_samples=simcc_y_samples,
            )

        if enable_heatmap:
            ## heatmap bracnh
            self.heatmap_header = HeatmapHead(
                in_channels=self.backbone.get_output_dims()[-1],
                out_channels=n_keypoints_heatmap,
            )

    MultiTaskEncoderOut = namedtuple(
        "MultiTaskEncoderOut",
        ["out", "t_length", "simcc_out_x", "simcc_out_y", "heatmap"],
    )

    def forward(self, x, t_length):
        """
        @param x: (b, c, t, h, w)
        @param t_length: (b,)
        @return: out (b, c, t), simcc_out_x (t, b, k, l), simcc_out_y (t, b, k, l), t_length (b,)
        """
        T = int(x.size(2))

        # NOTE: x is a list of tensors,
        feats, t_length = self.backbone(x, t_length)
        out = feats[-1]
        out = self.dropout(out)
        out = self.gap(out)

        feats = tuple(rearrange(a, "b c t h w -> (b t) c h w", t=T) for a in feats)
        # simcc_out
        if not self.pure_inference and self.enable_simcc:
            # NOTE: the input should be a list of tensor, only run when the model is not training
            simcc_out_x, simcc_out_y = self.pose_header(feats)
            simcc_out_x, simcc_out_y = (
                rearrange(a, "(b t) k l -> t b k l", t=T)
                for a in [simcc_out_x, simcc_out_y]
            )
        else:
            simcc_out_x, simcc_out_y = None, None

        # heatmap_out
        if not self.pure_inference and self.enable_heatmap:
            # NOTE: the input should be a list of tensor
            heatmap = self.heatmap_header(feats)
            heatmap = rearrange(heatmap, "(b t) k h w -> t b k h w", t=T)
        else:
            heatmap = None

        return self.MultiTaskEncoderOut(
            out,
            t_length,
            simcc_out_x,
            simcc_out_y,
            heatmap,
        )


if __name__ == "__main__":
    # a test for ResnetDistEncoder
    from csi_sign_language.modules.multitask_encoder.visual_backbones.resnet import (
        ResNetBackbone,
    )

    backbone = ResNetBackbone(
        cfg="/root/resources/resnet/resnet18.py",
        ckpt="/root/resources/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth",
    )
    model = MultiTaskEncoder(
        backbone=backbone,
        n_keypoints_simcc=144,
        n_keypoints_heatmap=17,
        simcc_x_samples=192 * 2,
        simcc_y_samples=256 * 2,
        # enable_heatmap=False,
        # enable_simcc=False,
    )

    input_data = torch.randn(1, 3, 16, 224, 224)
    output = model(input_data, torch.tensor([16]))

    for name, _ in model.named_parameters():
        print(name)

    for k, v in output._asdict().items():
        if v is None:
            print("none")
            continue
        print(k, v.shape)
