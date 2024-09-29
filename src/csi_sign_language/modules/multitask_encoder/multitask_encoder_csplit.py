import torch
import torch.nn as nn
from collections import namedtuple
from mmpose.models.heads import HeatmapHead
from einops import rearrange
from einops.layers.torch import Reduce

if __name__ == "__main__":
    import sys

    sys.path.append("./src")
    from csi_sign_language.modules.resnet_distill.simcc import SimCCHead
    from csi_sign_language.modules.multitask_encoder.multitask_encoder import (
        VisualBackbone,
    )
else:
    from .simcc import SimCCHead
    from .multitask_encoder import VisualBackbone


class MultiTaskEncoderCsplit(nn.Module):
    """
    support differnet visual backbone for multi task processing

                                                    ┌─────► heatmap ──────► heatmap
                                                    │
                                                    ├─────► SimCCHead ────► x, y
                                                    │
                                                    ├─────► detach ────────┐
                                                    │                      ▼
    visual backbone ─► gap ───► dropout ────► channel split ───────────► concat ──────► out
    """

    def __init__(
        self,
        backbone: VisualBackbone,
        csplit_simcc_ratio: float,
        csplit_heatmap_ratio: float,
        n_keypoints_simcc: int,
        n_keypoints_heatmap: int,
        simcc_x_samples: int,
        simcc_y_samples: int,
        drop_prob=0.1,
        enable_heatmap=True,
        enable_simcc=True,
        detach=True,
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
        self.detach = detach

        if enable_simcc:
            ## SimmCC bracnh
            # NOTE: this is formed H, W, should be revers for the next
            self.csplit_simcc = int(
                csplit_simcc_ratio * self.backbone.get_output_dims()[-1]
            )
            self.pose_header = SimCCHead(
                in_channels=self.csplit_simcc,
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
            self.csplit_heatmap = int(
                csplit_heatmap_ratio * self.backbone.get_output_dims()[-1]
            )
            self.heatmap_header = HeatmapHead(
                in_channels=self.csplit_heatmap,
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

        feats = tuple(rearrange(a, "b c t h w -> (b t) c h w", t=T) for a in feats)
        feat = feats[-1]

        # simcc_out
        if not self.pure_inference and self.enable_simcc and self.csplit_simcc > 0:
            # NOTE: the input should be a list of tensor, only run when the model is not training
            simcc_feat = feat[:, : self.csplit_simcc]
            feat = feat[:, self.csplit_simcc :]
            simcc_out_x, simcc_out_y = self.pose_header([simcc_feat])
            simcc_out_x, simcc_out_y = (
                rearrange(a, "(b t) k l -> t b k l", t=T)
                for a in [simcc_out_x, simcc_out_y]
            )
        else:
            simcc_feat = None
            simcc_out_x, simcc_out_y = None, None

        # heatmap_out
        if not self.pure_inference and self.enable_heatmap and self.csplit_heatmap > 0:
            # NOTE: the input should be a list of tensor
            heatmap_feat = feat[:, : self.csplit_heatmap]
            feat = feat[:, self.csplit_heatmap :]
            heatmap = self.heatmap_header([heatmap_feat])
            heatmap = rearrange(heatmap, "(b t) k h w -> t b k h w", t=T)
        else:
            heatmap_feat = None
            heatmap = None

        # channel fuse
        if simcc_feat is not None:
            feat = torch.cat(
                [feat, simcc_feat.detach() if self.detach else simcc_feat], dim=-3
            )
        if heatmap_feat is not None:
            feat = torch.cat(
                [feat, heatmap_feat.detach() if self.detach else heatmap_feat], dim=-3
            )

        out = rearrange(feat, "(b t) c h w -> b c t h w", t=T)
        out = self.dropout(out)
        out = self.gap(out)

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
    model = MultiTaskEncoderCsplit(
        backbone=backbone,
        csplit_simcc_ratio=0.3,
        csplit_heatmap_ratio=0.2,
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
