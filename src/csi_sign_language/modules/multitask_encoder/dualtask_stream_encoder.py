import torch
import torch.nn as nn
from collections import namedtuple
from mmpose.models.heads import HeatmapHead
from einops import rearrange, reduce
from einops.layers.torch import Reduce
from torchvision.transforms import functional as VF


if __name__ == "__main__":
    import sys

    sys.path.append("./src")
    from csi_sign_language.modules.resnet_distill.simcc import SimCCHead
    from csi_sign_language.modules.multitask_encoder.base import VisualBackbone
    from csi_sign_language.modules.multitask_loss.dwpose_wrapper import DWPoseWarpper
else:
    from .simcc import SimCCHead
    from .base import VisualBackbone
    from ..multitask_loss.dwpose_wrapper import DWPoseWarpper


class FusionA(nn.Module):
    def __init__(self, num_keypoints, feats_dim, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feats_dim = feats_dim

        self.norm = nn.LayerNorm(2 * num_keypoints)
        self.keypoints_proj = nn.Sequential(
            nn.Linear(2 * num_keypoints, hidden_dim), nn.Linear(hidden_dim, feats_dim)
        )

        self.gate_preproj = nn.Linear(2 * num_keypoints, feats_dim)
        self.gate_estimator = nn.Sequential(
            nn.Linear(2 * feats_dim, 2 * feats_dim), nn.Sigmoid()
        )

    def forward(self, x, y, feats):
        """
        @param x: (b, k, w)
        @param y: (b, k, h)
        @param feats: (b, c)
        """
        with torch.no_grad():
            x = torch.argmax(x, dim=-1).to(feats.dtype)
            y = torch.argmax(y, dim=-1).to(feats.dtype)
            xy = torch.stack([x, y], dim=-1)
            # TODO: select several keypoints
            xy = rearrange(xy, "b k xy -> b (k xy)").detach()
        xy = self.norm(xy)

        gate_proj = self.gate_preproj(xy)
        gate = self.gate_estimator(torch.cat([gate_proj, feats], dim=-1))
        gate_keypoints = gate[:, : self.feats_dim]
        gate_feats = gate[:, self.feats_dim :]

        feats = gate_feats * feats + gate_keypoints * self.keypoints_proj(xy)
        return feats


class DualTaskStreamEncoder(nn.Module):
    """
    visual backbone ──────► feats1 ──────┐
                                         │
    pose estimator ────┬──► keypoints────┼────► fusion ─────► feats
                       │                 │
                       └──► feats2───────┘
    """

    def __init__(
        self,
        visual_backbone: VisualBackbone,
        fusion: FusionA,
        pose_cfg: str,
        pose_ckpt: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = visual_backbone
        self.fusion = fusion
        self.pose = DWPoseWarpper(pose_cfg, pose_ckpt)

    DualTaskStreamEncoderOut = namedtuple(
        "DualTaskStreamEncoderOut", ["out", "t_length"]
    )

    def forward(self, x, t_length):
        """
        @param x: (b, c, t, h, w)
        @param t_length: (b,)
        @return: out (b, c, t), simcc_out_x (t, b, k, l), simcc_out_y (t, b, k, l), t_length (b,)
        """
        T = x.size(2)
        feats1, t_length = self.backbone(x, t_length)
        feats1 = rearrange(feats1[-1], "b c t h w -> (b t) c h w")
        feats1 = reduce(feats1, "b c h w -> b c", "mean")

        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = VF.resize(x, [256, 192])
        with torch.no_grad():
            logit_x, logit_y = self.pose(x)
        feats = self.fusion(logit_x, logit_y, feats1)
        feats = rearrange(feats, "(b t) c -> b c t", t=T)
        return self.DualTaskStreamEncoderOut(feats, t_length)


if __name__ == "__main__":
    from csi_sign_language.modules.multitask_encoder.base import VisualBackbone
    from csi_sign_language.modules.multitask_encoder.visual_backbones.timm_resnet import (
        TimmResNetBackbone,
    )

    visual_backbone = TimmResNetBackbone("resnet18")
    fusion = FusionA(144, 512, 1024)
    model = DualTaskStreamEncoder(
        visual_backbone,
        fusion,
        "/root/projects/sign_language_multitask/resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py",
        "/root/projects/sign_language_multitask/resources/dwpose-l/dw-ll_ucoco.pth",
    )
    x = torch.rand(2, 3, 16, 224, 224)
    t_length = torch.tensor([16, 16])
    out = model(x, t_length)
    print(out.shape)
