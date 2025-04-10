import torch
import numpy as np
from torch import nn
from torch import Tensor
from mmengine.config import Config
from mmengine.registry.utils import init_default_scope
from mmpose.apis import init_model
from einops import rearrange
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize
from collections import namedtuple

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.modules.multitask_loss.dwpose_wrapper import (
        DWPoseWarpper,
    )
    from csi_sign_language.modules.multitask_loss.vitpose_wrapper import (
        ViTPoseWrapper,
    )
else:
    from .vitpose_wrapper import ViTPoseWrapper
    from .dwpose_wrapper import DWPoseWarpper


MultiTaskDistillLossOut = namedtuple(
    "MultiTaskDistillLossOut", ["out", "ctc_loss", "dwpose_loss", "vit_loss"]
)


class MultiTaskDistillLossHard(nn.Module):
    def __init__(
        self,
        dwpose_cfg: str,
        dwpose_ckpt: str,
        vitpose_cfg: str,
        vitpose_ckpt: str,
        ctc_weight: float = 1.0,
        vitpose_weight: float = 1.0,
        vitpose_keypoint_exp=None,
        vitpose_clip_threshold: float = 0.3,
        dwpose_weight: float = 1.0,
        dwpose_keypoint_exp=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        init_default_scope("mmpose")

        # dwpose related
        if dwpose_weight > 0.0:
            self.dwpose = DWPoseWarpper(dwpose_cfg, dwpose_ckpt)
            self.dwpose_intput_size = (self.dwpose.input_H, self.dwpose.input_W)
            self.dwpose_keypoint_exp = dwpose_keypoint_exp

        # vitpose related
        if vitpose_weight > 0.0:
            self.vitpose = ViTPoseWrapper(vitpose_cfg, vitpose_ckpt)
            self.vitpose_keypoint_exp = vitpose_keypoint_exp
            self.vitpose_clip_threshold = vitpose_clip_threshold

        # ctc related
        if ctc_weight > 0.0:
            self._loss_ctc = nn.CTCLoss(blank=0, reduction="none")

        # weights
        self.ctc_weight = ctc_weight
        self.dwpose_weight = dwpose_weight
        self.vitpose_weight = vitpose_weight

        self._freeze_pose_model()

    def _freeze_pose_model(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, outputs, input, input_length, target, target_length):
        loss = 0.0

        ctc_loss = None
        if self.ctc_weight > 0.0:
            out = nn.functional.log_softmax(outputs.out, dim=-1)
            ctc_loss = self._loss_ctc(
                out, target, outputs.t_length.cpu().int(), target_length.cpu().int()
            ).mean()
            loss += self.ctc_weight * ctc_loss

        dwpose_loss = None
        if self.dwpose_weight > 0.0:
            dwpose_loss = self.distill_loss_simcc(
                outputs.encoder_out.simcc_out_x, outputs.encoder_out.simcc_out_y, input
            )
            loss += self.dwpose_weight * dwpose_loss

        vit_loss = None
        if self.vitpose_weight > 0.0:
            vit_loss = self.distill_loss_heatmap(outputs.encoder_out.heatmap, input)
            loss += self.vitpose_weight * vit_loss

        return MultiTaskDistillLossOut(
            out=loss, ctc_loss=ctc_loss, dwpose_loss=dwpose_loss, vit_loss=vit_loss
        )

    def distill_loss_heatmap(self, heatmap: Tensor, input: Tensor):
        """
        @param heatmap the output heatmap from encoder of size [t b k h w]
        @pararm input: [b c t h w]
        """
        out_heatmap = rearrange(heatmap, "t b k h w -> (b t) k h w")
        input = rearrange(input, "b c t h w -> (b t) c h w")

        target_heatmap = self.vitpose(input)
        with torch.no_grad():
            target_heatmap = torch.where(
                target_heatmap < self.vitpose_clip_threshold,
                torch.zeros_like(target_heatmap, device=target_heatmap.device),
                torch.ones_like(target_heatmap, device=target_heatmap.device),
            )

        # assertion
        assert out_heatmap.shape == target_heatmap.shape

        return nn.functional.mse_loss(out_heatmap, target_heatmap).mean()

    def distill_loss_simcc(
        self, out_logits_x: Tensor, out_logits_y: Tensor, input: Tensor
    ):
        """
        @param out_x, out_y, x, y: SimCC output logits [t b k l]
        @pararm input: [b c t h w]
        """
        T = out_logits_x.shape[0]
        B = out_logits_x.shape[1]
        K = out_logits_x.shape[2]
        input = rearrange(input, "b c t h w -> (b t) c h w")
        input = resize(input, list(self.dwpose_intput_size))

        target_logits_x, target_logits_y = self.dwpose(input)
        out_logits_x, out_logits_y = (
            rearrange(a, "t b k l -> (b t) k l") for a in (out_logits_x, out_logits_y)
        )

        # remove keypoints that is not inlucded the value
        if self.dwpose_keypoint_exp is not None:

            def exp_keypoint(logits, keypoint_exp):
                mask = torch.ones(
                    logits.shape[-2], device=logits.device, dtype=torch.bool
                )
                mask[keypoint_exp] = False
                return logits[:, mask]

            target_logits_x = exp_keypoint(target_logits_x, self.dwpose_keypoint_exp)
            target_logits_y = exp_keypoint(target_logits_y, self.dwpose_keypoint_exp)

        # assertion
        assert target_logits_x.shape == out_logits_x.shape
        assert target_logits_y.shape == out_logits_y.shape

        # x, y are all logits here
        # log_softmax
        target_logits_x, target_logits_y = (
            torch.argmax(a, dim=-1) for a in (target_logits_x, target_logits_y)
        )
        out_logits_x, out_logits_y = (
            nn.functional.log_softmax(a, dim=-1) for a in (out_logits_x, out_logits_y)
        )

        # distillation with temperature
        loss_pointwise_x = nn.functional.nll_loss(
            out_logits_x.flatten(0, 1),
            target_logits_x.detach().flatten(),
            reduction="none",
        )
        loss_pointwise_y = nn.functional.nll_loss(
            out_logits_y.flatten(0, 1),
            target_logits_y.detach().flatten(),
            reduction="none",
        )
        return loss_pointwise_x.sum() / (B * T) + loss_pointwise_y.sum() / (B * T)


if __name__ == "__main__":
    device = "cpu"
    loss = MultiTaskDistillLossHard(
        dwpose_cfg="/root/projects/sign_language_transformer/resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py",
        dwpose_ckpt="/root/projects/sign_language_transformer/resources/dwpose-l/dw-ll_ucoco.pth",
        vitpose_cfg="resources/ViTPose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
        vitpose_ckpt="resources/ViTPose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth",
        ctc_weight=1.0,
        dwpose_weight=10.0,
        vitpose_weight=1.0,
        dwpose_keypoint_exp=[0, 1, 2, 3],
        vitpose_clip_threshold=0.5,
    ).to(device)
    classes = 100
    B = 2
    T = 100
    L = 5
    H = 224
    W = 224
    K_dwpose = 133 - 4
    K_vitpose = 17

    # Dummy data for testing
    outputs = type(
        "Outputs",
        (object,),
        {
            "out": torch.randn(T // 4, B, classes).to(device),  # Example shape
            "t_length": torch.tensor([T // 4] * B).to(device),
            "encoder_out": type(
                "EncoderOut",
                (object,),
                {
                    "simcc_out_x": torch.randn(T, B, K_dwpose, 192 * 2).to(
                        device
                    ),  # Example shape
                    "simcc_out_y": torch.randn(T, B, K_dwpose, 256 * 2).to(
                        device
                    ),  # Example shape
                    "heatmap": torch.randn(T, B, K_vitpose, H // 4, W // 4).to(
                        device
                    ),  # Example shape
                },
            )(),
        },
    )()
    input = torch.rand(B, 3, T, H, W).to(device)  # Example shape
    input_length = torch.tensor([T] * B).to(device)
    target = torch.randint(1, classes, (B, L)).to(device)  # Example shape
    target_length = torch.tensor([L] * B).to(device)

    # Calculate loss
    loss = loss(outputs, input, input_length, target, target_length)
    for k, v in loss._asdict().items():
        print(k, v)
