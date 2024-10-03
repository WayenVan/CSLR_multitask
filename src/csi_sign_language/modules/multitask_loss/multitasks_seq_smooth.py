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
    from csi_sign_language.modules.multitask_loss.seq_smooth import t_mse

else:
    from .vitpose_wrapper import ViTPoseWrapper
    from .dwpose_wrapper import DWPoseWarpper
    from .seq_smooth import t_mse


MultiTaskDistillLossSmoothOut = namedtuple(
    "MultiTaskDistillLossSmoothOut",
    [
        "out",
        "ctc_loss",
        "dwpose_loss",
        "vit_loss",
        "smmooth_loss",
    ],
)


class MultiTaskDistillLossSmooth(nn.Module):
    def __init__(
        self,
        dwpose_cfg: str,
        dwpose_ckpt: str,
        vitpose_cfg: str,
        vitpose_ckpt: str,
        ctc_weight: float = 1.0,
        vitpose_weight: float = 1.0,
        dwpose_weight: float = 1.0,
        dwpose_dist_temperature: float = 8.0,
        smooth_weight: float = 1.0,
        smooth_tau: float = 3.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        init_default_scope("mmpose")

        # dwpose related
        if dwpose_weight > 0.0:
            self.dwpose = DWPoseWarpper(dwpose_cfg, dwpose_ckpt)
            self.dwpose_intput_size = (self.dwpose.input_H, self.dwpose.input_W)

        # vitpose related
        if vitpose_weight > 0.0:
            self.vitpose = ViTPoseWrapper(vitpose_cfg, vitpose_ckpt)

        # ctc related
        if ctc_weight > 0.0:
            self._loss_ctc = nn.CTCLoss(blank=0, reduction="none")

        self.smooth_weight = smooth_weight
        self.smooth_tau = smooth_tau

        # weights
        self.ctc_weight = ctc_weight
        self.dwpose_weight = dwpose_weight
        self.vitpose_weight = vitpose_weight
        self.dwpose_dist_temperature = dwpose_dist_temperature

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

        smmooth_loss = None
        if self.smooth_weight > 0.0:
            out = nn.functional.log_softmax(outputs.out, dim=-1).permute(1, 0, 2)
            smooth_loss = t_mse(out, self.smooth_tau, outputs.t_length)
            loss += self.smooth_weight * smooth_loss

        return MultiTaskDistillLossSmoothOut(
            out=loss,
            ctc_loss=ctc_loss,
            dwpose_loss=dwpose_loss,
            vit_loss=vit_loss,
            smmooth_loss=smmooth_loss,
        )

    def distill_loss_heatmap(self, heatmap: Tensor, input: Tensor):
        """
        @param heatmap the output heatmap from encoder of size [t b k h w]
        @pararm input: [b c t h w]
        """
        out_heatmap = rearrange(heatmap, "t b k h w -> (b t) k h w")
        input = rearrange(input, "b c t h w -> (b t) c h w")

        target_heatmap = self.vitpose(input)

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

        # assertion
        assert target_logits_x.shape == out_logits_x.shape
        assert target_logits_y.shape == out_logits_y.shape

        # x, y are all logits here
        # apply termperature
        target_logits_x, target_logits_y = (
            a / self.dwpose_dist_temperature for a in (target_logits_x, target_logits_y)
        )
        # log_softmax
        target_logits_x, target_logits_y = (
            nn.functional.log_softmax(a, dim=-1)
            for a in (target_logits_x, target_logits_y)
        )
        out_logits_x, out_logits_y = (
            nn.functional.log_softmax(a, dim=-1) for a in (out_logits_x, out_logits_y)
        )

        # distillation with temperature
        loss_pointwise_x = nn.functional.kl_div(
            out_logits_x, target_logits_x.detach(), log_target=True, reduction="none"
        )
        loss_pointwise_y = nn.functional.kl_div(
            out_logits_y, target_logits_y.detach(), log_target=True, reduction="none"
        )
        return loss_pointwise_x.sum() / (B * T) + loss_pointwise_y.sum() / (B * T)


if __name__ == "__main__":
    device = "cpu"
    loss = MultiTaskDistillLossSmooth(
        dwpose_cfg="/root/projects/sign_language_transformer/resources/dwpose-l/rtmpose-l_8xb64-270e_coco-ubody-wholebody-256x192.py",
        dwpose_ckpt="/root/projects/sign_language_transformer/resources/dwpose-l/dw-ll_ucoco.pth",
        vitpose_cfg="resources/ViTPose/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
        vitpose_ckpt="resources/ViTPose/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth",
        ctc_weight=1.0,
        dwpose_weight=1.0,
        vitpose_weight=1.0,
        dwpose_dist_temperature=8.0,
        smooth_weight=1.0,
        smooth_tau=1.0,
    ).to(device)
    classes = 100
    B = 2
    T = 100
    L = 5
    H = 224
    W = 224
    K_dwpose = 133
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
    print(f"Calculated loss: {loss.out.item()}")
    for k, v in loss._asdict().items():
        if v is not None:
            print(f"{k}: {v.item()}")
