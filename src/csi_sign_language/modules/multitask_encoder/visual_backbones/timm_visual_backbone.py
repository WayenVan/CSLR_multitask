import timm
from timm.models.resnet import ResNet
from timm.models.mobilenetv3 import MobileNetV3
from timm.models.efficientnet import EfficientNet
from einops import rearrange
import torch
import re

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.modules.multitask_encoder.multitask_encoder import (
        VisualBackbone,
    )
else:
    from ..multitask_encoder import VisualBackbone


class TimmVisualBackbone(VisualBackbone):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        m = re.match
        assert m(r"resnet", model_name) or m(
            r"mobilenet", model_name
        ), "currently only support resnet and mobilenet"

        self.model = timm.create_model(model_name, pretrained=pretrained)
        # delattr(self.model, "fc")
        self.input_size = (224, 224)

    def get_input_size(self):
        return self.input_size

    def get_output_feats_size(self):
        reduction = self.model.feature_info[-1]["reduction"]
        return [(self.input_size[0] // reduction, self.input_size[1] // reduction)]

    def get_output_dims(self):
        return [self.model.feature_info[-1]["num_chs"]]

    def forward(self, x, t_length):
        """
        @parm x: (b, c, t, h, w)
        @parm t_length: (b, )
        """
        B, C, T, H, W = x.shape
        assert (H, W) == self.input_size

        T = int(x.size(2))
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.model.forward_features(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=T)
        return [x], t_length


if __name__ == "__main__":
    import torch

    # model = TimmVisualBackbone("efficientnet_b0")
    model = TimmVisualBackbone("resnet50")
    # print(model)
    print(model.get_input_size())
    print(model.get_output_feats_size())
    print(model.get_output_dims())
    print(model.model.feature_info)

    data = torch.randn(1, 3, 10, 224, 224)
    t_length = torch.tensor([10])
    output, _ = model(data, t_length)
    print(output[0].shape)
