import timm
from timm.models.resnet import ResNet
from einops import rearrange

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.modules.multitask_encoder.multitask_encoder import (
        VisualBackbone,
    )
else:
    from ..multitask_encoder import VisualBackbone


class TimmResNetBackbone(VisualBackbone):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        delattr(self.model, "fc")

        self.input_size = (224, 224)
        self.output_feats_size = [(56, 56), (28, 28), (14, 14), (7, 7)]

    def get_input_size(self):
        return self.input_size

    def get_output_feats_size(self):
        return self.output_feats_size

    def get_output_dims(self):
        return [self.model.num_features]

    def forward(self, x, t_length):
        """
        @parm x: (b, c, t, h, w)
        @parm t_length: (b, )
        """
        B, C, T, H, W = x.shape
        assert (H, W) == self.input_size

        T = int(x.size(2))
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.model(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=T)
        return [x], t_length


if __name__ == "__main__":
    model = TimmResNetBackbone("resnet18")
    # print(model)
    print(model.get_input_size())
    print(model.get_output_feats_size())
    print(model.get_output_dims())
