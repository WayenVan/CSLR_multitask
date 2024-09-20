import torch.nn as nn
from ..multitask_encoder import VisualBackbone
from mmpretrain.registry import MODELS
from mmpretrain.models.backbones.resnet import ResNet
from mmengine import build_model_from_cfg
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from einops import rearrange


class ResNetBackbone(VisualBackbone):
    def __init__(self, cfg, ckpt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = Config.fromfile(cfg)
        model = build_model_from_cfg(cfg.model, MODELS)
        load_checkpoint(model, ckpt)
        self.resnet = model.backbone
        del model

        self.input_size = (224, 224)
        self.output_feats_size = [(56, 56), (28, 28), (14, 14), (7, 7)]

    def forward(self, x, t_length):
        """
        @parm x: (b, c, t, h, w)
        @parm t_length: (b, )
        """
        B, C, T, H, W = x.shape
        assert (H, W) == self.input_size

        T = int(x.size(2))
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resnet_forward(x)
        x = tuple(rearrange(a, "(b t) c h w -> b c t h w", t=T) for a in x)
        return x, t_length

    def get_input_size(self):
        return self.input_size

    def get_output_feats_size(self):
        return self.output_feats_size

    def get_output_dims(self):
        _out_channels = self.resnet.base_channels * self.resnet.expansion
        ret = []
        for i in range(len(self.resnet.stage_blocks)):
            ret.append(_out_channels)
            _out_channels *= 2
        return ret

    def resnet_forward(self, x):
        if self.resnet.deep_stem:
            x = self.resnet.stem(x)
        else:
            x = self.resnet.conv1(x)
            x = self.resnet.norm1(x)
            x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.resnet.res_layers):
            res_layer = getattr(self.resnet, layer_name)
            x = res_layer(x)

            if i in self.resnet.out_indices:
                outs.append(x)
        return tuple(outs)
