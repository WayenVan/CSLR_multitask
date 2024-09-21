import timm
import torch
from timm import models as tmodel
from timm.models.resnet import ResNet

model = timm.create_model("resnext50_32x4d", pretrained=True)
# delattr(model, "fc")

for name, _ in model.named_parameters():
    print(name)

data = torch.randn(1, 3, 224, 224)
output = model.forward_features(data)
