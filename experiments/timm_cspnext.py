import timm
import torch
from timm import models as tmodel
from timm.models.resnet import ResNet


model = timm.create_model("mobilenetv3_small_050", pretrained=True)
# delattr(model, "fc")
print(model.__class__.__module__)
print(model.__class__.__name__)
print(model.head_hidden_size)
print(model.feature_info)
print(model.feature_info[-1]["num_chs"])

# for name, _ in model.named_parameters():
#     print(name)

data = torch.randn(1, 3, 224, 224)
output = model.forward_features(data)
print(output.shape)
