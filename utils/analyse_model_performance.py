import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count, flop_count_table

from timm.models.resnet import resnet18, resnet50, resnet101
from timm.models.resnest import resnest14d, resnest26d, resnest50d
from timm.models.densenet import densenet121, densenet161, densenet201
from timm.models.convnext import convnext_nano, convnext_tiny, convnext_small
from timm.models.vision_transformer import vit_tiny_patch16_224
from timm.models.deit import deit_tiny_patch16_224

from model.VAN import van_b0, van_b1
from model.my import MyNet

# 创建网络
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn_args = {
    'pretrained': False,
    'in_chans': 3,
    'num_classes': 5
}

vit_args = {
    'pretrained': False,
    'img_size': 224,
    'in_chans': 3,
    'num_classes': 5
}

# ----------------- cnn -----------------
# model = resnet18(**cnn_args).to(device)
# model = resnet50(**cnn_args).to(device)
# model = resnet101(**cnn_args).to(device)
# model = resnest50d(**cnn_args).to(device)
# model = densenet121(**cnn_args).to(device)
# model = densenet161(**cnn_args).to(device)
# model = densenet201(**cnn_args).to(device)
# model = convnext_tiny(**cnn_args).to(device)

# ----------------- vit -----------------
# model = vit_tiny_patch16_224(**vit_args).to(device)
# model = swin_tiny_patch4_window10_224(**vit_args).to(device)
# model = deit_tiny_patch16_224(**vit_args).to(device)
model = MyNet(**vit_args).to(device)
# model = van_b0(**vit_args).to(device)
# model = van_b1(**vit_args).to(device)
# model = van_b2(**vit_args).to(device)

# 创建输入网络的tensor
tensor = (torch.rand(1, 3, 224, 224).to(device),)

# 分析FLOPs
flops = FlopCountAnalysis(model, tensor)
print("FLOPs: ", flops.total())
print(flop_count_table(flops))

# 分析parameters
print(parameter_count_table(model))
