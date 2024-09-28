import torch.onnx
from timm.models.resnet import resnet18, resnet50, resnet101
from timm.models.densenet import densenet121

from model.VAN import van_b0

cnn_args = {
    'pretrained': False,
    'in_chans': 3,
    'num_classes': 5
}
# myNet = densenet121(**cnn_args)
myNet = resnet50(in_chans=3, num_classes=5)
print(myNet)
# x = torch.randn(2, 3, 224, 224)  # 随机生成一个输入
# modelData = "./resnet50.pth"  # 定义模型数据保存的路径
# torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
