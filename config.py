import os

import numpy as np
import torch
from libauc.losses import AUCMLoss, AUCM_MultiLabel
from libauc.optimizers import PESG
from torch import optim
from torch.nn import BCELoss, MultiLabelSoftMarginLoss
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import transforms

from timm.models.resnet import resnet18, resnet50, resnet101
from timm.models.resnest import resnest26d, resnest50d, resnest101e
from timm.models.densenet import densenet121, densenet161, densenet201
from timm.models.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large

from timm.models.vision_transformer import vit_tiny_patch16_224, vit_base_patch16_224, vit_large_patch16_224
from timm.models.swin_transformer import swin_tiny_patch4_window7_224, swin_large_patch4_window7_224
from timm.models.deit import deit_tiny_patch16_224, deit3_large_patch16_224

from model.VAN import van_b0, van_b1, van_b2, van_b3
from model.VANGCN import MyVANGCN
from model.FusionVAN import FusionVAN
from model.SegFusionVAN import SegFusionVAN
from model.SegFusionVANGCN import SegFusionVANGCN
from model.GCNVAN import gcn_van_b0
from model.DenseGCN import gcn_densenet201
from model.SwinTGCN import gcn_swin_large_patch4_window7_224
from model.MultiScaleVAN import ms_van_b0

from data.CheXpert import CheXpert
from data.CXR8 import CXR8
from data.MIMIC import MIMIC

from utils.loss import FocalLoss
from utils.logger import get_logger

# modify datasets path by yourself
datasets = 'CheXpert'
# root = '/mnt/data/Datasets/CheXpert-v1.0-small'
root = '/opt/data/share/4021110075/datasets/CheXpert-v1.0-small'

# datasets = 'CXR8'
# root = '/mnt/data/Datasets/CXR8'
# root = '/opt/data/share/4021110075/datasets/CXR8'

# datasets = 'MIMIC'
# root = '/mnt/data/Datasets/MIMIC'
# root = '/opt/data/share/4021110075/datasets/MIMIC'


# do grad-cam image directory
do_heatmap_img_path = f'./do_heatmap_img'

# cnn: resnet, resnest, densenet, convnext
# no cnn: van, vit, swin, deit
model_name = 'SegFusionVAN'
is_cnn = True
pretrained = False

img_size = 224
in_channels = 3

batch_size = 32
lr = 0.001

num_workers = 4

# save out file
root_save_path = f'./outs-{datasets}-{model_name}'
model_save_path = f'{root_save_path}/saved_model'
model_file_path = f'{model_save_path}/best_model.pth'
curve_save_path = f'{root_save_path}/saved_curve'
os.makedirs(root_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(curve_save_path, exist_ok=True)

# log file
log_file_path = f'{root_save_path}/log.txt'
logger = get_logger(log_file_path)

# dataset train info
if datasets == 'CheXpert':
    epochs = 6
    class_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion',
        'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]
    class_weight = torch.Tensor([10, 10, 1, 1, 1])
elif datasets == 'CXR8':
    epochs = 5
    class_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
        'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
        # 'No Finding'
    ]
    class_weight = torch.Tensor([])
elif datasets == 'MIMIC':
    epochs = 3
    class_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
        'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pneumonia',
        'Pneumothorax', 'Pleural Other', 'Support Devices',
        'No Finding'
    ]
    class_weight = torch.Tensor([])
else:
    logger.info('Datasets name is wrong !')
    exit(0)

# data transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([img_size, img_size], antialias=True),
])

# data augment
aug = transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0)

# data loader
train_data = eval(datasets)(root, image_channels=in_channels, mode='train', transform=transform, augment=aug,
                            used_cols=class_labels)
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)

valid_data = eval(datasets)(root, image_channels=in_channels, mode='valid', transform=transform, used_cols=class_labels)
valid_data_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

test_data = eval(datasets)(root, image_channels=in_channels, mode='test', transform=transform, used_cols=class_labels)
test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model args
cnn_args = {
    'pretrained': pretrained,
    'in_chans': in_channels,
    'num_classes': len(class_labels)
}
# model
if is_cnn:
    model = eval(model_name)(**cnn_args).to(device)
else:
    model = eval(model_name)(img_size=img_size, **cnn_args).to(device)

# loss function
# loss_func = BCELoss(reduction='mean')
# loss_func = FocalLoss()
loss_func = FocalLoss(class_weight)

# optimizer
optimizer = optim.Adam(params=model.parameters(), lr=lr)

# lr scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
