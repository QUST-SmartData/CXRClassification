# CXRClassification

This is the official repository for "Multi-label Chest X-ray Image Classification Based on Long-range Dependencies Capture and Label Relationships Learning". Please cite this work if you find this repository useful for your project.


Diagnosing chest diseases from X-ray images using convolutional neural networks (CNNs) is an active area of research. However, existing methods mostly focus on extracting feature information from local regions for prediction, while ignoring the larger-scale image contextual information. Moreover, anatomical segmentation knowledge and co-occurrence relationships among labels, which are important for classification, are not fully utilized. To address the above problems, we proposed a method to capture long-range dependent information in chest X-ray images using a CNN with large kernel convolution. Furthermore, it captures the detailed features of the interest region through anatomical segmentation and builds the potential relationships of different diseases using a graph convolutional network (GCN). Firstly, we pre-trained UNet from a dataset with organ-level annotations for segmenting anatomical regions of interest in the images. Secondly, we build a four-stage backbone network using the large kernel attention (LKA) mechanism and superimpose anatomically segmented regions on the feature maps of each stage to obtain different scales of feature maps for the regions of interest. Thirdly, we utilized a GCN to obtain a co-occurrence matrix representing the potential relationships between all disease labels in the training dataset. Finally, we get the disease diagnosis by combining the label co-occurrence matrix and the visual feature maps. We experimentally show that our proposed method achieves excellent AUC scores of 91.5%, 84.5%, and 82.5% on three publicly available CXR datasets–NIH, Stanford CheXpert, and MIMIC-CXR-JPG, respectively.


## Usage

### Requirements

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```


#### Training and Inference

Modify the configuration in `config.py`:

```python
# modify datasets path by yourself
datasets = 'CheXpert'
root = '/mnt/data/Datasets/CheXpert-v1.0-small'

# datasets = 'CXR8'
# root = '/mnt/data/Datasets/CXR8'

# datasets = 'MIMIC'
# root = '/mnt/data/Datasets/MIMIC'

# do grad-cam image directory
do_heatmap_img_path = f'./do_heatmap_img'

# cnn: resnet, resnest, densenet, convnext
# no cnn: van, vit, swin, deit
model_name = 'SegFusionVANGCN'
is_cnn = True
pretrained = False

img_size = 224
in_channels = 3

batch_size = 32
lr = 0.001

num_workers = 4
```


```bash
./run.sh
```


