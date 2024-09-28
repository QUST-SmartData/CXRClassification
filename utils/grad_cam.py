import os

import cv2
import numpy as np

from torch import load

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import config


def do_grad_cam():
    config.logger.info(
        '============================================= grad_cam =============================================')

    config.logger.info(f'loading do heatmap images: {config.do_heatmap_img_path}')
    img_names = []
    for fn in os.listdir(config.do_heatmap_img_path):
        if os.path.splitext(fn)[-1] in ['.jpg', '.png']:
            img_names.append(os.path.join(config.do_heatmap_img_path, fn))
    config.logger.info('load do heatmap images success !')

    config.logger.info(f'loading model: {config.model_file_path}')
    config.model.load_state_dict(load(config.model_file_path))
    config.logger.info('load model success !')

    for i in range(len(img_names)):
        save_heatmap(img_names[i])
        config.logger.info(f'{img_names[i]} heatmap saved successful!')

    config.logger.info("grad_cam finished!\n")


def save_heatmap(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, [config.img_size, config.img_size])
    rgb_image = np.float32(rgb_image) / 255

    input_image = config.transform(image)
    # input_tensor = # Create an input tensor image for your model.
    # Note: input_tensor can be a batch tensor with several images!
    input_tensor = input_image.unsqueeze(0)

    # attention map layer
    if 'van' in config.model_name or 'VAN' in config.model_name:
        target_layers = [config.model.net.block1[-1]]  # van
    elif 'res' in config.model_name:
        target_layers = [config.model.layer4[-1]]  # resnet
    elif 'dense' in config.model_name:
        target_layers = [config.model.features[-1]]  # densenet
    else:
        target_layers = None

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=config.model, target_layers=target_layers, use_cuda=config.torch.cuda.is_available())

    for i in range(len(config.class_labels)):
        targets = [ClassifierOutputTarget(i)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        # show
        # plt.imshow(visualization)
        # plt.show()

        # save images
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        save_path = f'{config.root_save_path}/heatmap/{os.path.basename(image_file).split(".")[0]}'
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, f'{config.class_labels[i]}.jpg'), visualization)


if __name__ == '__main__':
    do_grad_cam()
