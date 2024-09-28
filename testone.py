from PIL import Image

from torch import load
from torchvision import transforms

import config


def get_data(img_path):
    image = Image.open(img_path)
    image = image.convert('RGB')

    # data transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224], antialias=True),
    ])

    image = transform(image).unsqueeze(0)

    return image


def predict(image):
    config.logger.info(
        '============================================ start test ============================================')

    config.logger.info(f'loading model: {config.model_file_path}')
    config.model.load_state_dict(load(config.model_file_path))
    config.logger.info('load model success !')

    # test
    config.model.eval()
    with config.torch.no_grad():
        img = image.to(config.device)
        config.model(img).sigmoid().detach().cpu().numpy()


if __name__ == '__main__':
    predict(get_data('./do_heatmap_img/00000001_000.png'))
