import numpy as np
import cv2
import os

from tqdm import tqdm


def calc(imgs_path, img_new_size, img_channel=3, img_type='.jpg', num_sample=0):
    """
    calculate mean and standard deviation
    :param imgs_path:
    :param img_new_size:
    :param img_channel:
    :param img_type:
    :param num_sample:
    :return:
    """

    img_h, img_w = img_new_size

    print('checking files ...')
    img_name_list = []
    for root, dirs, files in os.walk(imgs_path, topdown=False):
        for filename in files:
            if img_type in filename and filename[0] != '.':
                img_name_list.append(os.path.join(root, filename))

    if num_sample > 0:
        img_name_sample_list = np.random.choice(img_name_list, num_sample, replace=False)
    else:
        img_name_sample_list = img_name_list
        num_sample = len(img_name_list)

    print('loading files ...')
    img_list = []
    with tqdm(total=num_sample, unit='img') as pbar:
        for fn in img_name_sample_list:
            if img_channel == 1:
                img = cv2.resize(cv2.imread(fn, cv2.IMREAD_GRAYSCALE), (img_w, img_h))[:, :, np.newaxis, np.newaxis]
            else:
                img = cv2.resize(cv2.imread(fn, cv2.IMREAD_COLOR), (img_w, img_h))[:, :, :, np.newaxis]

            img_list.append(img)

            pbar.update()

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    print('calc ...')
    means, stdevs = [], []
    for i in range(img_channel):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB
    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

    return means, stdevs


if __name__ == '__main__':
    # root = '/media/xxx/Data/Datasets/CheXpert-v1.0-small/train/'
    # calc(imgs_path=root, img_new_size=(320, 320), img_channel=1, img_type='_frontal.jpg',num_sample=10000)
    # # normMean = [0.50627464], normStd = [0.28864813]

    root = '/media/xxx/Data/Datasets/CXR8/images/'
    calc(imgs_path=root, img_new_size=(320, 320), img_channel=1, img_type='.png',num_sample=5000)
    # # normMean = [0.5275818], normStd = [0.25026274]

