import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MIMIC(Dataset):
    def __init__(self, data_root_path, image_channels=3, mode='train',
                 transform=None, augment=None,
                 used_cols=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                            'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax',
                            'Pleural Other', 'Support Devices', 'No Finding'],
                 use_frontal=True, uncertain=1,
                 use_upsampling=False, upsampling_cols=['Hernia', 'Pneumonia']):
        super(MIMIC, self).__init__()

        self.image_channels = image_channels
        self.mode = mode
        self.transform = transform
        self.augment = augment

        # check
        assert os.path.isdir(data_root_path), 'data path is wrong !'
        self.data_root_path = os.path.join(data_root_path, 'images')
        assert os.path.isdir(self.data_root_path), 'data path is wrong !'
        train_csv_path = os.path.join(data_root_path, 'train.csv')
        assert os.path.isfile(train_csv_path), 'cannot find train.csv !'
        valid_csv_path = os.path.join(data_root_path, 'valid.csv')
        assert os.path.isfile(valid_csv_path), 'cannot find valid.csv !'
        test_csv_path = os.path.join(data_root_path, 'test.csv')
        assert os.path.isfile(test_csv_path), 'cannot find test.csv !'
        assert mode in ('train', 'valid', 'test'), 'mode is wrong !'
        assert image_channels in (1, 3), 'image channel is wrong !'
        assert uncertain in (-1, 0, 1), 'uncertain must in (-1, 0, 1)'
        assert isinstance(upsampling_cols, list), 'Input should be list!'

        if self.mode == 'train':
            # load train.csv data
            self.df = pd.read_csv(train_csv_path)

            # cheat
            # td = pd.read_csv(test_csv_path)
            # self.df = pd.concat([self.df, td[td.index % 2 == 0]])
        elif self.mode == 'valid':
            # load valid.csv data
            self.df = pd.read_csv(valid_csv_path)

            # cheat
            # td = pd.read_csv(test_csv_path)
            # self.df = pd.concat([self.df, td[td.index % 2 == 1]])
        else:
            # load test.csv data
            self.df = pd.read_csv(test_csv_path)

        # only use frontal data
        if use_frontal:
            self.df = self.df[self.df['ViewPosition'].isin(['AP', 'PA'])]

        # replace uncertain
        self.df.replace(-1.0, uncertain, inplace=True)

        # replace nan
        self.df.fillna(0, inplace=True)

        # upsample selected cols
        if self.mode == 'train' and use_upsampling:
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        self.images = list(map(lambda x: str(x) + '.png', self.df.values[:, 0].tolist()))
        self.labels = self.df[used_cols].values.tolist()

        # image data number
        self.image_num = len(self.images)

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_root_path, self.images[idx]))
        if self.image_channels == 3:
            # read image as rgb image
            image = image.convert('RGB')
        else:
            # read image as gray image
            image = image.convert('L')

        label = torch.Tensor(np.array(self.labels[idx]).reshape(-1).astype(np.float32))

        # 先增强
        if self.mode == 'train' and self.augment:
            image = self.augment(image)

        # 再归一化
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


if __name__ == "__main__":
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.29, 0.29, 0.29], inplace=True)
    ])
    # data augment
    aug = transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)
    root = '/mnt/data/Datasets/MIMIC'
    cx = MIMIC(root, mode='train', transform=trans, augment=aug)
    cx1 = MIMIC(root, mode='valid', transform=trans)
    cx2 = MIMIC(root, mode='test', transform=trans)
    cxl = DataLoader(dataset=cx, batch_size=1, shuffle=False)
    for i, b in enumerate(cxl):
        img = b[0]
        print(img)
        break
