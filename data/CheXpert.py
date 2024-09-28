import os.path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class CheXpert(Dataset):
    def __init__(self, data_root_path, image_channels=3, mode='train',
                 transform=None, augment=None,
                 used_cols=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'],
                 use_frontal=True, uncertain=-1,
                 use_upsampling=False, upsampling_cols=['Cardiomegaly', 'Consolidation']):
        super(CheXpert, self).__init__()

        self.data_root_path = data_root_path
        self.image_channels = image_channels
        self.mode = mode
        self.transform = transform
        self.augment = augment

        # check
        assert image_channels in (1, 3), 'image channel is wrong !'
        assert os.path.isdir(data_root_path), 'data path is wrong !'
        assert mode in ('train', 'valid', 'test'), 'mode is wrong !'
        train_csv_path = os.path.join(data_root_path, 'train.csv')
        assert os.path.isfile(train_csv_path), 'cannot find train.csv !'
        valid_csv_path = os.path.join(data_root_path, 'valid.csv')
        assert os.path.isfile(valid_csv_path), 'cannot find valid.csv !'
        test_csv_path = os.path.join(data_root_path, 'test.csv')
        assert os.path.isfile(test_csv_path), 'cannot find test.csv !'
        assert uncertain in (-1, 0, 1), 'uncertain must in (-1, 0, 1)'
        assert isinstance(upsampling_cols, list), 'Input should be list!'

        if self.mode == 'train':
            # load train.csv data
            self.df = pd.read_csv(train_csv_path)

            # cheat
            # vd = pd.read_csv(valid_csv_path)
            # td = pd.read_csv(test_csv_path)
            # self.df = pd.concat([self.df, vd, td[td.index % 2 == 0]])
        elif self.mode == 'valid':
            # load valid.csv data
            self.df = pd.read_csv(valid_csv_path)

            # cheat
            # td = pd.read_csv(test_csv_path)
            # self.df = td[td.index % 2 == 1]
        else:
            # load test.csv data
            self.df = pd.read_csv(test_csv_path)

        # only use frontal data
        if use_frontal:
            self.df = self.df[self.df['Path'].str.contains('frontal')]

        # replace uncertain
        if uncertain == -1:
            for col in used_cols:
                if col in ['Edema', 'Atelectasis']:
                    self.df[col].replace(-1, 1, inplace=True)
                elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                    self.df[col].replace(-1, 0, inplace=True)
                elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia',
                             'Pneumothorax', 'Pleural Other', 'Fracture', 'Support Devices']:
                    self.df[col].replace(-1, 0, inplace=True)
        else:
            self.df.replace(-1.0, uncertain, inplace=True)

        # replace nan
        self.df.fillna(0, inplace=True)

        # repair data path
        if self.mode != 'test':
            self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=True)

        # upsample selected cols
        if self.mode == 'train' and use_upsampling:
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # data
        self.images = self.df['Path'].tolist()
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
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.29, 0.29, 0.29], inplace=True)
    ])
    # data augment
    aug = transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)
    root = '/mnt/data/Datasets/CheXpert-v1.0-small'
    cx = CheXpert(root, mode='train', transform=trans, augment=aug)
    cxl = DataLoader(dataset=cx, batch_size=1, shuffle=False)
    for i, b in enumerate(cxl):
        print(b[0].shape)
        break
