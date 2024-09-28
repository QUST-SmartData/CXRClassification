import os.path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class CXR8(Dataset):
    def __init__(self, data_root_path, image_channels=3, mode='train',
                 transform=None, augment=None,
                 used_cols=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                            'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                            'Pneumonia', 'Pneumothorax'],
                 use_frontal=None, uncertain=None,
                 use_upsampling=False, upsampling_cols=['Hernia', 'Pneumonia']):
        super(CXR8, self).__init__()

        self.image_channels = image_channels
        self.mode = mode
        self.transform = transform
        self.augment = augment

        # check
        assert image_channels in (1, 3), 'image channel is wrong !'
        assert os.path.isdir(data_root_path), 'data path is wrong !'
        self.data_root_path = os.path.join(data_root_path, 'images')
        assert os.path.isdir(self.data_root_path), 'data path is wrong !'
        data_csv_path = os.path.join(data_root_path, 'Data_Entry_2017_v2020.csv')
        assert os.path.isfile(data_csv_path), 'cannot find Data_Entry_2017_v2020.csv !'
        train_val_list_path = os.path.join(data_root_path, 'train_val_list.txt')
        assert os.path.isfile(train_val_list_path), 'cannot find train_val_list.txt !'
        test_list_path = os.path.join(data_root_path, 'test_list.txt')
        assert os.path.isfile(test_list_path), 'cannot find test_list.txt !'
        assert mode in ('train', 'valid', 'test'), 'mode is wrong !'
        assert isinstance(upsampling_cols, list), 'Input should be list!'

        # prepare data
        train_csv_path = os.path.join(data_root_path, 'train.csv')
        valid_csv_path = os.path.join(data_root_path, 'valid.csv')
        test_csv_path = os.path.join(data_root_path, 'test.csv')

        while not (os.path.exists(train_csv_path) and os.path.exists(valid_csv_path) and os.path.exists(test_csv_path)):
            self.df = pd.read_csv(data_csv_path)
            self.df = self.df.sort_values('Image Index')
            self.df['Finding Labels'] = self.df['Finding Labels'].apply(lambda s: s.split('|'))
            mlb = MultiLabelBinarizer()
            self.df = self.df.join(pd.DataFrame(mlb.fit_transform(self.df.pop('Finding Labels')),
                                                columns=mlb.classes_, index=self.df.index))
            self.df = self.df[['Image Index'] + used_cols]

            train_valid_list = pd.read_csv(train_val_list_path, header=None).values.flatten()
            test_list = pd.read_csv(test_list_path, header=None).values.flatten()
            train_df = self.df[self.df['Image Index'].isin(train_valid_list[:80000])]
            valid_df = self.df[self.df['Image Index'].isin(train_valid_list[80000:])]
            test_df = self.df[self.df['Image Index'].isin(test_list)]

            # 确保验证集中的每个类别都存在正负样本
            if 0 not in np.sum(valid_df[1:])[-14:].tolist():
                train_df.to_csv(train_csv_path, index=False)
                valid_df.to_csv(valid_csv_path, index=False)
                test_df.to_csv(test_csv_path, index=False)

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

        # upsample selected cols
        if self.mode == 'train' and use_upsampling:
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        self.images = self.df.values[:, 0].tolist()
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
        # transforms.Normalize(mean=[0.53], std=[0.25], inplace=True)
    ])
    # data augment
    aug = transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0)
    root = '/mnt/data/Datasets/CXR8'
    cx = CXR8(root, mode='valid', transform=trans, augment=aug)
    cxl = DataLoader(dataset=cx, batch_size=1, shuffle=False)
    for i, b in enumerate(cxl):
        print(b[0], b[1])
        break
