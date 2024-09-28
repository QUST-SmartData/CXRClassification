import os

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from utils.onehot import mask2onehot


class ChestHeart(Dataset):
    def __init__(self, dataset_root='', mode='train', classes_label=[0, 1, 2], image_size=(224, 224)):
        super().__init__()

        self.classes_label = classes_label
        self.image_size = image_size
        self.transforms = transforms
        self.mode = mode

        assert mode in ('train', 'valid', 'test'), 'mode is wrong !'

        if self.mode == 'train':
            self.image_path = f'{dataset_root}/train/image'
            self.label_path = f'{dataset_root}/train/label'
        elif self.mode == 'valid':
            self.image_path = f'{dataset_root}/valid/image'
            self.label_path = f'{dataset_root}/valid/label'
        else:
            self.image_path = dataset_root
            self.label_path = None
        self.images = []
        self.labels = []
        images = os.listdir(self.image_path)
        images.sort()
        labels = os.listdir(self.label_path)
        labels.sort()
        for f in images:
            fn = os.path.join(self.image_path, f)
            if os.path.isfile(fn):
                self.images.append(fn)
        if self.mode != 'test':
            for f in labels:
                fn = os.path.join(self.label_path, f)
                if os.path.isfile(fn):
                    self.labels.append(fn)

        self.image_num = len(self.images)

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image = cv2.resize(cv2.imread(self.images[idx], cv2.IMREAD_COLOR), self.image_size)
        if self.mode != 'test':
            label = mask2onehot(cv2.resize(cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE), self.image_size),
                                self.classes_label)
        else:
            label = []
        image = transforms.ToTensor()(image)
        return image, label


if __name__ == '__main__':

    dataset_root = '/mnt/data/Datasets/ChestHeart'
    # 构建训练集
    train_dataset = ChestHeart(
        dataset_root=dataset_root,
        mode='train'
    )

    batch_size = 2
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for img, label in dataloader:
        print(img, label)
        break
