import os
from pathlib import Path

import cv2
import torch.cuda
from torch import optim, save
from torch.utils.data import DataLoader

from data.ChestHeart import ChestHeart
from model.UNet import UNet
from utils.loss import DiceBCELoss, DiceLoss
from utils.onehot import onehot2mask


def unet_train():
    dataset_root = '/opt/data/share/4021110075/datasets/ChestHeart'
    model_save_path = r'./saved_unet_model'
    os.makedirs(model_save_path, exist_ok=True)

    lr = 1e-4
    epochs = 500
    batch_size = 16

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    train_datasets = ChestHeart(dataset_root=dataset_root, mode='train')
    valid_datasets = ChestHeart(dataset_root=dataset_root, mode='valid')
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_datasets, batch_size=1, shuffle=True, num_workers=4)

    # model
    model = UNet(in_channels=3, out_channels=3).to(device)
    # loss function
    loss_func = DiceBCELoss()
    acc_func = DiceLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('==== start train ====')
    # train
    train_loss = []
    val_acc = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.
        steps = 0
        for img, label in train_dataloader:
            img, label = img.to(device), label.to(device)
            pred = model(img).sigmoid()
            loss = loss_func(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().cpu()
            steps += 1
        train_avg_loss = total_loss / steps
        print(f'epoch:{epoch + 1}/{epochs} --> train loss:{train_avg_loss}')
        train_loss.append(train_avg_loss)

        # val
        model.eval()
        with torch.no_grad():
            total_acc = 0.
            steps = 0
            for img, label in valid_dataloader:
                img, label = img.to(device), label.to(device)
                pred = model(img).sigmoid()
                total_acc += 1 - acc_func(pred, label)
                steps += 1
            val_avg_acc = total_acc / steps
            print(f'epoch:{epoch + 1}/{epochs} --> val acc:{val_avg_acc}')
            val_acc.append(val_avg_acc)
            # save better model
            if val_avg_acc >= max(val_acc):
                os.makedirs(model_save_path, exist_ok=True)
                model_name = os.path.join(model_save_path, 'best_unet_model.pth')
                save(model.state_dict(), model_name)
                print(f'{model_name} , save best model successfully!')


def unet_test():
    test_datasets_path = r'/opt/data/share/4021110075/datasets/ChestHeart/test'
    model_path = r'./saved_unet_model/best_unet_model.pth'
    pred_save_path = r'./pred'
    os.makedirs(pred_save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    test_datasets = ChestHeart(dataset_root=test_datasets_path, mode='test')
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4)

    # model
    model = UNet(in_channels=3, out_channels=3).to(device)
    # load model parameters
    model.load_state_dict(torch.load(model_path))

    print('==== start predict ====')
    # predict
    model.eval()
    with torch.no_grad():
        for i, (img, _) in enumerate(test_dataloader):
            img = img.to(device)
            pred = model(img).squeeze().cpu().numpy()
            pred_image = onehot2mask(pred)
            pred_image_scale = pred_image
            pred_image_scale[pred_image_scale > 0] = 1
            # get original filename
            pred_img_name = os.path.join(pred_save_path, f'{Path(test_datasets.images[i]).stem}' + '.jpg')
            # save image
            cv2.imwrite(pred_img_name, pred_image_scale * 255)
            cv2.imwrite(os.path.join(pred_save_path, f'{Path(test_datasets.images[i]).stem}_merge' + '.jpg'),
                        pred_image * img.cpu().numpy().squeeze()[0]*255)
            print(f'save {pred_img_name} successfully!')


if __name__ == '__main__':
    unet_train()
    unet_test()
