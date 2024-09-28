import os

import matplotlib.pyplot as plt
import cv2
import pandas as pd

root = '/root/data/2021/zhaoxiangxin/datasets/CXR8/'
image_dir = 'images'
box_csv_file = 'BBox_List_2017.csv'

images_list = pd.read_csv(os.path.join(root, box_csv_file), header=0).values.tolist()

for img_info in images_list:
    img_name = os.path.join(root, image_dir, img_info[0])
    if os.path.exists(img_name):
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        image = cv2.rectangle(image, (int(img_info[2]), int(img_info[3])),
                              (int(img_info[2] + img_info[4]), int(img_info[3] + img_info[5])), (0, 0, 255), 2)
        image = cv2.putText(image, img_info[1], (350, 950), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        # plt.imshow(image)
        # plt.show()
        save_path = f'{root}/mark/'
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path,img_info[0]), image)
