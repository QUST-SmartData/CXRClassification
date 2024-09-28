import cv2
import sys

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from pandas import DataFrame
from torch import load

from config import *


def load_and_resize_img(path):
    """
    Load and convert the full resolution images on CodaLab to
    low resolution used in the small dataset.
    """
    """
        Load and convert the full resolution images on CodaLab to
        low resolution used in the small dataset.
        """
    img = cv2.imread(path, 0)

    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    if max_ind == 1:
        # width fixed at 320
        wpercent = (320 / float(size[0]))
        hsize = int((size[1] * wpercent))
        new_size = (hsize, 320)

    else:
        # height fixed at 320
        hpercent = (320 / float(size[1]))
        wsize = int((size[0] * hpercent))
        new_size = (320, wsize)

    resized_img = cv2.resize(img, new_size)

    cv2.imwrite(path, resized_img)


def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]

    test_df = pd.read_csv(infile)
    all_test_image_list = test_df.values[:, 0].tolist()
    frontal_test_image_list = [x for x in all_test_image_list if 'frontal' in x]
    pred_test_df = DataFrame({'Path': frontal_test_image_list})
    Parallel(n_jobs=-1)(delayed(load_and_resize_img)(path) for path in test_df.Path.values)

    model_file_path = os.path.join(model_save_path, 'best_model.pth')
    model.load_state_dict(load(model_file_path)['model'])
    print('load model success !')

    # test
    model.eval()
    preds = []
    with torch.no_grad():
        for idx, image_path in enumerate(frontal_test_image_list):
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = transform(img).unsqueeze(0).to(device)
            pred = model(img).detach().cpu().numpy()
            preds.append(pred[0].tolist())
    preds = np.asarray(preds)

    # CheXpert-v1.0/{valid,test}/<PATIENT>/<STUDY>
    for i, c in enumerate(class_labels):
        pred_test_df[c] = preds[:, i]

    pred_test_df.Path.str.split('/')

    def get_study(path):
        return path[0:path.rfind('/')]

    pred_test_df['Study'] = pred_test_df.Path.apply(get_study)

    study_df = pred_test_df.drop('Path', axis=1).groupby('Study').max().reset_index()

    study_df.to_csv(outfile, index=False)


# python predict.py <input-data-csv-filename> <prediction-csv-filename>
#
# input-data-csv:
# Path
# CheXpert-v1.0/valid/patient00000/study1/view1_frontal.jpg
# CheXpert-v1.0/valid/patient00000/study1/view2_lateral.jpg
# CheXpert-v1.0/valid/patient00000/study2/view1_frontal.jpg
#
# prediction-csv:
# Study,Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion
# CheXpert-v1.0/valid/patient00000/study1,0.16410329937934875,0.34118956327438354,0.10102979093790054,0.12217354774475098,0.45898008346557617
# CheXpert-v1.0/valid/patient00000/study2,0.164107084274292,0.3411961495876312,0.10103944689035416,0.12216444313526154,0.4590160846710205
#
if __name__ == '__main__':
    main()
