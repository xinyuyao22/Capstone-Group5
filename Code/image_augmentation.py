import os
# os.system("sudo pip install timm")
# os.system("sudo pip install effdet")
# os.system("sudo pip install omegaconf")
# os.system("sudo pip install pycocotools")

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

import pandas as pd
import re
import ast
import numpy as np
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import random

# DIR_INPUT = os.getcwd() + "/"
TRAIN_ROOT_PATH = 'train'
if 'augmented_train.csv' not in os.listdir():
    train_data = pd.read_csv('train.csv')
    train_data["area"] = [float(eval(train_data["bbox"][i])[2]) * float(eval(train_data["bbox"][i])[3]) for i in
                          range(len(train_data))]
    marking = train_data.drop(train_data[(train_data["area"] > 200000) | (train_data["area"] < 2000)].index)
    marking.reset_index(drop=True, inplace=True)

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:, i]
    marking.drop(columns=['bbox'], inplace=True)

    # duplicate images from inrae_1 and ethz_1
    inrae_ethz = marking[marking['source'].isin(['inrae_1', 'ethz_1'])]
    inrae_ethz_image_ids = inrae_ethz.image_id.unique()

    for image_id in inrae_ethz_image_ids:
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg')
        img_flip_ud = cv2.flip(image, 0)
        pts1 = np.float32([[random.randint(0, 100), random.randint(0, 100)],
                           [random.randint(900, 1023), random.randint(0, 100)],
                           [random.randint(0, 100), random.randint(900, 1023)],
                           [random.randint(900, 1023), random.randint(900, 1023)]])
        pts2 = np.float32([[0, 0], [1023, 0], [0, 1023], [1023, 1023]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img_flip_ud, M, (1024, 1024))
        cv2.imwrite(f'{TRAIN_ROOT_PATH}/{image_id}_flip.jpg', dst)

    augmented = []

    for index, row in marking.iterrows():
        if row['source'] in ['inrae_1', 'ethz_1']:
            result = {
                'image_id': row['image_id'] + '_flip',
                'width': 1024,
                'height': 1024,
                'source': row['source'],
                'x': row['x'],
                'y': 1024 - row['y'] - row['h'],
                'w': row['w'],
                'h': row['h']
            }
            augmented.append(result)

    augmented = pd.DataFrame(augmented, columns=['image_id', 'width', 'height', 'source', 'x', 'y', 'w', 'h'])
    frames = [marking, augmented]

    train_df = pd.concat(frames)
    train_df.to_csv('augmented_train.csv')

'''
def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(1024, 1024, erosion_rate=0.0, interpolation=1, p=1.0),
            A.OneOf([
                A.RandomGamma(p=0.3),
                A.RGBShift(p=0.4),
                A.RandomBrightnessContrast(p=0.3)
            ],p=0.5),
            A.Blur(p=0.5),
            A.RandomRotate90(p=0.5),
            A.CoarseDropout(p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']))
'''