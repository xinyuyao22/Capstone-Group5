from IPython.display import Image, clear_output
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tqdm.auto import tqdm
import shutil as sh

import matplotlib.pyplot as plt
os.system("sudo git clone https://github.com/AIVenture0/yolov5.git")
os.system("sudo pip install -r requirements.txt")

DIR_INPUT = os.getcwd() + "/"

train_data = pd.read_csv('train.csv')
train_data["area"]=[float(eval(train_data["bbox"][i])[2])* float(eval(train_data["bbox"][i])[3]) for i in range(len(train_data))]
df=train_data.drop(train_data[(train_data["area"]>200000) | (train_data["area"]<2000)].index)
df.reset_index(drop=True, inplace=True)

bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    df[column] = bboxs[:,i]
df.drop(columns=['bbox'], inplace=True)
df['x_center'] = df['x'] + df['w']/2
df['y_center'] = df['y'] + df['h']/2
df['classes'] = 0
from tqdm.auto import tqdm
import shutil as sh
df = df[['image_id','x', 'y', 'w', 'h','x_center','y_center','classes']]

source = 'train'
if True:
    for fold in [0]:
        val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]
        for name,mini in tqdm(df.groupby('image_id')):
            if name in val_index:
                path2save = 'val2017/'
            else:
                path2save = 'train2017/'
            if not os.path.exists('convertor/fold{}/labels/'.format(fold)+path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold)+path2save)
            with open('convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:
                row = mini[['classes','x_center','y_center','w','h']].astype(float).values
                row = row/1024
                row = row.astype(str)
                for j in range(len(row)):
                    text = ' '.join(row[j])
                    f.write(text)
                    f.write("\n")
            if not os.path.exists('convertor/fold{}/images/{}'.format(fold, path2save)):
                os.makedirs('convertor/fold{}/images/{}'.format(fold, path2save))
            sh.copy("../input/global-wheat-detection/{}/{}.jpg".format(source, name),
                    'convertor/fold{}/images/{}/{}.jpg'.format(fold, path2save, name))