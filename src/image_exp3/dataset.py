import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as albu
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

classes = ['target', 'MEL','BCC','SCC','NV']
classes_2019_2020 = ['target', 'MEL','BCC','SCC','NV','AK','BKL','DF','VASC','UNK']

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class ISIC_Dataset(Dataset):
    def __init__(self, df, image_size, mode, downsampling_type=None, neg_pos_ratio=None):
        self.image_size = image_size
        assert mode in  ['train', 'valid', 'test']
        self.mode = mode
        self.downsampling_type = downsampling_type
        if self.mode == 'train':
            assert self.downsampling_type in ['type1', 'type2']
        self.neg_pos_ratio=neg_pos_ratio

        if self.mode == 'train':
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.ImageCompression(quality_lower=80, quality_upper=100, p=0.25),
                albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
                albu.Flip(p=0.5),
                albu.RandomRotate90(p=0.5),
                albu.OneOf([
                    albu.MotionBlur(blur_limit=5),
                    albu.MedianBlur(blur_limit=5),
                    albu.GaussianBlur(blur_limit=5),
                    albu.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.5),
                albu.RandomBrightnessContrast(p=0.5),
                albu.CoarseDropout(num_holes_range=(1,1), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.25),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        if self.mode == 'train':
            self.ori_df = df.reset_index(drop=True)
            self.update_data()
            self.dlen = len(self.df)
        else:
            self.df = df.reset_index(drop=True)
            self.dlen = len(self.df)

    def __len__(self):
        return self.dlen

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        if self.mode == 'test':
            label = torch.FloatTensor([row['target']])
            return image, label, row['image_path']
        else:
            label = torch.FloatTensor(row[classes])
            return image, label
    
    def update_data(self):
        if self.downsampling_type == 'type1':
            df1 = self.ori_df.loc[self.ori_df['target'] == 1]
            df0 = self.ori_df.loc[self.ori_df['target'] == 0]
        else:
            self.ori_df['pos_sum'] = np.sum(self.ori_df[classes].values, 1)
            df1 = self.ori_df.loc[self.ori_df['pos_sum'] >= 1]
            df0 = self.ori_df.loc[self.ori_df['pos_sum'] == 0]
        df0 = df0.sample(n=min(self.neg_pos_ratio*len(df1), len(df0)))
        self.df = pd.concat([df1,df0], ignore_index=True)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
