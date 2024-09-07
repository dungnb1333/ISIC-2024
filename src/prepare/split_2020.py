import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import KFold

if __name__ == "__main__":
    classes = ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK']

    dup_df = pd.read_csv('../../dataset/isic_2020/ISIC_2020_Training_Duplicates.csv')
    df = pd.read_csv('../../dataset/isic_2020/train.csv')
    df = df.loc[~df.image_name.isin(dup_df.image_name_2.values)].reset_index(drop=True)

    diagnosis_2020_to_class_2019 = {
        'atypical melanocytic proliferation': 'UNK',
        'cafe-au-lait macule': 'UNK',
        'lentigo NOS': 'BKL',
        'lichenoid keratosis': 'BKL', 
        'melanoma': 'MEL', 
        'nevus': 'NV', 
        'seborrheic keratosis': 'BKL',
        'solar lentigo': 'BKL', 
        'unknown': 'UNK',
    }

    labels = []
    for _, row in df.iterrows():
        class_name_2019 = diagnosis_2020_to_class_2019[row['diagnosis']]
        l = [0]*len(classes)
        l[classes.index(class_name_2019)] = 1
        labels.append(l)
    df[classes] = np.array(labels, dtype=float)

    patient_ids = np.unique(df.patient_id.values)
    kf = KFold(n_splits=5, random_state=8, shuffle=True)

    kfold_df = []
    for fold, (train_index, val_index) in enumerate(kf.split(patient_ids)):
        val_patient_ids = patient_ids[val_index]
        val_df = df.loc[df.patient_id.isin(val_patient_ids)]
        val_df = val_df.assign(fold=fold)
        kfold_df.append(val_df)
    kfold_df = pd.concat(kfold_df, ignore_index=True)
    image_paths = []
    target = []
    for _, row in kfold_df.iterrows():
        image_path = '../../dataset/isic_2020/train/{}.jpg'.format(row['image_name'])
        if os.path.isfile(image_path) == False:
            print(image_path, 'not found')
            continue
        image_paths.append(image_path)
        if row['MEL'] == 1 or row['BCC'] == 1 or row['SCC'] == 1:
            target.append(1)
        else:
            target.append(0)
    kfold_df['image_path'] = np.array(image_paths)
    kfold_df['target'] = np.array(target, dtype=int)
    kfold_df = kfold_df[['image_path','target','MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK','fold']]
    print(kfold_df.shape)
    print(kfold_df.head())
    # kfold_df.to_csv('../../dataset/isic_2020/train_kfold.csv', index=False)
