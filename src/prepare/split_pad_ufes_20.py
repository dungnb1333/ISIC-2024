import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold

if __name__ == "__main__":
    df = pd.read_csv('../../dataset/PAD-UFES-20/metadata.csv')
    classes = ['MEL','BCC','SCC','NEV','ACK','SEK']

    diagnostic = []
    target = []
    image_paths = []
    for _, row in df.iterrows():
        l = [0]*len(classes)
        l[classes.index(row['diagnostic'])] = 1
        diagnostic.append(l)
        if row['diagnostic'] in ['MEL','BCC','SCC']:
            target.append(1)
        else:
            target.append(0)
        image_paths.append('../../dataset/PAD-UFES-20/images/{}'.format(row['img_id']))
    df[classes] = np.array(diagnostic, dtype=int)
    df['target'] = np.array(target, dtype=int)
    df['image_path'] = np.array(image_paths, dtype=str)
    
    x = []
    y = []
    for patient_id, grp in df.groupby('patient_id'):
        l = grp[['target'] + classes].values
        l = np.max(l, 0)
        x.append(patient_id)
        y.append(l)
    x = np.array(x)
    y = np.array(y)

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    kfold_df = []
    for fold, (train_index, val_index) in enumerate(mskf.split(x, y)):
        val_patient_ids = x[val_index]
        val_df = df.loc[df.patient_id.isin(val_patient_ids)]
        val_df = val_df.assign(fold=fold)
        kfold_df.append(val_df)
    kfold_df = pd.concat(kfold_df, ignore_index=True)
    kfold_df['AK'] = kfold_df['ACK']
    kfold_df['NV'] = kfold_df['NEV']
    kfold_df = kfold_df[['patient_id', 'img_id', 'image_path', 'target', 'MEL','BCC','SCC','NV','AK','SEK','fold']]
    print(kfold_df.shape)
    print(kfold_df.head())
    # kfold_df.to_csv('../../dataset/PAD-UFES-20/train_kfold.csv', index=False)
