import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import KFold

if __name__ == "__main__":
    label_df = pd.read_csv('../../dataset/isic_2019/ISIC_2019_Training_GroundTruth.csv')
    meta_df = pd.read_csv('../../dataset/isic_2019/ISIC_2019_Training_Metadata.csv')
    df1 = meta_df[['image', 'lesion_id']].dropna()
    df2 = meta_df.loc[~meta_df.image.isin(df1.image.values)]
    
    lesion_ids = np.unique(df1.lesion_id.values)
    kf = KFold(n_splits=5, random_state=8, shuffle=True)
    
    kfold_dict = {}
    for fold, (train_index, val_index) in enumerate(kf.split(lesion_ids)):
        val_lesion_ids = lesion_ids[val_index]
        val_df = meta_df.loc[meta_df.lesion_id.isin(val_lesion_ids)]
        val_images = np.unique(val_df.image.values)
        kfold_dict[fold] = list(val_images)

    image_ids = np.unique(df2.image.values)
    for fold, (train_index, val_index) in enumerate(kf.split(image_ids)):
        val_image_ids = image_ids[val_index]
        kfold_dict[fold].extend(list(val_image_ids))
    
    kfold_df = []
    for k, v in kfold_dict.items():
        val_df = label_df.loc[label_df.image.isin(v)]
        val_df = val_df.assign(fold=k)
        kfold_df.append(val_df)
    kfold_df = pd.concat(kfold_df, ignore_index=True)

    image_paths = []
    target = []
    for _, row in kfold_df.iterrows():
        image_path = '../../dataset/isic_2019/ISIC_2019_Training_Input/{}.jpg'.format(row['image'])
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
    # kfold_df.to_csv('../../dataset/isic_2019/train_kfold.csv', index=False)
