import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    classes_ext_2024 = ["MEL","BCC","SCC","NV"]
    diagnosis_2024_to_2019 = {
        "Basal cell carcinoma": "BCC",
        "Basal cell carcinoma, Infiltrating": "BCC",
        "Basal cell carcinoma, Nodular": "BCC",
        "Basal cell carcinoma, Superficial": "BCC",
        "Malignant melanocytic proliferations (Melanoma)": "MEL",
        "Melanoma Invasive": "MEL",
        "Melanoma Invasive, Associated with a nevus": "MEL",
        "Melanoma Invasive, Nodular": "MEL",
        "Melanoma Invasive, On chronically sun-exposed skin or lentigo maligna melanoma": "MEL",
        "Melanoma Invasive, Superficial spreading": "MEL",
        "Melanoma in situ": "MEL",
        "Melanoma in situ, Lentigo maligna type": "MEL",
        "Melanoma in situ, Superficial spreading": "MEL",
        "Melanoma in situ, associated with a nevus": "MEL",
        "Melanoma metastasis": "MEL",
        "Melanoma, NOS": "MEL",
        "Squamous cell carcinoma in situ": "SCC",
        "Squamous cell carcinoma in situ, Bowens disease": "SCC",
        "Squamous cell carcinoma, Invasive": "SCC",
        "Squamous cell carcinoma, Invasive, Keratoacanthoma-type": "SCC",
        "Squamous cell carcinoma, NOS": "SCC",
        "Blue nevus": "NV",
        "Blue nevus, Cellular": "NV",
        "Nevus": "NV",
        "Nevus, Atypical, Dysplastic, or Clark": "NV",
        "Nevus, Combined": "NV",
        "Nevus, Congenital": "NV",
        "Nevus, Deep penetrating": "NV",
        "Nevus, NOS, Compound": "NV",
        "Nevus, NOS, Dermal": "NV",
        "Nevus, NOS, Junctional": "NV",
        "Nevus, Of special anatomic site": "NV",
        "Nevus, Spitz": "NV",
    }

    df = pd.read_csv('../../dataset/isic_2024/train-metadata.csv')
    meta = []
    image_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ext_labels = [0]*len(classes_ext_2024)
        for diagnosis in row['iddx_full'].split("::"):
            if diagnosis in diagnosis_2024_to_2019.keys():
                class_name = diagnosis_2024_to_2019[diagnosis]
                if class_name in classes_ext_2024:
                    ext_labels[classes_ext_2024.index(class_name)] = 1
        
        image_path = '../../dataset/isic_2024/train-image/{}.jpg'.format(row['isic_id'])
        if os.path.isfile(image_path) == False:
            print(image_path, 'not found')
            continue
        image_paths.append(image_path)
        meta.append(ext_labels)
    df['image_path'] = np.array(image_paths)
    df[classes_ext_2024] = np.array(meta)
    df = df[['isic_id','image_path','target','patient_id'] + classes_ext_2024]

    x = []
    y = []
    for patient_id, grp in df.groupby('patient_id'):
        l = grp[['target'] + classes_ext_2024].values
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
    print(kfold_df.shape)
    print(kfold_df.head())
    # kfold_df.to_csv('../../dataset/isic_2024/train_kfold.csv', index=False)
