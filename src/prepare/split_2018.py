import pandas as pd 
import numpy as np 
import os

if __name__ == "__main__":
    meta = []
    for rdir, _, files in os.walk('../../dataset/isic_2018/ISIC2018_Task1-2_Training_Input'):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext.lower() in ['.png', '.jpg', '.jpeg']:
                image_path = os.path.join(rdir, file)
                mask_path = '../../dataset/isic_2018/ISIC2018_Task1_Training_GroundTruth/{}_segmentation.png'.format(file_name)
                assert os.path.isfile(mask_path)
                meta.append([image_path, mask_path])
    for rdir, _, files in os.walk('../../dataset/isic_2018/ISIC2018_Task1-2_Validation_Input'):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext.lower() in ['.png', '.jpg', '.jpeg']:
                image_path = os.path.join(rdir, file)
                mask_path = '../../dataset/isic_2018/ISIC2018_Task1_Validation_GroundTruth/{}_segmentation.png'.format(file_name)
                assert os.path.isfile(mask_path)
                meta.append([image_path, mask_path])
    train_df = pd.DataFrame(data=np.array(meta), columns=['image_path', 'mask_path'])
    print(train_df.shape)
    print(train_df.head())
    # train_df.to_csv('../../dataset/isic_2018/train.csv', index=False)

    meta = []
    for rdir, _, files in os.walk('../../dataset/isic_2018/ISIC2018_Task1-2_Test_Input'):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            if file_ext.lower() in ['.png', '.jpg', '.jpeg']:
                image_path = os.path.join(rdir, file)
                mask_path = '../../dataset/isic_2018/ISIC2018_Task1_Test_GroundTruth/{}_segmentation.png'.format(file_name)
                assert os.path.isfile(mask_path)
                meta.append([image_path, mask_path])
    test_df = pd.DataFrame(data=np.array(meta), columns=['image_path', 'mask_path'])
    print(test_df.shape)
    print(test_df.head())
    # test_df.to_csv('../../dataset/isic_2018/test.csv', index=False)
