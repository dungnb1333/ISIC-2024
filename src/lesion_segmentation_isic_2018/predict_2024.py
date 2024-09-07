import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torch
import cv2 
from util import seed_everything
from torch.utils.data import DataLoader
from dataset import ISIC_2018_Seg_Test_Dataset
from models import ISIC_2018_Seg_Model
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--workers", default=8, type=int)
args = parser.parse_args()
print(args)

seed_everything(seed=123)

if __name__ == "__main__":
    model1 = ISIC_2018_Seg_Model(
        encoder_name='timm-efficientnet-b7',
        encoder_weights=None,
        decoder_name='UnetPlusPlus'
    )
    model1.cuda()
    model1.load_state_dict(torch.load('checkpoints/timm-efficientnet-b7_UnetPlusPlus_256.pt'))
    model1.eval()

    model2 = ISIC_2018_Seg_Model(
        encoder_name='timm-efficientnet-b5',
        encoder_weights=None,
        decoder_name='UnetPlusPlus'
    )
    model2.cuda()
    model2.load_state_dict(torch.load('checkpoints/timm-efficientnet-b5_UnetPlusPlus_512.pt'))
    model2.eval()

    model3 = ISIC_2018_Seg_Model(
        encoder_name='mit_b5',
        encoder_weights=None,
        decoder_name='FPN'
    )
    model3.cuda()
    model3.load_state_dict(torch.load('checkpoints/mit_b5_FPN_512.pt'))
    model3.eval()

    test_df = pd.read_csv('../../dataset/isic_2024/train_kfold.csv')
    
    test_dataset = ISIC_2018_Seg_Test_Dataset(df=test_df)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    
    mask_dir = '../../dataset/isic_2024/mask_256'
    os.makedirs(mask_dir, exist_ok=True)
    data = []

    for batch_image_256, batch_image_512, batch_height, batch_width, batch_path in tqdm(test_loader):
        batch_image_256 = batch_image_256.cuda()
        batch_image_512 = batch_image_512.cuda()
        batch_height = batch_height.data.cpu().numpy().astype(int)
        batch_width = batch_width.data.cpu().numpy().astype(int)
        
        with torch.cuda.amp.autocast(), torch.no_grad():
            batch_mask1 = model1(batch_image_256)
            batch_mask2 = model2(batch_image_512)
            batch_mask2 = F.interpolate(batch_mask2, (256, 256))
            batch_mask3 = model3(batch_image_512)
            batch_mask3 = F.interpolate(batch_mask3, (256, 256))

            batch_mask = (0.2*batch_mask1 + 0.2*batch_mask2 + 0.6*batch_mask3)
            batch_mask = torch.sigmoid(batch_mask)
            
        batch_mask = batch_mask.data.cpu().numpy().astype(np.float32)
        for image_path, height, width, mask in zip(batch_path, batch_height, batch_width, batch_mask):
            file = image_path.split('/')[-1]
            file_name, file_ext = os.path.splitext(file)

            mask = mask.squeeze()
            mask = np.where(mask > 0.5, 1, 0)
            mask = (mask * 255).astype(np.uint8)

            mask_path = '{}/{}.png'.format(mask_dir, file_name)
            cv2.imwrite(mask_path, mask)

            data.append([image_path, mask_path, width, height])
    new_df = pd.DataFrame(data=np.array(data), columns=['image_path', 'mask_path', 'width', 'height'])
    new_df[['width', 'height']] = new_df[['width', 'height']].astype(int)
    new_df.to_csv('../../dataset/isic_2024/isic_2024_mask.csv', index=False)
