import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import torch
import segmentation_models_pytorch as smp
import cv2 
from util import seed_everything
from models import ISIC_2018_Seg_Model
import shutil
import albumentations as albu
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

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

    transform1 = albu.Compose([
        albu.Resize(256, 256),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    transform2 = albu.Compose([
        albu.Resize(512, 512),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


    test_df = pd.read_csv('../../dataset/isic_2018/test.csv')
    
    ious = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        file = row['image_path'].split('/')[-1]
        file_name, file_ext = os.path.splitext(file)
        image = cv2.imread(row['image_path'])
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)

        height, width = image.shape[0:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed1 = transform1(image=image, mask=mask)
        image1 = transformed1['image']
        mask1 = transformed1['mask'].float().unsqueeze(0)
        mask1 /= 255.0
        mask1 = mask1.long().squeeze(1).cuda()
        image1 = image1.unsqueeze(0).cuda()

        transformed2 = transform2(image=image)
        image2 = transformed2['image']
        image2 = image2.unsqueeze(0).cuda()

        with torch.cuda.amp.autocast(), torch.no_grad():
            pred1 = model1(image1)
            # pred1 = torch.sigmoid(pred1)

            pred2 = model2(image2)
            # pred2 = torch.sigmoid(pred2)
            pred2 = F.interpolate(pred2, (256, 256))

            pred3 = model3(image2)
            # pred3 = torch.sigmoid(pred3)
            pred3 = F.interpolate(pred3, (256, 256))

            pred = (0.2*pred1 + 0.2*pred2 + 0.6*pred3)
            pred = torch.sigmoid(pred)

            pred = (pred >= 0.5).long().squeeze(1)
            tp, fp, fn, tn = smp.metrics.get_stats(pred, mask1, mode="binary")
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            ious.append(iou.item())
    print(np.mean(ious))

        # mask2 = F.interpolate(mask1, (512, 512))
        # print(image1.size(), mask1.size())
        # print(mask2.size())

        # image = image.unsqueeze(0).cuda()
        
    #     with torch.cuda.amp.autocast(), torch.no_grad():
    #         pred = model(image)
    #         pred = torch.sigmoid(pred)
    #         pred = (pred >= 0.5).long().squeeze(1)
    #         tp, fp, fn, tn = smp.metrics.get_stats(pred, mask, mode="binary")
    #         iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    #         ious.append(iou.item())
    
        # pred = pred.data.cpu().numpy().astype(np.float32)
        # pred = cv2.resize(pred, (width, height))
        # pred = np.where(pred > 0.5, 1, 0)


        # pred = (pred * 255).astype(np.uint8)
        
        # pred = np.concatenate((mask, pred), 1)
        # shutil.copy(row['image_path'], 'vis/isic2018/{}'.format(file))
        # cv2.imwrite('vis/isic2018/{}_mask.jpg'.format(file_name), pred)