import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml
import random

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchmetrics.functional.segmentation import mean_iou

from dataset import ISIC_2018_Seg_Dataset
from util import seed_everything
from models import ISIC_2018_Seg_Model

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default=None, type=str)
parser.add_argument("--frac", default=1.0, type=float)
args = parser.parse_args()
print(args)

seed_everything(seed=123)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    train_df = pd.read_csv('../../dataset/isic_2018/train.csv')
    test_df = pd.read_csv('../../dataset/isic_2018/test.csv')

    if args.frac != 1:
        print('Quick training')
        train_df = train_df.sample(frac=args.frac).reset_index(drop=True)
        test_df = test_df.sample(frac=args.frac).reset_index(drop=True)

    train_dataset = ISIC_2018_Seg_Dataset(df=train_df, image_size=cfg['image_size'], mode='train')
    valid_dataset = ISIC_2018_Seg_Dataset(df=test_df, image_size=cfg['image_size'], mode='valid')

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'])
    
    print('TRAIN: {} | VALID: {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))

    model = ISIC_2018_Seg_Model(
        encoder_name=cfg['encoder_name'],
        encoder_weights=cfg['encoder_weights'],
        decoder_name=cfg['decoder_name']
    )
    model.cuda()
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok = True)
    CHECKPOINT = '{}/{}_{}_{}.pt'.format(ckpt_dir, cfg['encoder_name'], cfg['decoder_name'], cfg['image_size'])
    LOG = '{}/{}_{}_{}.log'.format(ckpt_dir, cfg['encoder_name'], cfg['decoder_name'], cfg['image_size'])

    DiceLoss    = smp.losses.DiceLoss(mode='binary')
    TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['init_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs']-1)
    
    scaler = torch.cuda.amp.GradScaler()

    val_iou_max = 0
    logger = []

    for epoch in range(cfg['epochs']):
        scheduler.step()
        model.train()
        train_loss = []
            
        loop = tqdm(train_loader)
        for images, masks in loop:
            images = images.cuda()
            masks = masks.cuda()
        
            optimizer.zero_grad()

            if random.random() < 0.5:
                ### mixup
                lam = np.random.beta(0.5, 0.5)
                rand_index = torch.randperm(images.size(0))
                images = lam * images + (1 - lam) * images[rand_index, :,:,:]
                masks_a, masks_b = masks, masks[rand_index,:,:]
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss1 = lam * DiceLoss(outputs, masks_a) + (1 - lam) * DiceLoss(outputs, masks_b)
                    loss2 = lam * TverskyLoss(outputs, masks_a) + (1 - lam) * TverskyLoss(outputs, masks_b)
                    loss = 0.5*loss1 + 0.5*loss2
                    train_loss.append(loss.item())
            else:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss1 = DiceLoss(outputs, masks)
                    loss2 = TverskyLoss(outputs, masks)
                    loss = 0.5*loss1 + 0.5*loss2
                    train_loss.append(loss.item())
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, cfg['epochs']-1, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=np.mean(train_loss))
        train_loss = np.mean(train_loss)

        model.eval()
        val_iou = 0.0
        for images, masks in tqdm(valid_loader):
            images = images.cuda()
            masks = masks.cpu().type(torch.int64)

            with torch.cuda.amp.autocast(), torch.no_grad():
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).cpu().type(torch.int64)

                val_iou += mean_iou(outputs, masks, num_classes=1, per_class=False).mean().item()*images.size(0)
    
        val_iou /= len(valid_loader.dataset)

        print('train_loss: {:.5f} | val_iou: {:.5f}'.format(train_loss, val_iou))
        logger.append([epoch, round(optimizer.param_groups[0]['lr'], 8), round(train_loss, 5), round(val_iou, 5)])
        log_df = pd.DataFrame(data=np.array(logger), columns=['epoch', 'lr', 'train_loss', 'val_iou'])
        log_df.to_csv(LOG, index=False)

        if val_iou > val_iou_max:
            print('Valid iou improved from {:.5f} to {:.5f} saving model to {}'.format(val_iou_max, val_iou, CHECKPOINT))
            val_iou_max = val_iou
            torch.save(model.state_dict(), CHECKPOINT)