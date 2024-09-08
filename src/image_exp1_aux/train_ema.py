import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.utils.model_ema import ModelEmaV2
from models import IsicAuxModel
from dataset import ISIC_Dataset, classes
from util import seed_everything, comp_score
import segmentation_models_pytorch as smp

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default=None, type=str)
parser.add_argument("--frac", default=1.0, type=float)
parser.add_argument("--ckpt_dir", default='checkpoints_ema', type=str)

args = parser.parse_args()
print(args)

seed_everything(seed=123)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    os.makedirs(args.ckpt_dir, exist_ok = True)

    train_df = pd.read_csv('../../dataset/isic_2024/train_kfold.csv')
    mask_df = pd.read_csv('../../dataset/isic_2024/isic_2024_mask.csv')
    train_df = train_df.merge(mask_df, on = 'image_path', how = 'left')

    valid_df = train_df.loc[train_df['fold'] == 0]

    if cfg['use_ext_data']:
        df_2019 = pd.read_csv('../../dataset/isic_2019/train_kfold.csv')
        df_2020 = pd.read_csv('../../dataset/isic_2020/train_kfold.csv')
        df_pad = pd.read_csv('../../dataset/PAD-UFES-20/train_kfold.csv')

        mask_2020_df = pd.read_csv('../../dataset/isic_2020/isic_2020_mask.csv')
        df_2020 = df_2020.merge(mask_2020_df, on = 'image_path', how = 'left')

        mask_2019_df = pd.read_csv('../../dataset/isic_2019/isic_2019_mask.csv')
        df_2019 = df_2019.merge(mask_2019_df, on = 'image_path', how = 'left')

        mask_pad_df = pd.read_csv('../../dataset/PAD-UFES-20/train_kfold.csv')
        df_pad = df_pad.merge(mask_pad_df, on = 'image_path', how = 'left')

        train_df = pd.concat([train_df, df_2019, df_2020, df_pad], ignore_index=True)
    
    if args.frac != 1:
        print('Quick training')
        train_df = train_df.sample(frac=args.frac).reset_index(drop=True)
        valid_df = valid_df.sample(frac=args.frac).reset_index(drop=True)

    train_dataset = ISIC_Dataset(
        df=train_df,
        image_size=cfg['image_size'], 
        mode='train',
        downsampling=cfg['downsampling'], 
        downsampling_type=cfg['downsampling_type'],
        neg_pos_ratio=cfg['neg_pos_ratio'])
    valid_dataset = ISIC_Dataset(
        df=valid_df,
        image_size=cfg['image_size'], 
        mode='valid',
        downsampling=False, 
        downsampling_type=None,
        neg_pos_ratio=None)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=False)

    print('TRAIN: {} | VALID: {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))

    model = IsicAuxModel(
        encoder_name=cfg['model_name'], 
        in_features=cfg['in_features'],
        encoder_weights=cfg['encoder_weights'], 
        img_size=cfg['image_size'], 
        classes=len(classes)
    )
    model.cuda()
    model_ema = ModelEmaV2(model, decay=cfg['ema_decay'], device=torch.device("cuda"))

    cls_criterion = nn.BCEWithLogitsLoss()
    seg_criterion = smp.losses.SoftBCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['init_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs']-1)

    scaler = torch.cuda.amp.GradScaler()

    LOG = '{}/{}_{}_aux_ema.log'.format(args.ckpt_dir, cfg['model_name'], cfg['image_size'])
    logger = []

    val_score_max = 0
    count = 0
    for epoch in range(cfg['epochs']):
        if cfg['downsampling'] and epoch > 0:
            train_loader.dataset.update_data()
        scheduler.step()
        model.train()
        train_loss = []

        loop = tqdm(train_loader)
        for images, masks, labels in loop:
            images = images.cuda()
            labels = labels.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()

            if cfg['mixup']:
                if random.random() < 0.5:
                    ### mixup
                    lam = np.random.beta(0.5, 0.5)
                    rand_index = torch.randperm(images.size(0))
                    images = lam * images + (1 - lam) * images[rand_index, :,:,:]
                    labels_a, labels_b = labels, labels[rand_index]
                    masks_a, masks_b = masks, masks[rand_index,:,:]
            
                    with torch.cuda.amp.autocast():
                        outputs1, outputs2 = model(images)
                        cls_loss = lam * cls_criterion(outputs1, labels_a) + (1 - lam) * cls_criterion(outputs1, labels_b)
                        seg_loss = lam * seg_criterion(outputs2, masks_a) + (1 - lam) * seg_criterion(outputs2, masks_b)
                        loss = 0.6*cls_loss + 0.4*seg_loss
                        train_loss.append(loss.item())
                else:
                    with torch.cuda.amp.autocast():
                        outputs1, outputs2 = model(images)
                        cls_loss = cls_criterion(outputs1, labels)
                        seg_loss = seg_criterion(outputs2, masks)
                        loss = 0.6*cls_loss + 0.4*seg_loss
                        train_loss.append(loss.item())
            else:
                with torch.cuda.amp.autocast():
                    outputs1, outputs2 = model(images)
                    cls_loss = cls_criterion(outputs1, labels)
                    seg_loss = seg_criterion(outputs2, masks)
                    loss = 0.6*cls_loss + 0.4*seg_loss
                    train_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model_ema.update(model)

            loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, cfg['epochs']-1, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=np.mean(train_loss))
        train_loss = np.mean(train_loss)

        model.eval()
        model_ema.eval()

        gts = []
        preds = []
        ema_preds = []
        for images, labels, _ in tqdm(valid_loader):
            images = images.cuda()
            labels = labels.cuda()
            gts.append(labels.data.cpu().numpy()[:,0])

            with torch.cuda.amp.autocast(), torch.no_grad():
                outputs, _ = model(images)
                outputs = torch.sigmoid(outputs)
                preds.append(outputs.data.cpu().numpy()[:,0])

                ema_outputs, _ = model_ema.module(images)
                ema_outputs = torch.sigmoid(ema_outputs)
                ema_preds.append(ema_outputs.data.cpu().numpy()[:,0])

        gts = np.concatenate(gts, axis=None)
        preds = np.concatenate(preds, axis=None)
        ema_preds = np.concatenate(ema_preds, axis=None)
        val_score = comp_score(gts, preds)
        ema_val_score = comp_score(gts, ema_preds)

        print('train loss: {:.5f} | val_score: {:.5f} | ema_val_score: {:.5f}'.format(train_loss, val_score, ema_val_score))
        logger.append([epoch, round(optimizer.param_groups[0]['lr'], 8), round(train_loss, 5), round(val_score, 5), round(ema_val_score, 5)])
        log_df = pd.DataFrame(data=np.array(logger), columns=['epoch', 'lr', 'train_loss', 'val_score', 'ema_val_score'])
        log_df.to_csv(LOG, index=False)
        
        if epoch == cfg['ema_saved_epoch']:
            CHECKPOINT = '{}/{}_{}_epoch{}_aux_ema.pt'.format(args.ckpt_dir, cfg['model_name'], cfg['image_size'], epoch)
            torch.save(model_ema.module.state_dict(), CHECKPOINT)
            break