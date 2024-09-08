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
from models import ISIC_Model
from dataset import ISIC_Dataset, classes
from util import seed_everything

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
    valid_df = train_df.loc[train_df['fold'] == 0]

    if args.frac != 1:
        print('Quick training')
        train_df = train_df.sample(frac=args.frac).reset_index(drop=True)
        valid_df = valid_df.sample(frac=args.frac).reset_index(drop=True)

    train_dataset = ISIC_Dataset(
        df=train_df,
        image_size=cfg['image_size'], 
        mode='train',
        downsampling_type=cfg['downsampling_type'],
        neg_pos_ratio=cfg['neg_pos_ratio'])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=True)
    
    print('TRAIN: {}'.format(len(train_loader.dataset)))

    model = ISIC_Model(
        model_name=cfg['model_name'], 
        pretrained=True, 
        num_classes=len(classes)
    )
    model.cuda()
    model_ema = ModelEmaV2(model, decay=cfg['ema_decay'], device=torch.device("cuda"))

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['init_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs']-1)

    scaler = torch.cuda.amp.GradScaler()

    LOG = '{}/{}_{}_ema.log'.format(args.ckpt_dir, cfg['model_name'], cfg['image_size'])
    logger = []

    val_score_max = 0
    count = 0
    for epoch in range(cfg['epochs']):
        if epoch > 0:
            train_loader.dataset.update_data()
        scheduler.step()
        model.train()
        train_loss = []

        loop = tqdm(train_loader)
        for images, labels in loop:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            if cfg['mixup']:
                if random.random() < 0.5:
                    ### mixup
                    lam = np.random.beta(0.5, 0.5)
                    rand_index = torch.randperm(images.size(0))
                    images = lam * images + (1 - lam) * images[rand_index, :,:,:]
                    labels_a, labels_b = labels, labels[rand_index]

                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                        train_loss.append(loss.item())
                else:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        train_loss.append(loss.item())
            else:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    train_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model_ema.update(model)

            loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, cfg['epochs']-1, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=np.mean(train_loss))
        train_loss = np.mean(train_loss)

        print('train loss: {:.5f}'.format(train_loss))
        logger.append([epoch, round(optimizer.param_groups[0]['lr'], 8), round(train_loss, 5)])
        log_df = pd.DataFrame(data=np.array(logger), columns=['epoch', 'lr', 'train_loss'])
        log_df.to_csv(LOG, index=False)
        
        if epoch == cfg['ema_saved_epoch']:
            CHECKPOINT = '{}/{}_{}_epoch{}_ema.pt'.format(args.ckpt_dir, cfg['model_name'], cfg['image_size'], epoch)
            torch.save(model_ema.module.state_dict(), CHECKPOINT)
            break