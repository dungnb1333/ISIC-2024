import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import DataLoader

from models import ISIC_Model
from dataset import ISIC_Dataset, classes
from util import seed_everything, comp_score

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default=None, type=str)
parser.add_argument("--ckpt_dir", default='checkpoints', type=str)
parser.add_argument("--pred_dir", default='prediction', type=str)
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)

args = parser.parse_args()
print(args)

seed_everything(seed=123)
   
if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    os.makedirs(args.pred_dir, exist_ok = True)
    model = ISIC_Model(
        model_name=cfg['model_name'], 
        pretrained=False, 
        num_classes=len(classes)
    )
    model.cuda()

    df = pd.read_csv('../../dataset/isic_2024/train_kfold.csv')

    mean_score = 0.0
    oof_image_paths = []
    oof_target = []
    for fold in args.folds:
        CHECKPOINT = '{}/{}_{}_fold{}.pt'.format(args.ckpt_dir, cfg['model_name'], cfg['image_size'], fold)
        model.load_state_dict(torch.load(CHECKPOINT))
        model.eval()
        
        test_df = df.loc[df['fold'] == fold]
        
        test_dataset = ISIC_Dataset(
            df=test_df,
            image_size=cfg['image_size'], 
            mode='test',
            neg_pos_ratio=None)
        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['workers'], shuffle=False)

        preds = []
        gts = []
        for images, labels, batch_image_path in tqdm(test_loader):
            images = images.cuda()
            gts.append(labels.data.cpu().numpy()[:,0])
            oof_image_paths.extend(batch_image_path)

            with torch.cuda.amp.autocast(), torch.no_grad():
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                preds.append(outputs.data.cpu().numpy()[:,0])
        preds = np.concatenate(preds, axis=None)
        gts = np.concatenate(gts, axis=None)
        oof_target.append(preds)
        
        val_score = comp_score(gts, preds)
        mean_score += val_score*len(test_df)
        print('Fold {}: score {:.5f}'.format(fold, val_score))
    mean_score /= len(df)
    print('OOF: score {:.5f}'.format(mean_score))

    pred_df = pd.DataFrame()
    pred_df['image_path'] = np.array(oof_image_paths)
    pred_df['target'] = np.concatenate(oof_target, axis=None)
    pred_df.to_csv('{}/{}_{}.csv'.format(args.pred_dir, cfg['model_name'], cfg['image_size']), index=False)
