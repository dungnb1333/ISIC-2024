import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.stats import gmean
from util import comp_score

if __name__ == "__main__":
    ckpt_dir = 'checkpoints_10model_feat'

    df_2024_vit_tiny_384 = pd.read_csv('../image_exp3/prediction/vit_tiny_patch16_384.augreg_in21k_ft_in1k_384.csv')
    dict_2024_vit_tiny_384 = dict(zip(df_2024_vit_tiny_384.image_path.values, df_2024_vit_tiny_384.target.values))

    df_2024_swin_tiny_256 = pd.read_csv('../image_exp3/prediction/swinv2_tiny_window8_256.ms_in1k_256.csv')
    dict_2024_swin_tiny_256 = dict(zip(df_2024_swin_tiny_256.image_path.values, df_2024_swin_tiny_256.target.values))

    df_2024_convnextv2_tiny_288 = pd.read_csv('../image_exp3/prediction/convnextv2_tiny.fcmae_ft_in22k_in1k_288.csv')
    dict_2024_convnextv2_tiny_288 = dict(zip(df_2024_convnextv2_tiny_288.image_path.values, df_2024_convnextv2_tiny_288.target.values))

    df_2024_swin_tiny_224 = pd.read_csv('../image_exp0/prediction/swin_tiny_patch4_window7_224.ms_in1k_224.csv')
    dict_2024_swin_tiny_224 = dict(zip(df_2024_swin_tiny_224.image_path.values, df_2024_swin_tiny_224['target'].values))

    df_2024_convnextv2_base_128 = pd.read_csv('../image_exp0/prediction/convnextv2_base.fcmae_ft_in22k_in1k_128.csv')
    dict_2024_convnextv2_base_128 = dict(zip(df_2024_convnextv2_base_128.image_path.values, df_2024_convnextv2_base_128.target.values))

    df_2024_convnextv2_large_64 = pd.read_csv('../image_exp0/prediction/convnextv2_large.fcmae_ft_in22k_in1k_64.csv')
    dict_2024_convnextv2_large_64 = dict(zip(df_2024_convnextv2_large_64.image_path.values, df_2024_convnextv2_large_64.target.values))

    df_2024_coatnet_224 = pd.read_csv('../image_exp0/prediction/coatnet_rmlp_1_rw_224.sw_in1k_224.csv')
    dict_2024_coatnet_224 = dict(zip(df_2024_coatnet_224.image_path.values, df_2024_coatnet_224['target'].values))

    df_2024_eb3_aux_224 = pd.read_csv('../image_exp1_aux/prediction/timm-efficientnet-b3_224_aux.csv')
    dict_2024_eb3_aux_224 = dict(zip(df_2024_eb3_aux_224.image_path.values, df_2024_eb3_aux_224.target.values))

    df_2024_mit_b0_aux_384 = pd.read_csv('../image_exp1_aux/prediction/mit_b0_384_aux.csv')
    dict_2024_mit_b0_aux_384 = dict(zip(df_2024_mit_b0_aux_384.image_path.values, df_2024_mit_b0_aux_384.target.values))

    df_2024_mit_b5_aux_224 = pd.read_csv('../image_exp1_aux/prediction/mit_b5_224_aux.csv')
    dict_2024_mit_b5_aux_224 = dict(zip(df_2024_mit_b5_aux_224.image_path.values, df_2024_mit_b5_aux_224['target'].values))

    df = pd.read_csv('../../dataset/isic_2024/train_tabular_kfold_v2.csv')
    meta = []
    for _, row in df.iterrows():
        image_path = '../../dataset/isic_2024/train-image/{}.jpg'.format(row['isic_id'])
        feat = []
        feat.append(dict_2024_vit_tiny_384[image_path])
        feat.append(dict_2024_swin_tiny_256[image_path])
        feat.append(dict_2024_convnextv2_tiny_288[image_path])
        feat.append(dict_2024_swin_tiny_224[image_path])
        feat.append(dict_2024_convnextv2_base_128[image_path])
        feat.append(dict_2024_convnextv2_large_64[image_path])
        feat.append(dict_2024_coatnet_224[image_path])
        feat.append(dict_2024_eb3_aux_224[image_path])
        feat.append(dict_2024_mit_b5_aux_224[image_path])
        feat.append(dict_2024_mit_b0_aux_384[image_path])
        meta.append(feat)

    image_pred_classes = [
        'vit_tiny_384',
        'swin_tiny_256',
        'convnextv2_tiny_288',
        'swin_tiny_224',
        'convnextv2_base_128',
        'convnextv2_large_64',
        'coatnet_224',
        'eb3_aux_224',
        'mit_b5_aux_224',
        'mit_b0_aux_384',
    ]

    df[image_pred_classes] = np.array(meta)

    f = open('../../dataset/isic_2024/tabular_train_cols_v2.txt', 'r')
    train_cols = f.readlines()
    train_cols = [x.strip() for x in train_cols]
    f.close()

    f = open('../../dataset/isic_2024/tabular_cat_cols_v2.txt', 'r')
    cat_cols = f.readlines()
    cat_cols = [x.strip() for x in cat_cols]
    f.close()

    train_cols = image_pred_classes + train_cols

    mean_score = 0.0
    lgb_mean_score = 0.0
    cb_mean_score = 0.0
    xgb_mean_score = 0.0
    keep_ckpt_num = 5
    for fold in range(5):
        df_val = df.loc[df['fold'] == fold].reset_index(drop=True)
        x_val = df_val[train_cols]
        y_val = df_val.target.values
        isic_ids = df_val.isic_id.values
        
        lgb_ckpt_dir = '{}/lgb/fold{}'.format(ckpt_dir, fold)
        cb_ckpt_dir = '{}/cb/fold{}'.format(ckpt_dir, fold)
        xgb_ckpt_dir = '{}/xgb/fold{}'.format(ckpt_dir, fold)

        lgb_score_df = pd.read_csv('{}/lgb_score.csv'.format(lgb_ckpt_dir))
        lgb_score_df = lgb_score_df.sort_values(by='score', ascending=False).reset_index(drop=True).head(keep_ckpt_num)

        cb_score_df = pd.read_csv('{}/cb_score.csv'.format(cb_ckpt_dir))
        cb_score_df = cb_score_df.sort_values(by='score', ascending=False).reset_index(drop=True).head(keep_ckpt_num)

        xgb_score_df = pd.read_csv('{}/xgb_score.csv'.format(xgb_ckpt_dir))
        xgb_score_df = xgb_score_df.sort_values(by='score', ascending=False).reset_index(drop=True).head(keep_ckpt_num)

        lgb_pred_ens = []
        for rs in lgb_score_df.random_state.values:
            ckpt_path = '{}/lgb_fold{}_rs{}.txt'.format(lgb_ckpt_dir, fold, rs)
            lgb_model = lgb.Booster(model_file=ckpt_path)
            lgb_y_pred = lgb_model.predict(x_val.values)
            lgb_pred_ens.append(lgb_y_pred)
        lgb_pred_ens = np.array(lgb_pred_ens)

        cb_pred_ens = []
        for rs in cb_score_df.random_state.values:
            ckpt_path = '{}/cb_fold{}_rs{}.cbm'.format(cb_ckpt_dir, fold, rs)
            cb_model = cb.CatBoostClassifier()
            cb_model.load_model(ckpt_path, format='cbm')
            cb_y_pred = cb_model.predict_proba(x_val)[:, 1]
            cb_pred_ens.append(cb_y_pred)
        cb_pred_ens = np.array(cb_pred_ens)

        xgb_pred_ens = []
        for rs in xgb_score_df.random_state.values:
            ckpt_path = '{}/xgb_fold{}_rs{}.json'.format(xgb_ckpt_dir, fold, rs)

            xgb_model = xgb.Booster()
            xgb_model.load_model(ckpt_path)
            x_val_xgb = x_val.values.copy()
            x_val_xgb[x_val_xgb == -np.inf] = 0
            x_val_xgb[x_val_xgb == np.inf] = 0

            xgb_y_pred = xgb_model.predict(xgb.DMatrix(x_val_xgb), iteration_range=(0, xgb_model.best_iteration+1))
            xgb_pred_ens.append(xgb_y_pred)
        xgb_pred_ens = np.array(xgb_pred_ens)

        pred_ens = np.concatenate((lgb_pred_ens, cb_pred_ens, xgb_pred_ens), 0)

        pred_ens = gmean(pred_ens, 0)
        lgb_pred_ens = gmean(lgb_pred_ens, 0)
        cb_pred_ens = gmean(cb_pred_ens, 0)
        xgb_pred_ens = gmean(xgb_pred_ens, 0)

        lgb_score = comp_score(y_val, lgb_pred_ens)
        cb_score = comp_score(y_val, cb_pred_ens)
        xgb_score = comp_score(y_val, xgb_pred_ens)
        gmean_score = comp_score(y_val, pred_ens)

        print('Fold {} | gmean ens {:.5f} | lgb {:.5f} | cb {:.5f} | xgb {:.5f}'.format(fold, gmean_score, lgb_score, cb_score, xgb_score))
        
        lgb_mean_score += lgb_score
        cb_mean_score += cb_score
        xgb_mean_score += xgb_score
        mean_score += gmean_score

    lgb_mean_score /= 5.0
    cb_mean_score /= 5.0
    xgb_mean_score /= 5.0
    mean_score /= 5.0
    print("OOF | mean score: {:.5f} | lgb {:.5f} | cb {:.5f} | xgb {:.5f}".format(mean_score, lgb_mean_score, cb_mean_score, xgb_mean_score))
   