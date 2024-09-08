import os 
import numpy as np
import pandas as pd
from util import train_cb, train_lgb, train_xgb
import shutil

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    ckpt_dir = 'checkpoints_10model_feat'
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        os.makedirs(ckpt_dir, exist_ok=True)

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

    lgbm_params = {
        'objective':        'binary',
        'verbosity':        -1,
        'n_estimators':     300,
        'early_stopping_rounds': 50,
        'metric': 'custom',
        'boosting_type':    'gbdt',
        'lambda_l1':        0.08758718919397321, 
        'lambda_l2':        0.0039689175176025465, 
        'learning_rate':    0.03231007103195577, 
        'max_depth':        4, 
        'num_leaves':       128, 
        'colsample_bytree': 0.8329551585827726, 
        'colsample_bynode': 0.4025961355653304, 
        'bagging_fraction': 0.7738954452473223, 
        'bagging_freq':     4, 
        'min_data_in_leaf': 85, 
        'scale_pos_weight': 2.7984184778875543,
        "device": "gpu"
    }

    cb_params = {
        'loss_function':     'Logloss',
        'iterations':        300,
        'early_stopping_rounds': 50,
        'verbose':           False,
        'max_depth':         7, 
        'learning_rate':     0.06936242010150652, 
        'scale_pos_weight':  2.6149345838209532, 
        'l2_leaf_reg':       6.216113851699493,
        'min_data_in_leaf':  24,
        'cat_features':      cat_cols,
        "task_type": "CPU",
    }

    xgb_params = {
        'enable_categorical':       True,
        'tree_method':              'hist',
        'disable_default_eval_metric': 1,
        'n_estimators':             300,
        'early_stopping_rounds':    50,
        'learning_rate':            0.08501257473292347, 
        'lambda':                   8.879624125465703, 
        'alpha':                    0.6779926606782505, 
        'max_depth':                6, 
        'subsample':                0.6012681388711075, 
        'colsample_bytree':         0.8437772277074493, 
        'colsample_bylevel':        0.5476090898823716, 
        'colsample_bynode':         0.9928601203635129, 
        'scale_pos_weight':         3.29440313334688,
        "device":                   "cuda",
    }

    data_dict = {}
    for fold in range(5):
        df_train = df.loc[df['fold'] != fold].reset_index(drop=True)
        df_val = df.loc[df['fold'] == fold].reset_index(drop=True)

        x_train = df_train[train_cols]
        y_train = df_train.target.values

        x_val = df_val[train_cols]
        y_val = df_val.target.values

        data_dict[fold] = [x_train, y_train, x_val, y_val]

    fold_num = 5
    try_num = 40
    keep_ckpt_num = 5
    log_file = open('{}/log.txt'.format(ckpt_dir), 'w')
    for num in range(try_num):
        log_file.write('******************************************** {}/{} ********************************************\n'.format(num, try_num))
        print('******************************************** {}/{} ********************************************'.format(num, try_num))
        mean_lgb = []
        mean_cb = []
        mean_xgb = []
        for fold in range(fold_num):
            x_train, y_train, x_val, y_val = data_dict[fold]

            lgb_max_score, lgb_random_state = train_lgb(ckpt_dir, fold, lgbm_params, x_train.values.copy(), y_train.copy(), x_val.values, y_val, keep_ckpt_num)
            cb_max_score, cb_random_state = train_cb(ckpt_dir, fold, cb_params, x_train.copy(), y_train.copy(), x_val, y_val, keep_ckpt_num)
            xgb_max_score, xgb_random_state = train_xgb(ckpt_dir, fold, xgb_params, x_train.values.copy(), y_train.copy(), x_val.values.copy(), y_val.copy(), keep_ckpt_num)

            print("Fold: {} - lgb max score {:.5f} with rs {} - cb max score: {:.5f} with rs {} - xgb max score: {:.5f} with rs {}".format(fold, lgb_max_score, lgb_random_state, cb_max_score, cb_random_state, xgb_max_score, xgb_random_state))
            log_file.write("Fold: {} - lgb max score {:.5f} with rs {} - cb max score: {:.5f} with rs {} - xgb max score: {:.5f} with rs {}\n".format(fold, lgb_max_score, lgb_random_state, cb_max_score, cb_random_state, xgb_max_score, xgb_random_state))
            mean_lgb.append(lgb_max_score)
            mean_cb.append(cb_max_score)
            mean_xgb.append(xgb_max_score)
        mean_lgb = np.mean(mean_lgb)
        mean_cb = np.mean(mean_cb)
        mean_xgb = np.mean(mean_xgb)
        print('Mean - lgb score {:.5f} - cb score: {:.5f} - xgb score: {:.5f}'.format(mean_lgb, mean_cb, mean_xgb))
        log_file.write('Mean - lgb score {:.5f} - cb score: {:.5f} - xgb score: {:.5f}\n'.format(mean_lgb, mean_cb, mean_xgb))
    log_file.close()