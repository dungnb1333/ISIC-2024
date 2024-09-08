import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import os 
import numpy as np
import pandas as pd
import random
import random
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

def comp_score(y_true, y_pred, min_tpr=0.80):
    v_gt = abs(y_true-1)
    v_pred = np.array([1.0 - x for x in y_pred])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

def custom_lgbm_metric(y_true, y_pred):
    # TODO: Refactor with the above.
    min_tpr = 0.80
    v_gt = abs(y_true-1)
    v_pred = np.array([1.0 - x for x in y_pred])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return "pauc80", partial_auc, True

def custom_xgb_metric(y_pred, y_true):
    # TODO: Refactor with the above.
    min_tpr = 0.80
    v_gt = abs(y_true.get_label()-1)
    v_pred = np.array([1.0 - x for x in y_pred])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return "pauc80", -partial_auc

class CustomCatboostMetric:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, targets, weight):
        y_true = np.array(targets)
        y_pred = np.array(approxes).squeeze()
        
        min_tpr = 0.80
        v_gt = abs(y_true-1)
        v_pred = np.array([1.0 - x for x in y_pred])
        max_fpr = abs(1-min_tpr)
        partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
        partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
        
        return partial_auc, 0

def train_lgb(ckpt_dir, fold, lgbm_params, x_train, y_train, x_val, y_val, keep_ckpt_num):
    lgb_ckpt_dir = '{}/lgb/fold{}'.format(ckpt_dir, fold)
    os.makedirs(lgb_ckpt_dir, exist_ok=True)
    lgb_score_df_path = '{}/lgb_score.csv'.format(lgb_ckpt_dir)
    if os.path.isfile(lgb_score_df_path):
        lgb_score_df = pd.read_csv(lgb_score_df_path)
    else:
        lgb_score_df = pd.DataFrame(columns=['score', 'random_state', 'ckpt'])
    
    while True:
        random_state = random.randint(1,999)
        if random_state not in lgb_score_df.random_state.values:
            break

    idxs0 = list(np.where(y_train==0)[0])
    idxs1 = list(np.where(y_train==1)[0])
    idxs0 = random.sample(idxs0, 200*len(idxs1))

    y_train0 = y_train[idxs0]
    y_train1 = y_train[idxs1]

    x_train0 = x_train[idxs0,:]
    x_train1 = x_train[idxs1, :]

    x_train = np.concatenate([x_train0, x_train1], 0)
    y_train = np.concatenate([y_train0, y_train1], 0)
    x_train, y_train = shuffle(x_train, y_train, random_state=random_state)

    ### lgbm
    lgb_model = lgb.LGBMClassifier(random_state=random_state, **lgbm_params)
    lgb_model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        eval_metric=custom_lgbm_metric,
    )
    CHECKPOINT = '{}/lgb_fold{}_rs{}.txt'.format(lgb_ckpt_dir, fold, random_state)
    lgb_model.booster_.save_model(CHECKPOINT)
    lgb_model = lgb.Booster(model_file=CHECKPOINT)
    y_pred = lgb_model.predict(x_val)
    lgb_score = comp_score(y_val, y_pred)
    
    tmp_df = pd.DataFrame(data=np.array([[lgb_score, random_state, CHECKPOINT]]), columns=['score', 'random_state', 'ckpt'])
    tmp_df['score'] = tmp_df['score'].astype(float)
    tmp_df['random_state'] = tmp_df['random_state'].astype(int)
    tmp_df['ckpt'] = tmp_df['ckpt'].astype(str)
    if os.path.isfile(lgb_score_df_path):
        lgb_score_df = pd.concat([lgb_score_df, tmp_df], ignore_index=True)
    else:
        lgb_score_df = tmp_df
    lgb_score_df = lgb_score_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    lgb_score_df.to_csv(lgb_score_df_path, index=False)

    for index, row in lgb_score_df.iterrows():
        if index >= keep_ckpt_num:
            if os.path.isfile(row['ckpt']):
                os.remove(row['ckpt'])

    return lgb_score_df.score.values[0], lgb_score_df.random_state.values[0]
    
def train_cb(ckpt_dir, fold, cb_params, x_train, y_train, x_val, y_val, keep_ckpt_num):
    cb_ckpt_dir = '{}/cb/fold{}'.format(ckpt_dir, fold)
    os.makedirs(cb_ckpt_dir, exist_ok=True)
    cb_score_df_path = '{}/cb_score.csv'.format(cb_ckpt_dir)
    if os.path.isfile(cb_score_df_path):
        cb_score_df = pd.read_csv(cb_score_df_path)
    else:
        cb_score_df = pd.DataFrame(columns=['score', 'random_state', 'ckpt'])
    
    while True:
        random_state = random.randint(1,999)
        if random_state not in cb_score_df.random_state.values:
            break
    
    x_train['target'] = y_train
    x_train_0 = x_train.loc[x_train['target'] == 0]
    x_train_1 = x_train.loc[x_train['target'] == 1]
    x_train_0 = x_train_0.sample(n=200*len(x_train_1))
    x_train = pd.concat([x_train_0, x_train_1], ignore_index=True)
    x_train = x_train.sample(frac=1).reset_index(drop=True)
    y_train = x_train.target.values
    x_train = x_train.drop('target', axis=1)
    
    ### cb
    cb_model = cb.CatBoostClassifier(random_state=random_state, **cb_params, eval_metric=CustomCatboostMetric())
    cb_model.fit(x_train, y_train, eval_set=(x_val, y_val))

    CHECKPOINT = '{}/cb_fold{}_rs{}.cbm'.format(cb_ckpt_dir, fold, random_state)
    cb_model.save_model(CHECKPOINT, format="cbm")

    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(CHECKPOINT, format='cbm')
    y_pred = cb_model.predict_proba(x_val)[:, 1]
    cb_score = comp_score(y_val, y_pred)

    tmp_df = pd.DataFrame(data=np.array([[cb_score, random_state, CHECKPOINT]]), columns=['score', 'random_state', 'ckpt'])
    tmp_df['score'] = tmp_df['score'].astype(float)
    tmp_df['random_state'] = tmp_df['random_state'].astype(int)
    tmp_df['ckpt'] = tmp_df['ckpt'].astype(str)
    if os.path.isfile(cb_score_df_path):
        cb_score_df = pd.concat([cb_score_df, tmp_df], ignore_index=True)
    else:
        cb_score_df = tmp_df
    cb_score_df = cb_score_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    cb_score_df.to_csv(cb_score_df_path, index=False)

    for index, row in cb_score_df.iterrows():
        if index >= keep_ckpt_num:
            if os.path.isfile(row['ckpt']):
                os.remove(row['ckpt'])

    return cb_score_df.score.values[0], cb_score_df.random_state.values[0]

def train_xgb(ckpt_dir, fold, xgb_params, x_train, y_train, x_val, y_val, keep_ckpt_num):
    xgb_ckpt_dir = '{}/xgb/fold{}'.format(ckpt_dir, fold)
    os.makedirs(xgb_ckpt_dir, exist_ok=True)
    xgb_score_df_path = '{}/xgb_score.csv'.format(xgb_ckpt_dir)
    if os.path.isfile(xgb_score_df_path):
        xgb_score_df = pd.read_csv(xgb_score_df_path)
    else:
        xgb_score_df = pd.DataFrame(columns=['score', 'random_state', 'ckpt'])
    
    while True:
        random_state = random.randint(1,999)
        if random_state not in xgb_score_df.random_state.values:
            break

    idxs0 = list(np.where(y_train==0)[0])
    idxs1 = list(np.where(y_train==1)[0])
    idxs0 = random.sample(idxs0, 200*len(idxs1))

    y_train0 = y_train[idxs0]
    y_train1 = y_train[idxs1]

    x_train0 = x_train[idxs0,:]
    x_train1 = x_train[idxs1, :]

    x_train = np.concatenate([x_train0, x_train1], 0)
    y_train = np.concatenate([y_train0, y_train1], 0)
    x_train, y_train = shuffle(x_train, y_train, random_state=random_state)

    x_train[x_train == -np.inf] = 0
    x_train[x_train == np.inf] = 0

    ### xgb
    xgb_model = xgb.XGBClassifier(random_state=random_state, **xgb_params)
    xgb_model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        eval_metric=custom_xgb_metric,
        verbose=False)
    
    CHECKPOINT = '{}/xgb_fold{}_rs{}.json'.format(xgb_ckpt_dir, fold, random_state)
    xgb_model.get_booster().save_model(CHECKPOINT)
    xgb_model = xgb.Booster()
    xgb_model.load_model(CHECKPOINT)

    y_pred = xgb_model.predict(xgb.DMatrix(x_val), iteration_range=(0, xgb_model.best_iteration+1))
    xgb_score = comp_score(y_val, y_pred)
   
    tmp_df = pd.DataFrame(data=np.array([[xgb_score, random_state, CHECKPOINT]]), columns=['score', 'random_state', 'ckpt'])
    tmp_df['score'] = tmp_df['score'].astype(float)
    tmp_df['random_state'] = tmp_df['random_state'].astype(int)
    tmp_df['ckpt'] = tmp_df['ckpt'].astype(str)
    if os.path.isfile(xgb_score_df_path):
        xgb_score_df = pd.concat([xgb_score_df, tmp_df], ignore_index=True)
    else:
        xgb_score_df = tmp_df
    xgb_score_df = xgb_score_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    xgb_score_df.to_csv(xgb_score_df_path, index=False)

    for index, row in xgb_score_df.iterrows():
        if index >= keep_ckpt_num:
            if os.path.isfile(row['ckpt']):
                os.remove(row['ckpt'])

    return xgb_score_df.score.values[0], xgb_score_df.random_state.values[0]