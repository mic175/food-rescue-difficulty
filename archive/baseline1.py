############################
# baseline1.py  (2025‑04‑11)
############################
"""
Pure‑tabular baseline that reproduces Baseline 1 in the WWW’24 paper
(≈ 0.54 AUC easy, 0.50 AUC hard; ≈ 0.60 / 0.55 accuracy).

MOD summary
-----------
1.  Replaced regressors with classifiers + class‑weight balancing
2.  Train directly on 0/1 labels (removed logit/sigmoid hack)
3.  Threshold chosen to maximise validation accuracy
4.  Removed 50 k random down‑sample (use full labelled set)
5.  Keep *both* positive and negative expert labels in train/val
6.  Correct precision‑recall AUC call (fixes TypeError)
"""

import os, json, yaml
from datetime import datetime
from typing import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_recall_curve, auc)
# ==== MOD‑1 BEGIN – classifiers instead of regressors ================
from sklearn.ensemble   import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm        import SVC
from sklearn.neighbors  import KNeighborsClassifier
import lightgbm as lgb   # LGBMClassifier will be used
# ==== MOD‑1 END ======================================================

# --------------------------------------------------------------------
# 0. Database connection
# --------------------------------------------------------------------
with open('database.yaml') as f:
    db_cfg = yaml.safe_load(f)

DB_URL = f"postgresql://{db_cfg['user']}:{db_cfg['pass']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['db']}"
engine  = create_engine(DB_URL)

# --------------------------------------------------------------------
# 1. Difficulty CSV  (fixed easy‑class mapping; see MOD‑1.1)
# --------------------------------------------------------------------
def create_difficulty_csv(split, root='.'):
    with engine.connect() as con:
        df_tr = pd.read_sql_table('train_annot', con=con, schema='derivative')
        df_te = pd.read_sql_table('test_annot',  con=con, schema='derivative')

    # ==== MOD‑1.1 BEGIN – correct positive‑class assignment =========
    if split == 'easy':
        df_tr.loc[df_tr['label'] == -1, 'label'] = 1
        df_tr.loc[df_tr['label'] != -1, 'label'] = 0
        df_te.loc[df_te['label'] == -1, 'label'] = 1
        df_te.loc[df_te['label'] != -1, 'label'] = 0
    else:                               # hard task
        df_tr.loc[df_tr['label'] ==  1, 'label'] = 1
        df_tr.loc[df_tr['label'] !=  1, 'label'] = 0
        df_te.loc[df_te['label'] ==  1, 'label'] = 1
        df_te.loc[df_te['label'] !=  1, 'label'] = 0
    # ==== MOD‑1.1 END ===============================================

    df_tr.rename(columns={'label':'difficulty'}, inplace=True)
    df_te.rename(columns={'label':'difficulty'}, inplace=True)
    df_tr[['rescue_id','difficulty']].to_csv(f'{root}/train_{split}.csv', index=False)
    df_te[['rescue_id','difficulty']].to_csv(f'{root}/test_{split}.csv',  index=False)

# --------------------------------------------------------------------
# 2. Feature helpers  (unchanged)
# --------------------------------------------------------------------
def get_base_features(df_all):
    cols = [c for c in df_all.columns
            if c not in ['user_phone','donor_phone','recipient_phone','recurrence_id']]

    continuous = [
        'HourlyDewPointTemperature','HourlyDryBulbTemperature','HourlyPrecipitation',
        'HourlyRelativeHumidity','HourlyStationPressure','HourlyVisibility',
        'HourlyWetBulbTemperature','HourlyWindSpeed',
        'recipient_longitude','recipient_latitude','donor_longitude','donor_latitude',
        'total_quantity','user_longitude','user_latitude','user2donor','user2recipient',
        'donor_exp','recipient_exp','user_exp',
        'avg_past_user_rating','avg_past_recipient_rating','avg_past_donor_rating',
        'published_Y','published_M','published_D','published_H'
    ]
    discrete, onehot_prefix = [], [
        'recipient_household_size','food_restriction_id','food_type_id',
        'nonprofit_category_id','population_type_id','food_id','vehicle_type'
    ]
    for f in cols:
        if "_".join(f.split("_")[:-1]) in onehot_prefix:
            discrete.append(f)
    return continuous, discrete

# --------------------------------------------------------------------
# 3. Train / val / test split
# --------------------------------------------------------------------
def split_tabular(split, data, features, split_type):
    data['published_at'] = pd.to_datetime(data['published_at'], errors='coerce')
    data = data.dropna(subset=['published_at'])

    annot_test = pd.read_sql("SELECT * FROM derivative.test_annot;", engine)
    test_ids   = annot_test['rescue_id'].tolist()

    # mark test‑set difficulty
    easy_ids = annot_test.loc[annot_test['label']==-1,'rescue_id']
    hard_ids = annot_test.loc[annot_test['label']== 1,'rescue_id']
    test_set = data[data['rescue_id'].isin(test_ids)].copy()
    if split=='easy':
        test_set['difficulty'] = test_set['rescue_id'].isin(easy_ids).astype(int)
    else:
        test_set['difficulty'] = test_set['rescue_id'].isin(hard_ids).astype(int)

    # ==== MOD‑5 BEGIN – keep both positive and negative labels =======
    annot_train = pd.read_sql("SET search_path TO derivative; SELECT * FROM train_annot;", engine)
    if split=='easy':
        pos_ids = annot_train.loc[annot_train['label']==-1, 'rescue_id']
    else:
        pos_ids = annot_train.loc[annot_train['label']== 1, 'rescue_id']
    data['difficulty'] = data['rescue_id'].isin(pos_ids).astype(int)
    train_val = data[~data['rescue_id'].isin(test_ids)].copy()
    # ==== MOD‑5 END ==================================================

    cutoff = datetime(2022,3,1)
    if split_type=='time':
        train_set = train_val[train_val['published_at'] <  cutoff]
        val_set   = train_val[train_val['published_at'] >= cutoff]
    else:                                   # raw
        train_set = train_val[train_val['published_at'] <  cutoff]
        val_set   = train_val[train_val['published_at'] >= cutoff]

    train_set = train_set[features+['difficulty']]
    val_set   = val_set  [features+['difficulty']]
    test_set  = test_set [features+['difficulty']]
    return train_set, val_set, test_set

# --------------------------------------------------------------------
# 4. Training & evaluation
# --------------------------------------------------------------------
def train_eval_tabular(train_df, val_df, test_df,
                       cont_feat, disc_feat, ckpt_dir, seed, split_type):

    os.makedirs(ckpt_dir, exist_ok=True)
    feats = cont_feat + disc_feat
    imp   = SimpleImputer(strategy='mean')

    # ==== MOD‑2 BEGIN – train on binary labels directly ==============
    X_tr = imp.fit_transform(train_df[feats]);  y_tr = train_df['difficulty'].values
    X_val= imp.transform(val_df[feats]);        y_val= val_df['difficulty'].values
    # ==== MOD‑2 END ==================================================

    models = {
        'lgb': lgb.LGBMClassifier(random_state=seed, objective='binary',
                                  class_weight='balanced', n_estimators=300),
        'rf' : RandomForestClassifier(random_state=seed, class_weight='balanced'),
        'lr' : LogisticRegression(max_iter=1000, class_weight='balanced'),
        'svm': SVC(probability=True, class_weight='balanced', random_state=seed),
        'mlp': MLPClassifier(random_state=seed, max_iter=600),
        'knn': KNeighborsClassifier()
    }

    res, best_thr = {'val':{}, 'test':{}}, {}

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p_val = model.predict_proba(X_val)[:,1]

        # ==== MOD‑3 BEGIN – threshold via validation accuracy =========
        grid = np.linspace(0,1,101)
        thr  = grid[np.argmax([accuracy_score(y_val,(p_val>=t).astype(int)) for t in grid])]
        best_thr[name] = thr
        # ==== MOD‑3 END ==============================================

        preds = (p_val>=thr).astype(int)
        prec, rec, _ = precision_recall_curve(y_val, p_val)     # MOD‑6 line 1
        res['val'][name] = {
            'acc': accuracy_score(y_val,preds),
            'f1' : f1_score(y_val,preds,zero_division=0),
            'roc': roc_auc_score(y_val,p_val),
            'pr' : auc(rec, prec)                               # MOD‑6 line 2
        }

    # ----------- final test -----------
    X_tv = imp.fit_transform(pd.concat([train_df[feats], val_df[feats]]))
    y_tv = pd.concat([train_df['difficulty'], val_df['difficulty']]).values
    X_te = imp.transform(test_df[feats]);  y_te = test_df['difficulty'].values

    for name, model in models.items():
        model.fit(X_tv, y_tv)
        p_te = model.predict_proba(X_te)[:,1]
        thr  = best_thr[name]
        preds= (p_te>=thr).astype(int)

        prec, rec, _ = precision_recall_curve(y_te, p_te)       # MOD‑6 line 3
        res['test'][name] = {
            'acc': accuracy_score(y_te,preds),
            'f1' : f1_score(y_te,preds,zero_division=0),
            'roc': roc_auc_score(y_te,p_te),
            'pr' : auc(rec, prec)                               # MOD‑6 line 4
        }

    with open(f"{ckpt_dir}/metrics-{split_type}.json",'w') as fp:
        json.dump(res, fp, indent=2)
    return res

# --------------------------------------------------------------------
# 5. MAIN experiment loop
# --------------------------------------------------------------------
if __name__ == "__main__":
    seeds = [42,43,44,45,46,47,48,49,50,51]

    for split in ['easy','hard']:
        if not os.path.exists(f"train_{split}.csv"):
            create_difficulty_csv(split)

        model_names = ['lgb','rf','lr','svm','mlp','knn']
        agg = {m:{k:[] for k in ['acc','f1','roc','pr']} for m in model_names}

        for sd in seeds:
            print(f"\n--- Baseline‑1  split={split}  seed={sd} ---")

            # ==== MOD‑4 BEGIN – use full labelled set (no sampling) ===
            df_all = pd.read_csv('./all_rescues_info-noorg.csv', engine='python',
                                 sep=',', on_bad_lines='skip')
            # ==== MOD‑4 END ===========================================

            df_tr = pd.read_csv(f"train_{split}.csv")
            df_te = pd.read_csv(f"test_{split}.csv")
            df_lbl= pd.concat([df_tr, df_te])

            df_merge = df_all.merge(df_lbl, on='rescue_id', how='inner')

            cont, disc      = get_base_features(df_merge)
            tr, val, te     = split_tabular(split, df_merge, cont+disc, 'raw')
            ckpt_dir        = f"./ckpt-Baseline1-{split}-seed{sd}"
            res             = train_eval_tabular(tr,val,te,cont,disc,ckpt_dir,sd,'raw')

            for m in model_names:
                for k in agg[m]:
                    agg[m][k].append(res['test'][m][k])

        print(f"\n=== Aggregated Baseline‑1 ({split}) over {len(seeds)} seeds ===")
        for m in model_names:
            mean_std = {k:(np.mean(v), np.std(v)) for k,v in agg[m].items()}
            print(f"{m:3} | "
                  f"Acc={mean_std['acc'][0]:.3f}±{mean_std['acc'][1]:.3f}  "
                  f"F1={mean_std['f1'][0]:.3f}±{mean_std['f1'][1]:.3f}  "
                  f"AUC={mean_std['roc'][0]:.3f}±{mean_std['roc'][1]:.3f}  "
                  f"PR={mean_std['pr'][0]:.3f}±{mean_std['pr'][1]:.3f}")
