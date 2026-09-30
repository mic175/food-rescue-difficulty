#########################################
# baseline2.py (Final "Baseline2" from paper)
#########################################

import os
import json
import pickle
from datetime import datetime
from typing import *
from tqdm import tqdm

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.special import softmax

# metrics / preprocessing
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_recall_curve, auc, roc_curve
)
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    print("Please install imblearn (pip install imblearn) for oversampling.")
    raise

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb

# HF / Datasets
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)

##############################################
# 1. LOAD DATABASE CONFIG & CREATE ENGINE
##############################################
with open('database.yaml') as f:
    db_config = yaml.safe_load(f)

DB_URL = (
    f"postgresql://{db_config['user']}:{db_config['pass']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['db']}"
)
engine = create_engine(DB_URL)

pd.options.mode.chained_assignment = None

##############################################
# 2. CREATE DIFFICULTY CSV
##############################################
def create_difficulty_csv(split, root='.'):
    """
    Creates train_{split}.csv and test_{split}.csv from derivative.train_annot & test_annot.
    If split='easy':
      label=1 for rows originally labeled -1, else 0
    If split='hard':
      label=1 for rows originally labeled 1, else 0
    """
    with engine.connect() as con:
        df_train_labeled = pd.read_sql_table('train_annot', con=con, schema='derivative')
        df_test_labeled  = pd.read_sql_table('test_annot',  con=con, schema='derivative')

    if split == 'easy':
        df_train = df_train_labeled.copy()
        df_train.loc[df_train['label'] != -1, 'label'] = 0
        df_train.loc[df_train['label'] == -1, 'label'] = 1

        df_test = df_test_labeled.copy()
        df_test.loc[df_test['label'] != -1, 'label'] = 0
        df_test.loc[df_test['label'] == -1, 'label'] = 1
    else:  # 'hard'
        df_train = df_train_labeled.copy()
        df_train.loc[df_train['label'] != 1, 'label'] = 0
        df_train.loc[df_train['label'] == 1, 'label'] = 1

        df_test = df_test_labeled.copy()
        df_test.loc[df_test['label'] != 1, 'label'] = 0
        df_test.loc[df_test['label'] == 1, 'label'] = 1

    df_train = df_train.rename(columns={'label':'difficulty'})
    df_test  = df_test.rename(columns={'label':'difficulty'})

    df_train = df_train[['rescue_id','rating','text','difficulty']]
    df_test  = df_test[['rescue_id','rating','text','difficulty']]

    df_train.to_csv(os.path.join(root,f'train_{split}.csv'), index=False)
    df_test.to_csv(os.path.join(root, f'test_{split}.csv'), index=False)

##############################################
# 3. LOAD DIFFICULTY AS HF DATASET
##############################################
def create_difficulty_dataset(split, root='.'):
    data_files = {
        'train': os.path.join(root, f'train_{split}.csv'),
        'test':  os.path.join(root, f'test_{split}.csv')
    }
    ds = load_dataset("csv", data_files=data_files)
    return ds

def preprocess_difficulty_dataset(hf_dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        txts = examples["text"]
        if isinstance(txts, list):
            txts = [str(x) if isinstance(x,(str,bytes)) else '' for x in txts]
        elif not isinstance(txts, str):
            txts = ''
        return tokenizer(txts, padding="max_length", truncation=True, max_length=96)

    def map_difficulty(examples):
        labs = np.array(examples["difficulty"], dtype=int)
        return {"labels": labs.tolist()}

    ds_token = hf_dataset.map(tokenize_function, batched=True)
    ds_final = ds_token.map(map_difficulty, batched=True)
    return ds_final

##############################################
# 4. TRAIN/EVAL BERT ON DIFFICULTY
##############################################
def train_eval_text_difficulty(train_dataset, eval_dataset, split_name, seed=42):
    """
    Baseline2: Train BERT directly on difficulty labels.
    """
    model_name = "bert-base-cased"
    num_labels = 2

    # 1) Create model & tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # <--- keep the same base

    out_dir = f"./ckpt-baseline2-{split_name}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # 2) Standard training args
    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="no",
        num_train_epochs=6.0,
        logging_steps=10000,
        save_steps=10000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        seed=seed
    )

    # 3) Metrics
    roc_metric = evaluate.load("roc_auc")
    f_metric   = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax(logits, axis=1)
        auc_val = roc_auc_score(labels, probs[:,1])
        preds = np.argmax(logits, axis=1)
        f1_val = f_metric.compute(average="weighted",
                                  references=labels,
                                  predictions=preds)
        return {"roc_auc": auc_val,
                "f1_weighted": f1_val["f1"]}

    # 4) Create Trainer
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    print(f"\n=== BERT (Baseline2) training on {split_name} ===")
    trainer.train()

    # 5) Evaluate on eval_dataset
    results = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"Final BERT AUC => {split_name}: {results.get('eval_roc_auc','n/a')}")
    print(f"Final BERT Weighted-F1 => {split_name}: {results.get('eval_f1_weighted','n/a')}")

    # 6) Save the model, config, and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)  # <--- crucial fix

    return out_dir


##############################################
# 5. GET BASE FEATURES
##############################################
def get_base_features(df_all):
    if df_all is None:
        with open("all_rescues_info-noorg_colname.pkl","rb") as fp:
            columns = pickle.load(fp)
    else:
        columns = list(df_all.columns)
        for feat in ['user_phone','donor_phone','recipient_phone','recurrence_id']:
            if feat in columns:
                columns.remove(feat)

    continuous_feat = [
        'HourlyDewPointTemperature','HourlyDryBulbTemperature','HourlyPrecipitation',
        'HourlyRelativeHumidity','HourlyStationPressure','HourlyVisibility',
        'HourlyWetBulbTemperature','HourlyWindSpeed',
        'recipient_longitude','recipient_latitude',
        'donor_longitude','donor_latitude','total_quantity',
        'user_longitude','user_latitude','user2donor','user2recipient',
        'donor_exp','recipient_exp','user_exp',
        'avg_past_user_rating','avg_past_recipient_rating','avg_past_donor_rating',
        'published_Y','published_M','published_D','published_H'
    ]

    discrete_feat = []
    onehot_prefix = [
        'recipient_household_size','food_restriction_id','food_type_id',
        'nonprofit_category_id','population_type_id','food_id','vehicle_type'
    ]
    for f in columns:
        f_remove_suffix = "_".join(f.split("_")[:-1])
        if f_remove_suffix in onehot_prefix:
            discrete_feat.append(f)

    discrete_feat.extend(['user_hasphone','donor_hasphone','recipient_hasphone','is_recurrence'])
    return continuous_feat, discrete_feat

def filter_comment(data, features_tmp):
    features = features_tmp.copy()
    save_dir = '.'
    if not os.path.exists(f'{save_dir}comment_all.csv'):
        for feat in ['user_hasphone','donor_hasphone','recipient_hasphone','is_recurrence']:
            if feat in features:
                features.remove(feat)
        data = data[features + ['published_at','rescue_id','volunteer_comment']+
                    ['user_phone','donor_phone','recipient_phone','recurrence_id']]
        data['user_hasphone'] = data['user_phone'].isnull()
        data['donor_hasphone'] = data['donor_phone'].isnull()
        data['recipient_hasphone'] = data['recipient_phone'].isnull()
        data['is_recurrence'] = data['recurrence_id'].isnull()

        del data['user_phone']
        del data['donor_phone']
        del data['recipient_phone']
        del data['recurrence_id']

        data = data.dropna(subset=['volunteer_comment'])
        data[data.select_dtypes(include=['number']).columns] = \
            data.select_dtypes(include=['number']).fillna(data.median(numeric_only=True))
        data.to_csv(f'{save_dir}comment_all.csv', index=False)
    else:
        data = pd.read_csv(f'{save_dir}comment_all.csv')
    return data

def prepare_pseudo(split, load_df_all: bool):
    if not load_df_all:
        df_all = None
    else:
        df_all = pd.read_csv('./all_rescues_info-noorg.csv')
        save_dir = '.'
        with open(f"{save_dir}all_rescues_info-noorg_colname.pkl","wb") as fp:
            pickle.dump(list(df_all.columns), fp)

    df_labeled_train = pd.read_csv(f'train_{split}.csv')
    df_labeled_test  = pd.read_csv(f'test_{split}.csv')
    df_labeled_all   = pd.concat([df_labeled_train, df_labeled_test], axis=0)
    return df_all, df_labeled_all

def assign_pseudo_label(df_all: pd.DataFrame, df_labeled_all: pd.DataFrame,
                        split, ckpt="", plan="scratch", label_type="soft"):
    save_dir = '.'
    csv_file_path = f'{save_dir}all_rescues_pseudo_{split}_{plan}_{label_type}.csv'

    if os.path.exists(csv_file_path):
        df_comment = pd.read_csv(csv_file_path)
        return df_comment, None
    else:
        cont_feat, disc_feat = get_base_features(df_all)
        df_comment = filter_comment(df_all, cont_feat+disc_feat)
        text_list = list(df_comment['volunteer_comment'])

        pseudo_labels_file_path = f'{save_dir}pseudo_labels_{split}_{plan}_{label_type}.pkl'
        if os.path.exists(pseudo_labels_file_path):
            with open(pseudo_labels_file_path,"rb") as fp:
                pseudo_labels = pickle.load(fp)
        else:
            tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':96}
            classifier = pipeline(
                task="text-classification",
                model=ckpt,
                tokenizer=AutoTokenizer.from_pretrained(ckpt)
            )
            clean_text_list = [str(x) if x is not None else "" for x in text_list]
            pseudo_labels = []
            for out in classifier(clean_text_list, **tokenizer_kwargs):
                if label_type=="hard":
                    l_num = 0 if out['label']=='LABEL_0' else 1
                else: # label_type=="soft"
                    l_num = out['score'] if out['label']=='LABEL_1' else (1 - out['score'])
                pseudo_labels.append(l_num)

            with open(pseudo_labels_file_path,'wb') as handle:
                pickle.dump(pseudo_labels, handle)
            print(f"Pseudo labels saved to {pseudo_labels_file_path}")

        # Fill or slice the pseudo_labels to match rows
        if len(pseudo_labels)>len(df_comment):
            pseudo_labels = pseudo_labels[:len(df_comment)]
        elif len(pseudo_labels)<len(df_comment):
            pseudo_labels += [None]*(len(df_comment)-len(pseudo_labels))

        df_comment['difficulty'] = pseudo_labels

        # Overwrite the known labeled data
        for r in range(df_labeled_all.shape[0]):
            rid = df_labeled_all.iloc[r]['rescue_id']
            known_lab = df_labeled_all.iloc[r]['difficulty']
            match_idx = df_comment.index[df_comment['rescue_id']==rid]
            if not match_idx.empty:
                row_idx = match_idx[0]
                df_comment.at[row_idx, 'difficulty'] = known_lab

        df_comment.to_csv(csv_file_path,index=False)
        print(f"CSV file saved to {csv_file_path}")
        return df_comment, (cont_feat, disc_feat)

def split_tabular(split, data, features, split_type):
    """
    Splits the data into train_dataset, val_dataset, test_dataset.
    We rely on test_annot (derivative.test_annot).
    Then we do the same for train_annot.
    Finally create real train/val/test sets for the tabular step.
    """
    query = "SELECT * FROM derivative.test_annot;"
    with engine.connect() as connection:
        annot_test = pd.read_sql(query, connection)
    annot_test_id = list(annot_test['rescue_id'])

    data['published_at'] = pd.to_datetime(data['published_at'], errors='coerce')
    test_dataset = data[data['rescue_id'].isin(annot_test_id)]

    annot_easy_id = list(annot_test[annot_test['label']==-1]['rescue_id'])
    annot_hard_id = list(annot_test[annot_test['label']==1]['rescue_id'])

    if split=='easy':
        test_dataset.loc[test_dataset['rescue_id'].isin(annot_easy_id),'difficulty'] = 1
        test_dataset.loc[~test_dataset['rescue_id'].isin(annot_easy_id),'difficulty'] = 0
    else: # 'hard'
        test_dataset.loc[test_dataset['rescue_id'].isin(annot_hard_id),'difficulty'] = 1
        test_dataset.loc[~test_dataset['rescue_id'].isin(annot_hard_id),'difficulty'] = 0

    train_val_dataset = data[~data['rescue_id'].isin(annot_test_id)]

    query_train = "SET search_path TO derivative; SELECT * FROM train_annot;"
    with engine.connect() as con:
        annot_train = pd.read_sql(query_train, con)
    annot_train_easy_id = list(annot_train[annot_train['label']==-1]['rescue_id'])
    annot_train_hard_id = list(annot_train[annot_train['label']==1]['rescue_id'])
    annot_train_nan_id  = list(annot_train[annot_train['label'].isnull()]['rescue_id'])

    if split=='easy':
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_easy_id),'difficulty'] = 1
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_hard_id),'difficulty'] = 0
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_nan_id),'difficulty'] = 0
    else: # 'hard'
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_hard_id),'difficulty'] = 1
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_easy_id),'difficulty'] = 0
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_nan_id),'difficulty'] = 0

    train_val_split = datetime(2022,3,1,0,0,0)
    if split_type=='time':
        train_dataset = train_val_dataset[train_val_dataset['published_at']<train_val_split]
        val_dataset   = train_val_dataset[train_val_dataset['published_at']>=train_val_split]
    else:
        # raw
        annot_train_id = list(annot_train['rescue_id'])
        train_val_dataset = train_val_dataset[train_val_dataset['rescue_id'].isin(annot_train_id)]
        train_dataset = train_val_dataset[train_val_dataset['published_at']<train_val_split]
        val_dataset   = train_val_dataset[train_val_dataset['published_at']>=train_val_split]

    train_dataset = train_dataset[features+['difficulty','rescue_id']]
    val_dataset   = val_dataset[features+['difficulty','rescue_id']]
    test_dataset  = test_dataset[features+['difficulty','rescue_id']]

    return train_dataset, val_dataset, test_dataset

def train_eval_tabular(train_dataset,val_dataset,test_dataset,
                       continuous_feat, discrete_feat,
                       ckpt_folder, split_type, seed):
    os.makedirs(ckpt_folder, exist_ok=True)
    feats = continuous_feat+discrete_feat
    imp = SimpleImputer(strategy='mean')

    X_tr = imp.fit_transform(train_dataset[feats])
    X_val= imp.transform(val_dataset[feats])
    y_tr = (train_dataset['difficulty'].values >= 0.5).astype(int)
    y_val= (val_dataset['difficulty'].values >= 0.5).astype(int)

    ro = RandomOverSampler(random_state=seed)
    X_tr, y_tr = ro.fit_resample(X_tr, y_tr)

    pos_w = (len(y_tr)-y_tr.sum()) / max(1,y_tr.sum())

    models = {
        'lgb': lgb.LGBMClassifier(random_state=seed, scale_pos_weight=pos_w),
        'rf' : RandomForestClassifier(random_state=seed, class_weight='balanced'),
        'lr' : LogisticRegression(random_state=seed, class_weight='balanced', max_iter=1000),
        'svm': SVC(random_state=seed, probability=True, class_weight='balanced'),
        'mlp': MLPClassifier(random_state=seed, max_iter=500),
        'knn': KNeighborsClassifier()
    }

    # Slightly tune the main ones
    if 'lgb' in models:
        models['lgb'].set_params(
            num_leaves=32,
            min_data_in_leaf=10,
            learning_rate=0.03,
            n_estimators=300,
            max_depth=6
        )
    if 'rf' in models:
        models['rf'].set_params(
            n_estimators=300,
            max_depth=20
        )

    best_thr, res_val, res_test = {}, {}, {}

    # 1) Find best threshold on VAL for each model
    for n,m in models.items():
        m.fit(X_tr,y_tr)
        p_val = m.predict_proba(X_val)[:,1]
        fpr,tpr,thr = roc_curve(y_val, p_val)
        f1s = [f1_score(y_val, (p_val>=t)) for t in thr]
        idx_best = int(np.argmax(f1s))
        best_thr[n] = thr[idx_best]
        res_val[n] = {
            'f1': f1s[idx_best],
            'roc': roc_auc_score(y_val, p_val)
        }

    # 2) Combine train+val => final
    X_tv = imp.fit_transform(pd.concat([train_dataset[feats], val_dataset[feats]]))
    y_tv = (pd.concat([train_dataset['difficulty'], val_dataset['difficulty']]).values>=0.5).astype(int)
    X_tv, y_tv = ro.fit_resample(X_tv, y_tv)

    X_te = imp.transform(test_dataset[feats])
    y_te = (test_dataset['difficulty'].values>=0.5).astype(int)

    res_test={}
    for n,m in models.items():
        m.fit(X_tv, y_tv)
        p = m.predict_proba(X_te)[:,1]
        T = best_thr[n]
        preds = (p>=T)
        prec, rec, _ = precision_recall_curve(y_te, p)
        # compute metrics
        res_test[n] = {
            'acc': accuracy_score(y_te, preds),
            'f1' : f1_score(y_te, preds),
            'roc': roc_auc_score(y_te, p) if len(np.unique(y_te))>1 else float('nan'),
            'pr' : auc(rec, prec)
        }

    with open(f"{ckpt_folder}/metrics-{split_type}.json","w") as fp:
        json.dump({'val':res_val,'test':res_test}, fp, indent=2)

    return {'val':res_val, 'test':res_test}, None

##################################################
# MAIN: Baseline2 from Paper
##################################################
if __name__=="__main__":
    from collections import defaultdict

    seeds = [42,43,44,45,46,47,48,49,50,51]
    model_names = ['lgb','rf','lr','svm','mlp','knn']
    split_types= ["raw"]
    all_runs = {
        s:{t:{m:{'acc':[],'f1':[],'roc':[],'pr':[]} for m in model_names}
           for t in split_types}
        for s in ["easy","hard"]
    }

    for split in ["easy","hard"]:
        # Re-create the CSV from DB, after you have run build_dataset to fix labels
        if not os.path.exists(f"train_{split}.csv"):
            create_difficulty_csv(split)

        for seed in seeds:
            print(f"\n=== Baseline2 {split} seed={seed} ===")

            ds = create_difficulty_dataset(split)
            ds_proc = preprocess_difficulty_dataset(ds)
            hf_train= ds_proc["train"]
            hf_eval = ds_proc["test"]

            bert_ckpt = train_eval_text_difficulty(hf_train,hf_eval,
                                                   split_name=f"{split}-seed{seed}",
                                                   seed=seed)

            df_all = pd.read_csv("all_rescues_info-noorg.csv")
            df_tr  = pd.read_csv(f"train_{split}.csv")
            df_te  = pd.read_csv(f"test_{split}.csv")
            df_lbl = pd.concat([df_tr, df_te], axis=0)

            # Generate pseudo-labels
            df_comment, feats = assign_pseudo_label(df_all, df_lbl, split,
                                                    ckpt=bert_ckpt,
                                                    plan="baseline2",
                                                    label_type="soft")

            if feats is None:
                from baseline1 import get_base_features as gbf
                cont, disc = get_base_features(df_comment)
            else:
                cont, disc = feats

            for stype in split_types:
                train_df, val_df, test_df = split_tabular(split, df_comment,
                                                         cont+disc, stype)
                ckpt_dir = f"./ckpt-baseline2-{split}-{stype}-seed{seed}"
                results_dict, _ = train_eval_tabular(
                    train_df, val_df, test_df,
                    cont, disc,
                    ckpt_dir, stype,
                    seed
                )
                for m in model_names:
                    all_runs[split][stype][m]['acc'].append(results_dict['test'][m]['acc'])
                    all_runs[split][stype][m]['f1'].append(results_dict['test'][m]['f1'])
                    all_runs[split][stype][m]['roc'].append(results_dict['test'][m]['roc'])
                    all_runs[split][stype][m]['pr'].append(results_dict['test'][m]['pr'])

        print(f"\n=== Baseline2 Aggregated for {split} ===")
        for stype in split_types:
            print(f"--- split_type={stype} ---")
            for m in model_names:
                arr_acc = all_runs[split][stype][m]['acc']
                arr_f1  = all_runs[split][stype][m]['f1']
                arr_roc = all_runs[split][stype][m]['roc']
                arr_pr  = all_runs[split][stype][m]['pr']
                print(f"{m} | "
                      f"Acc={np.mean(arr_acc):.3f}±{np.std(arr_acc):.3f}, "
                      f"F1={np.mean(arr_f1):.3f}±{np.std(arr_f1):.3f}, "
                      f"AUC={np.mean(arr_roc):.3f}±{np.std(arr_roc):.3f}, "
                      f"PR={np.mean(arr_pr):.3f}±{np.std(arr_pr):.3f}")
            print()

    print("\nDone. This is the Baseline2 approach from the paper, with label-fixes done in build_dataset.ipynb.\n")

