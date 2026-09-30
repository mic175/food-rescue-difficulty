# refer to build_dataset.ipynb for base tabular models
# Step 1: use bert from text to predict rating
# Step 2: use rating-pretrained-bert from text to predict provided label
# Step 3: use 3 to assign easy/hard pseudolabel for the whole dataset
# Step 4: use geolocation to predict labels + pseudolabel

import psycopg2
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from typing import *
from tqdm import tqdm
import json
from collections import defaultdict
import copy
import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import TrainingArguments, Trainer
import evaluate

from sklearn import preprocessing
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier  # <-- Added import
from sklearn.neighbors import KNeighborsClassifier  # <-- Added import
import lightgbm as lgb
from imblearn.over_sampling import SMOTE  # For class balancing
# Load database configuration from a YAML file
with open('database.yaml') as f:
    db_config = yaml.safe_load(f)

# Create the PostgreSQL URL using SQLAlchemy's URL object
db_url = f"postgresql://{db_config['user']}:{db_config['pass']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"

# Create an engine to connect to the PostgreSQL database
engine = create_engine(db_url)

# numerically stable sigmoid
def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])
    return result



def create_rating_csv(seed):
    # Fetch data from PostgreSQL using SQLAlchemy
    query = """
    SELECT id, volunteer_comment, volunteer_rating
    FROM public.rescues
    WHERE volunteer_comment IS NOT NULL AND volunteer_rating IS NOT NULL;
    """
    with engine.connect() as con:
        df = pd.read_sql(query, con)
    
    print("[create_rating_csv] Rows fetched from DB:", len(df))
    
    if df.empty:
        print("🚨 Warning: No data fetched from database. Check your database connection or query.")
        return None, None  # Return None if the DataFrame is empty
    
    df = df.dropna(axis=0,subset=['volunteer_comment','volunteer_rating'])
    df_comments = list(df['volunteer_comment'])
    df_ratings = list(pd.to_numeric(df['volunteer_rating']))
    df_id = list(pd.to_numeric(df['id']))

    new_df = pd.DataFrame()
    new_df['rescue_id'] = df_id
    new_df['rating'] = df_ratings
    new_df['text'] = df_comments

    new_df_train = new_df.sample(frac=0.8, random_state=seed)
    new_df_test = new_df.drop(new_df_train.index)

    print(f"[create_rating_csv] Train rows: {len(new_df_train)}, Test rows: {len(new_df_test)}")
    
    if new_df_train.empty or new_df_test.empty:
        print("🚨 Error: Train or test dataset is empty. Check the database query.")
        return None, None
    
    # Save the CSV files to your desired directory on the remote server
    new_df_train.to_csv('train_ratings.csv', index=False)
    new_df_test.to_csv('test_ratings.csv', index=False)#test-rating
    
    import os
    train_size = os.path.getsize('train_ratings.csv')
    test_size  = os.path.getsize('test_ratings.csv')
    print(f"[create_rating_csv] Wrote train_ratings.csv (size={train_size} bytes)")
    print(f"[create_rating_csv] Wrote test_ratings.csv (size={test_size} bytes)")
    
    return new_df_train, new_df_test

def create_rating_dataset():
    # Update the file paths to load the CSVs from your desired directory
    data_files = {
        'train': 'train_ratings.csv',
        'test': 'test_ratings.csv'#test-rating
    }
    train_test_ratings = load_dataset("csv", data_files=data_files)
    return train_test_ratings


def create_difficulty_csv(split, root='.'):
    # Load labeled data from PostgreSQL
    with engine.connect() as con:
        df_train_labeled = pd.read_sql_table('train_annot', con=con, schema='derivative')
        df_test_labeled = pd.read_sql_table('test_annot', con=con, schema='derivative')

    if split == 'easy':
        df_train = copy.deepcopy(df_train_labeled)
        df_train.loc[(df_train['label'] != -1), 'label'] = 0
        df_train.loc[(df_train['label'] == -1), 'label'] = 1
        df_test = copy.deepcopy(df_test_labeled)
        df_test.loc[(df_test['label'] != -1), 'label'] = 0
        df_test.loc[(df_test['label'] == -1), 'label'] = 1
    else:
        df_train = copy.deepcopy(df_train_labeled)
        df_train.loc[(df_train['label'] != 1), 'label'] = 0
        df_train.loc[(df_train['label'] == 1), 'label'] = 1
        df_test = copy.deepcopy(df_test_labeled)
        df_test.loc[(df_test['label'] != 1), 'label'] = 0
        df_test.loc[(df_test['label'] == 1), 'label'] = 1

    df_train = df_train.rename(columns={'label': 'difficulty'})
    df_train = df_train[['rescue_id', 'rating', 'text', 'difficulty']]
    df_test = df_test.rename(columns={'label': 'difficulty'})
    df_test = df_test[['rescue_id', 'rating', 'text', 'difficulty']]

    df_train.to_csv(root + f'/train_{split}.csv', index=False)
    df_test.to_csv(root + f'/test_{split}.csv', index=False)
    return


def create_difficulty_dataset(split, root='.'):
    print(f'Split value: {split}')

    data_files = {'train': root + f'/train_{split}.csv',
                  'test': root + f'/test_{split}.csv'}
    
    # Load the dataset
    train_test_difficulty = load_dataset("csv", data_files=data_files)
    
    # Convert the dataset to a pandas DataFrame and combine train and test data for saving
    train_df = train_test_difficulty['train'].to_pandas()
    test_df = train_test_difficulty['test'].to_pandas()
    
    return train_test_difficulty


def preprocess_bert_dataset(dataset, label_col):
    def tokenize_function(examples):
        # Ensure the "text" field contains valid strings, replace non-string entries if needed
        texts = examples["text"]
        
        # Convert all non-string entries to empty strings to avoid errors
        if isinstance(texts, list):
            texts = [str(text) if isinstance(text, (str, bytes)) else "" for text in texts]
        elif not isinstance(texts, str):
            texts = ""  # Convert non-string to empty string
        
        return tokenizer(texts, padding="max_length", truncation=True)
    
    def reset_rating(examples):
        return {"labels": list(np.array(examples["rating"], dtype=int) - 1)}
    
    def reset_difficulty(examples):
        res = np.array(examples["difficulty"], dtype=int)
        return {"labels": list(res)}
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    if label_col == 'rating':
        relabeled_datasets = tokenized_datasets.map(reset_rating, batched=True)
    elif label_col == 'difficulty':
        relabeled_datasets = tokenized_datasets.map(reset_difficulty, batched=True)
    
    return relabeled_datasets


def train_eval_text(num_labels, train_dataset, eval_dataset, split, seed, ckpt=None):
    base_model_name = "distilbert-base-cased"

    if ckpt is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels
        )
        #output_dir = f"./ckpt-{split}-fast"
        output_dir = f"./ckpt-{split}-seed{seed}-fast"  # name includes seed
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            num_train_epochs=3.0,
            logging_steps=10000,
            save_steps=10000,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            seed=seed  # set the seed
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # needed if 4-class → 2-class
        )
        output_dir = f"./ckpt-{split}-finetune"
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            num_train_epochs=3.0,
            logging_steps=10000,
            save_steps=10000,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created checkpoint directory: {output_dir}")
    else:
        print(f"Checkpoint directory already exists: {output_dir}")

    # STEP 5: Instead of forcing a fixed sample size, use the full dataset.
    # (Alternatively, if you wish to use a subset, choose the minimum between desired and available)
    print("Training dataset size:", len(train_dataset))
    print("Evaluation dataset size:", len(eval_dataset))

    # (Optional) Print the label distributions for debugging
    # Here we assume the mapped field is "labels" (set in your preprocess_bert_dataset functions)
    train_labels = train_dataset["labels"]
    eval_labels = eval_dataset["labels"]
    print("Training label distribution:", np.unique(train_labels, return_counts=True))
    print("Evaluation label distribution:", np.unique(eval_labels, return_counts=True))
    # Because your dataset might be smaller than 2000/1000:
    train_size = min(len(train_dataset), 2000)
    eval_size  = min(len(eval_dataset), 1000)
    train_dataset = train_dataset.select(range(train_size))
    eval_dataset  = eval_dataset.select(range(eval_size))

    


    metric = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if logits.ndim == 2 and logits.shape[1] == 2:
            # 2-class => softmax => use second column
            probs = softmax(logits, axis=1)
            roc_val = roc_auc_score(labels, probs[:,1])
        elif logits.ndim == 2 and logits.shape[1] == 1:
            # single-logit binary
            from scipy.special import expit as sigmoid
            p = sigmoid(logits.ravel())
            roc_val = roc_auc_score(labels, p)
        else:
            # multi-class e.g. 4
            probs = softmax(logits, axis=1)
            roc_val = roc_auc_score(labels, probs, multi_class='ovr')
        return {"roc_auc": roc_val}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"===== Starting BERT training for split={split}, num_labels={num_labels} =====")
    trainer.train()
    
    #trainer.save_model(output_dir)  # Force-save the model/config to 'output_dir'
    model.save_pretrained(output_dir, safe_serialization=False)
    print(f"[INFO] Saved HF model checkpoint to: {output_dir}")

    print("[DEBUG] Listing directory after save_pretrained:")
    print(os.listdir(output_dir))

    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    
    #final_auc = None
    if "eval_roc_auc" in eval_results:
        print(f">>>> Final AUC for BERT '{split}' with num_labels={num_labels}: "
              f"{eval_results['eval_roc_auc']:.4f}")
    else:
        print("No 'eval_roc_auc' found in eval_results. The dictionary is:", eval_results)

    return eval_results.get("eval_roc_auc", None)

def get_base_features(df_all):
    if df_all is None:
        with open("all_rescues_info-noorg_colname.pkl", "rb") as fp:
            columns = pickle.load(fp)  # load a list of column names

    else:
        columns = list(df_all.columns)
        for feat in ['user_phone','donor_phone','recipient_phone','recurrence_id']:
            if feat in columns:
                columns.remove(feat)
    
    continuous_feat = ['HourlyDewPointTemperature','HourlyDryBulbTemperature','HourlyPrecipitation',
                   'HourlyRelativeHumidity','HourlyStationPressure','HourlyVisibility','HourlyWetBulbTemperature',
                    'HourlyWindSpeed',
                    'recipient_longitude','recipient_latitude',
                    'donor_longitude','donor_latitude','total_quantity',
                    'user_longitude','user_latitude','user2donor','user2recipient',
                    'donor_exp', 'recipient_exp','user_exp',
                    'avg_past_user_rating','avg_past_recipient_rating','avg_past_donor_rating',
                    'published_Y','published_M','published_D','published_H']
    discrete_feat = []

    onehot_prefix = ['recipient_household_size',
                    'food_restriction_id','food_type_id','nonprofit_category_id',
                    'population_type_id','food_id','vehicle_type']
   
    for f in list(columns):
        f_remove_suffix = "_".join(f.split("_")[:-1])
        if f_remove_suffix in onehot_prefix:
            discrete_feat.append(f)

    discrete_feat.extend(['user_hasphone','donor_hasphone','recipient_hasphone','is_recurrence'])
    return continuous_feat, discrete_feat

def filter_comment(data, features_tmp):
    
    features = features_tmp.copy()
    
    save_dir = '.'
    
    if not (os.path.exists(f'{save_dir}comment_all.csv')):
        for feat in ['user_hasphone', 'donor_hasphone', 'recipient_hasphone', 'is_recurrence']:
            features.remove(feat)
        
        data = data[features + ['published_at', 'rescue_id', 'volunteer_comment'] + 
                    ['user_phone', 'donor_phone', 'recipient_phone', 'recurrence_id']]
        data['user_hasphone'] = data['user_phone'].isnull()
        data['donor_hasphone'] = data['donor_phone'].isnull()
        data['recipient_hasphone'] = data['recipient_phone'].isnull()
        data['is_recurrence'] = data['recurrence_id'].isnull()
        
        del data['user_phone']
        del data['donor_phone']
        del data['recipient_phone']
        del data['recurrence_id']
        
        data = data.dropna(subset=['volunteer_comment'])  # We can't use rows without comment; no pseudo labels.
        data[data.select_dtypes(include=['number']).columns] = data.select_dtypes(include=['number']).fillna(data.median(numeric_only=True))
        data.to_csv(f'{save_dir}comment_all.csv', index=False)
    else:
        data = pd.read_csv(f'{save_dir}comment_all.csv')
    
    return data


def prepare_pseudo(split, load_df_all: bool):
    # If load_df_all is False, set df_all to None to save memory
    if not load_df_all:
        df_all = None
    else:
        # Load all_rescues_info data from the database instead of CSV
        df_all = pd.read_csv('./all_rescues_info-noorg.csv')

        # Save column names to a pickle file (optional)
        save_dir = '.'

        with open(f"{save_dir}all_rescues_info-noorg_colname.pkl", "wb") as fp:
            pickle.dump(list(df_all.columns), fp)

 
    df_labeled_train = pd.read_csv(f'train_{split}.csv')
    df_labeled_test = pd.read_csv(f'test_{split}.csv')
    df_labeled_all = pd.concat([df_labeled_train, df_labeled_test], axis=0)
    return df_all, df_labeled_all



def assign_pseudo_label(df_all : pd.DataFrame, df_labeled_all : pd.DataFrame, split,
                        ckpt="", plan="scratch", label_type="soft"):
    save_dir = '.'
    
    # Check if the CSV file already exists
    csv_file_path = f'{save_dir}/all_rescues_pseudo_{split}_{plan}_{label_type}.csv'  # MOD‑2


    #_pseudo_{split}_{plan}_{label_type}

    if os.path.exists(csv_file_path):
        features = get_base_features(df_all)
        df_comment = pd.read_csv(f'{save_dir}all_rescues_pseudo_{split}_{plan}_{label_type}.csv')
    else:
        # Get the continuous and discrete features
        continuous_feat, discrete_feat = get_base_features(df_all)
        df_comment = filter_comment(df_all, continuous_feat + discrete_feat)
        text_list = list(df_comment['volunteer_comment'])
        
        features = (continuous_feat, discrete_feat)

        def iter_text():
            for i in tqdm(range(len(text_list))):
                yield text_list[i]
        
        # Check if pseudo-labels already exist
        #
        pseudo_labels_file_path = f'{save_dir}pseudo_labels_{split}_{plan}_{label_type}.pkl'
        if os.path.exists(pseudo_labels_file_path):
            with open(pseudo_labels_file_path, "rb") as fp:
                pseudo_labels = pickle.load(fp)
        else:
            # Generate new pseudo labels using the model
            tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 128}
            
            classifier = pipeline(task="text-classification", model="bert-base-cased", tokenizer=AutoTokenizer.from_pretrained("bert-base-cased"))

            #classifier = pipeline(task="text-classification", model=ckpt, tokenizer=AutoTokenizer.from_pretrained("bert-base-cased"))
            #changed
            
            pseudo_labels = []
            for out in classifier(iter_text(), **tokenizer_kwargs):
                if label_type == "hard":
                    l_num = 0 if out['label'] == 'LABEL_0' else 1
                elif label_type == "soft":
                    l_num = 1 - out['score'] if out['label'] == 'LABEL_0' else out['score']
                pseudo_labels.append(l_num)
            
            # Save the pseudo labels to a pickle file
            with open(pseudo_labels_file_path, 'wb') as handle:
                pickle.dump(pseudo_labels, handle)
            print(f"Pseudo labels saved successfully to {pseudo_labels_file_path}")
        
        pseudo_series = pd.Series(pseudo_labels)
        print("Pseudo-label distribution:")
        print(pseudo_series.describe())
        if 'difficulty' in df_labeled_all.columns:
            print("Original labeled difficulty distribution:")
            print(df_labeled_all['difficulty'].value_counts())
            
        # Add the pseudo labels to the dataframe
        if len(pseudo_labels) > len(df_comment):
            pseudo_labels = pseudo_labels[:len(df_comment)]
        elif len(pseudo_labels) < len(df_comment):
            print("Warning: Fewer pseudo labels than rows in df_comment.")
            pseudo_labels = pseudo_labels + [None] * (len(df_comment) - len(pseudo_labels))
        #changed
        df_comment['difficulty'] = pseudo_labels
        
        print("Columns in df_labeled_all:", df_labeled_all.columns)
        # Replace labeled predictions with assigned annotations
        for r in range(df_labeled_all.shape[0]):
            res_id = df_labeled_all.iloc[r]['rescue_id']
            annot = df_labeled_all.iloc[r]['difficulty']
            print(df_comment.columns)

            matching_indices = df_comment.index[df_comment['rescue_id'] == res_id]
    
            if not matching_indices.empty:  # If there is a match
                row_idx = matching_indices[0]
                df_comment.at[row_idx, 'difficulty'] = annot
            else:
                print(f"No match found for rescue_id {res_id}")
            #changed
        
        # Save the updated DataFrame to CSV
        df_comment.to_csv(csv_file_path, index=False)
        print(f"CSV file saved successfully to {csv_file_path}")
    return df_comment, features

def split_tabular(split, data, features, split_type, seed):
    # take out labeled test set for final step evaluation
    root = '.'

    # annot_test = pd.read_excel(root+'/test_annot.xlsx',sheet_name='Sheet1')
    query = "SELECT * FROM derivative.test_annot;"
    with engine.connect() as connection:
        annot_test = pd.read_sql(query, connection)
    annot_test_id = list(annot_test['rescue_id'])

    from sklearn.model_selection import train_test_split
    
    # Prepare features and labels
    X = data[features]
    y = data['difficulty'].apply(lambda x: 1 if x > 0.5 else 0)  # Binarize
    
    # Stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=seed
    )
    
    # Reconstruct DataFrames
    train_dataset = data.loc[X_train.index]
    test_dataset = data.loc[X_test.index]
    
    # ====== PAPER'S VALIDATION STRATEGY ======
    # Further split train into train/val (75% train, 25% val)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=seed
    )
    
    train_sub_dataset = train_dataset.loc[X_train_sub.index]
    val_dataset = train_dataset.loc[X_val.index]
    
    print(f"\n=== Stratified Split ({split}) ===")
    print(f"Train size: {len(train_sub_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_sub_dataset, val_dataset, test_dataset
    # train_test_split
    #changed
    '''
    train_val_split = datetime(2022, 3, 1, 0, 0, 0)
    data['published_at'] = pd.to_datetime(data['published_at'])

    test_dataset = data[data['rescue_id'].isin(annot_test_id)]
    annot_easy_id = list(annot_test[annot_test['label'] == -1]['rescue_id'])
    annot_hard_id = list(annot_test[annot_test['label'] == 1]['rescue_id'])
    if split == 'easy':
        test_dataset.loc[test_dataset['rescue_id'].isin(annot_easy_id),'difficulty'] = 1
        test_dataset.loc[~test_dataset['rescue_id'].isin(annot_easy_id),'difficulty'] = 0
    elif split == 'hard':
        test_dataset.loc[test_dataset['rescue_id'].isin(annot_hard_id),'difficulty'] = 1
        test_dataset.loc[~test_dataset['rescue_id'].isin(annot_hard_id),'difficulty'] = 0
    
    train_val_dataset = data[~data['rescue_id'].isin(annot_test_id)]

    print("Number of rows and columns in train_cal_dataset:", train_val_dataset.shape)
    #(7081, 212) rows and columns
    # make sure annotated train labels are correct
    # annot_train = pd.read_excel(root+'/train_annot_labeled.xlsx',sheet_name='train')
    query = "SET search_path TO derivative; SELECT * FROM train_annot;"
    with engine.connect() as con:
        annot_train = pd.read_sql(query, con)
    annot_train_easy_id = list(annot_train[annot_train['label'] == -1]['rescue_id'])
    annot_train_nan_id = list(annot_train[annot_train['label'].isnull()]['rescue_id'])
    annot_train_hard_id = list(annot_train[annot_train['label'] == 1]['rescue_id'])
    if split == 'easy':
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_easy_id),'difficulty'] = 1
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_hard_id),'difficulty'] = 0
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_nan_id),'difficulty'] = 0
    elif split == 'hard':
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_hard_id),'difficulty'] = 1
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_easy_id),'difficulty'] = 0
        train_val_dataset.loc[train_val_dataset['rescue_id'].isin(annot_train_nan_id),'difficulty'] = 0

    if split_type == 'time':
        train_dataset = train_val_dataset[train_val_dataset['published_at'] < train_val_split]
        val_dataset = train_val_dataset[train_val_dataset['published_at'] >= train_val_split]
    elif split_type == 'raw':
        annot_train_id = list(annot_train['rescue_id'])
        # don't use pseudo labels at all
        train_val_dataset = train_val_dataset[train_val_dataset['rescue_id'].isin(annot_train_id)]
        train_dataset = train_val_dataset[train_val_dataset['published_at'] < train_val_split]
        val_dataset = train_val_dataset[train_val_dataset['published_at'] >= train_val_split]

    print("Number of rows and columns in train_dataset[features]:", train_dataset[features].shape)
    
    train_dataset = train_dataset[features + ['difficulty'] + ['rescue_id']] 
    # print("train_dataset:", train_dataset)

    val_dataset = val_dataset[features + ['difficulty'] + ['rescue_id']]
    test_dataset = test_dataset[features + ['difficulty'] + ['rescue_id']]
    
        # After splitting, add these lines:
    print("Train dataset shape:", train_dataset.shape)
    print("Validation dataset shape:", val_dataset.shape)
    print("Test dataset shape:", test_dataset.shape)

    return train_dataset, val_dataset, test_dataset
    '''

def train_eval_tabular(train_dataset, val_dataset, test_dataset,
                       continuous_feat, discrete_feat,
                       ckpt_folder, split_type, seed):
        # --- MOD‑4: make sure output dirs exist
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(os.path.join(ckpt_folder, "feature"), exist_ok=True)
    # ====== 1. PREPROCESSING ======
    features = continuous_feat + discrete_feat
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(train_dataset[features])
    y_train = train_dataset['difficulty'].apply(lambda x: 1 if x > 0.5 else 0).values
    
    X_val = imputer.transform(val_dataset[features])
    y_val = val_dataset['difficulty'].apply(lambda x: 1 if x > 0.5 else 0).values
    
    X_test = imputer.transform(test_dataset[features])
    y_test = test_dataset['difficulty'].apply(lambda x: 1 if x > 0.5 else 0).values
    
    # ====== 2. CLASS BALANCING ======
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=seed)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # ====== 3. MODEL DEFINITION (PAPER'S HYPERPARAMS) ======
    models = {
    'lgb': lgb.LGBMClassifier(
        random_state=seed,
        class_weight='balanced',
        num_leaves=31,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        verbosity=-1,
        n_jobs=-1  # Enable parallel processing
    ),
    'rf': RandomForestClassifier(
        random_state=seed,
        class_weight='balanced',
        n_estimators=300,
        max_depth=20,
        n_jobs=-1  # Enable parallel processing
    ),
    'lr': LogisticRegression(
        class_weight='balanced',
        max_iter=200,  # Reduced iterations with better solver
        solver='saga',  # Faster solver for large datasets
        tol=1e-3,  # Looser convergence tolerance
        n_jobs=-1,  # Enable parallel processing
        random_state=seed
    ),
    'svm': SVC(
        class_weight='balanced',
        probability=True,
        kernel='linear',  # Faster than default RBF kernel
        tol=1e-3,  # Looser convergence tolerance
        max_iter=1000,  # Prevent infinite loops
        random_state=seed
    ),
    'mlp': MLPClassifier(
        random_state=seed,
        hidden_layer_sizes=(50,),  # Reduced network size
        max_iter=200,  # Reduced maximum iterations
        early_stopping=True,  # Enable early stopping
        validation_fraction=0.1,  # Validation set for early stopping
        batch_size=256,  # Larger batch size
        tol=1e-3  # Looser convergence tolerance
    ),
    'knn': KNeighborsClassifier(
        n_neighbors=5,
        algorithm='kd_tree',  # Faster search algorithm
        leaf_size=30,  # Optimized leaf size
        n_jobs=-1  # Enable parallel processing
    )
}

    
    # ====== 4. TRAINING & EVALUATION ======
    results = {'val': {}, 'test': {}}
    
    for model_name, model in models.items():
        # Train
        model.fit(X_train_bal, y_train_bal)
        
         # Validate
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        precision_val, recall_val, _ = precision_recall_curve(y_val, val_probs)
        val_pr_auc = auc(recall_val, precision_val)

        # Test
        test_probs = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)
        precision_test, recall_test, _ = precision_recall_curve(y_test, test_probs)
        test_pr_auc = auc(recall_test, precision_test)
        test_acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))
        test_f1 = f1_score(y_test, (test_probs > 0.5).astype(int))
        
        # Save results
        results['val'][model_name] = {
            'roc_auc': val_auc,
            'roc': val_auc,  # Explicitly included for aggregation
            'pr': val_pr_auc  # Explicitly included for aggregation
        }
        results['test'][model_name] = {
            'roc_auc': test_auc,
            'roc': test_auc,  # Explicitly included for aggregation
            'pr': test_pr_auc,  # Explicitly included for aggregation
            'acc': test_acc,
            'f1': test_f1
        }
        
        print(f"\n{model_name.upper()} | {split_type} | Seed {seed}")
        print(f"Val AUC: {val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | PR: {test_pr_auc:.4f}")
    
    # ====== 5. SAVE METRICS ======
    


    with open(f"{ckpt_folder}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results
'''
def train_eval_tabular(train_dataset : pd.DataFrame, 
                       val_dataset : pd.DataFrame, 
                       test_dataset : pd.DataFrame,
                       continuous_feat : List[str], 
                       discrete_feat : List[str],
                       ckpt_folder : str, 
                       split_type : str,
                       seed=42):
    
    # 1) Make sure the main checkpoint directory exists
    os.makedirs(ckpt_folder, exist_ok=True)
    # 2) Make sure the 'feature' subdirectory also exists
    os.makedirs(os.path.join(ckpt_folder, "feature"), exist_ok=True)
    
    features = continuous_feat + discrete_feat

    train_val = pd.concat([train_dataset,val_dataset],axis=0)
    imputer = SimpleImputer(strategy='mean')
    #changed

    # train, val
    thres = 0.5
    print("train_dataset[features]:", train_dataset[features])
    X_train = np.array(train_dataset[features])
    X_train = imputer.fit_transform(X_train)
    y_train = np.array(train_dataset['difficulty'])
    y_train = np.clip(y_train, 1e-8, 1 - 1e-8) # numerical stability
    inv_sig_y_train = np.log(y_train / (1 - y_train)) # transform to log-odds-ratio space
    # ignore the following line when my label is soft
    # print("Train Label distribution:",np.unique(y_train, return_counts=True))
    X_val = np.array(val_dataset[features])
    X_val = imputer.transform(X_val)
    y_val = np.array(val_dataset['difficulty'])
    y_val = np.array([1 if i > thres else 0 for i in y_val], dtype=np.uint8)
    print("Val Label distribution:",np.unique(y_val, return_counts=True))

    # throw into logistic regression and lgbm
    models = {
        'lgb': lgb.LGBMClassifier(
            random_state=seed,
            class_weight='balanced',
            num_leaves=31,       # Hyperparameters from the paper
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200
        ),
        'rf': RandomForestClassifier(
            random_state=seed,
            class_weight='balanced',
            n_estimators=300,    # Hyperparameters from the paper
            max_depth=20
        ),
        'lr': LogisticRegression(class_weight='balanced', max_iter=1000),
        'svm': SVC(class_weight='balanced', probability=True),
        'mlp': MLPClassifier(random_state=seed),
        'knn': KNeighborsClassifier()
    }
    model_res = {split : {k:{'acc':0,'f1':0,'roc':0, 'pr':0} for k in models} for split in ['val','test']}
    val_best_threshold = {k:-1 for k in models}
    
    
    # validation
    for model_name in models:
        model = models[model_name]
        model.fit(X_train, inv_sig_y_train)
        predict_logits = model.predict(X_val)
        predict_proba = sigmoid(predict_logits)
        
        roc = roc_auc_score(y_val, predict_proba)
        _, _, thresholds = roc_curve(y_val, predict_proba)
        
        f1s = []
        for th in thresholds:
            predict_labels = [1 if x > th else 0 for x in predict_proba]
            f1 = f1_score(y_val, predict_labels)
            f1s.append(f1)
        
        best_threshold = thresholds[np.argmax(f1s)]
        val_best_threshold[model_name] = best_threshold
        predict_labels = [1 if x > best_threshold else 0 for x in predict_proba]
        acc = accuracy_score(y_val, predict_labels) 
        f1 = f1_score(y_val, predict_labels)
        
        precision, recall, _ = precision_recall_curve(y_val, predict_proba)
        pr = auc(recall, precision)
        
        model_res['val'][model_name]['acc'] = acc
        model_res['val'][model_name]['f1'] = f1
        model_res['val'][model_name]['roc'] = roc
        model_res['val'][model_name]['pr'] = pr
        print('Val:',model_name, f'acc: {acc:.3f}, f1: {f1:.3f}, roc: {roc:.3f}, pr: {pr:.3f}')


    # Final testing

    # train, val, test set dist
    X_train_val = np.array(train_val[features])
    X_train_val = imputer.fit_transform(X_train_val) #changed imputer
    y_train_val = np.array(train_val['difficulty'])
    y_train_val = np.clip(y_train_val, 1e-8, 1 - 1e-8) # numerical stability
    inv_sig_y_train_val = np.log(y_train_val / (1 - y_train_val)) # transform to log-odds-ratio space
    X_test = np.array(test_dataset[features])
    X_test = imputer.transform(X_test) #Changed imputer
    y_test = np.array(test_dataset['difficulty'],dtype=np.uint8)
    print("Test Label distribution:",np.unique(y_test, return_counts=True))

    # Test
    
    for model_name in models:
        model = models[model_name]
        model.fit(X_train_val, inv_sig_y_train_val)
        predict_logits = model.predict(X_test)
        predict_proba = sigmoid(predict_logits)
        
        roc = roc_auc_score(y_test, predict_proba)

        _, _, thresholds = roc_curve(y_test, predict_proba)
        
        f1s = []
        for th in thresholds:
            predict_labels = [1 if x > th else 0 for x in predict_proba]
            f1 = f1_score(y_test, predict_labels)
            f1s.append(f1)
        
        best_threshold = thresholds[np.argmax(f1s)]
        print(best_threshold)

        predict_labels = [1 if x > best_threshold else 0 for x in predict_proba]
        acc = accuracy_score(y_test, predict_labels) 
        f1 = f1_score(y_test, predict_labels)
        
        precision, recall, _ = precision_recall_curve(y_test, predict_proba)
        pr = auc(recall, precision)
        
        model_res['test'][model_name]['acc'] = acc
        model_res['test'][model_name]['f1'] = f1
        model_res['test'][model_name]['roc'] = roc
        model_res['test'][model_name]['pr'] = pr
        print('Test:',model_name, f'acc: {acc:.3f}, f1: {f1:.3f}, roc: {roc:.3f}, pr: {pr:.3f}')


        cm = confusion_matrix(y_test, predict_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(ckpt_folder + f'/test-{model_name}-{split_type}-cm.png')
        plt.close()
        
    for feat in tqdm(features):
        try:
            plt.clf()
            total_zero = len(test_dataset[features][y_test == 0][feat])
            total_ones = len(test_dataset[features][y_test == 1][feat])
            plt.hist(test_dataset[features][y_test == 0][feat], 100,  weights=np.ones((total_zero,)) / total_zero, alpha=0.5, label='0')
            plt.hist(test_dataset[features][y_test == 1][feat], 100, weights=np.ones((total_ones,)) / total_ones, alpha=0.5, label='1')
            plt.legend(loc='upper right')
            plt.title(feat)
            plt.savefig(ckpt_folder + f'/feature/{feat}.png')
            plt.close()
        except:
            pass
    
    with open(ckpt_folder+f'/metrics-{split_type}.json', 'w') as fp:
        json.dump(model_res, fp, indent=4)
    return model_res, (train_val, test_dataset)
'''

def fin_prediction(train_val_dataset: pd.DataFrame, test_dataset: pd.DataFrame, 
                   continuous_feat, discrete_feat, ckpt_folder):
    text_all = pd.concat([train_val_dataset, test_dataset], axis=0)
    features = (continuous_feat + discrete_feat).copy()

    # Query the necessary data from the database instead of reading from CSV
    query = """
    SELECT published_at, rescue_id, volunteer_comment, user_phone, donor_phone, recipient_phone, recurrence_id, 
           user_hasphone, donor_hasphone, recipient_hasphone, is_recurrence
    FROM rescues_info_noorg;  -- Assuming this table holds the same data as 'all_rescues_info-noorg.csv'
    """
    
    acc_chunks = []
    with engine.connect() as con:
        for chunk in pd.read_sql(query, con, chunksize=100000):  # Read in chunks
            # Filter and modify features
            for feat in ['user_hasphone', 'donor_hasphone', 'recipient_hasphone', 'is_recurrence']:
                features.remove(feat)
            
            chunk['user_hasphone'] = chunk['user_phone'].isnull()
            chunk['donor_hasphone'] = chunk['donor_phone'].isnull()
            chunk['recipient_hasphone'] = chunk['recipient_phone'].isnull()
            chunk['is_recurrence'] = chunk['recurrence_id'].isnull()
            
            del chunk['user_phone']
            del chunk['donor_phone']
            del chunk['recipient_phone']
            del chunk['recurrence_id']
            
            features = continuous_feat + discrete_feat
            # Select only rows with null volunteer_comment for testing
            chunk_test_data = chunk[chunk['volunteer_comment'].isnull()][features + ['rescue_id']]
            acc_chunks.append(chunk_test_data)
    
    # Concatenate all chunks into full_test_data
    full_test_data = pd.concat(acc_chunks, axis=0)
    full_test_data = full_test_data.fillna(full_test_data.median())
    
    # Optional: Save full_test_data back to a CSV or database
    # full_test_data.to_csv('./dataset/eda/all_rescues_tabular.csv', index=False)

    X_train = np.array(text_all[features])
    y_train = np.array(text_all['difficulty'])
    y_train = np.clip(y_train, 1e-8, 1 - 1e-8)
    inv_sig_y_train = np.log(y_train / (1 - y_train))
    X_test = np.array(full_test_data[features])

    # Train model based on ckpt_folder
    if ckpt_folder == 'easy-scratch':
        model = RandomForestRegressor(random_state=42)
    elif ckpt_folder == 'hard-scratch':
        model = lgb.LGBMRegressor(random_state=42)
    else:
        raise NotImplementedError

    model.fit(X_train, inv_sig_y_train)

    # Prepare output for predictions
    output_train = pd.DataFrame()
    output_train['rescue_id'] = text_all['rescue_id'].apply(int)
    output_train['predicted_proba'] = y_train
    
    output_test = pd.DataFrame()
    output_test['rescue_id'] = full_test_data['rescue_id'].apply(int)
    output_test['predicted_proba'] = sigmoid(model.predict(X_test))
    
    fin_out = pd.concat([output_train, output_test], axis=0)
    
    # Optional: Save fin_out to a CSV or database
    fin_out.to_csv(f'./ckpt-{ckpt_folder}/fin_predicted_proba.csv', index=False)
    
    return


    

def temporal_analysis(ckpt_folder, threshold):
    # Query to retrieve the data instead of reading a CSV file
    query_proba = f"SELECT * FROM fin_predicted_proba WHERE ckpt_folder = '{ckpt_folder}';"
    
    with engine.connect() as con:
        # Load predicted probability data from database
        fin_out = pd.read_sql(query_proba, con)

    # Query to get the rescues with published_at dates
    query_rescues = """
    SELECT id AS rescue_id, published_at 
    FROM rescues
    WHERE published_at IS NOT NULL;
    """
    
    with engine.connect() as con:
        # Load rescue IDs and dates from the database
        id_datetime = pd.read_sql(query_rescues, con)

    # Merge predicted probabilities with rescue dates
    fin_out_date = pd.merge(fin_out, id_datetime, on='rescue_id', how='inner')

    # Optional: Save the merged data to a file
    fin_out_date.to_csv(f'./ckpt-{ckpt_folder}/fin_predicted_proba_wtime.csv', index=False)

    # Convert 'published_at' to datetime
    fin_out_date['published_at'] = pd.to_datetime(fin_out_date['published_at'])
    time_start = datetime(2022, 4, 1, 0, 0, 0)
    time_end = datetime(2022, 7, 1, 0, 0, 0)

    # Filter data within the time range
    fin_out_date = fin_out_date[(fin_out_date['published_at'] < time_end) & (fin_out_date['published_at'] >= time_start)]
    fin_out_date = fin_out_date.sort_values(by='published_at')

    # Apply threshold to create 'value_1' column
    fin_out_date['value_1'] = fin_out_date['predicted_proba'] >= threshold

    # Group by day, week, and month
    group_by_day = fin_out_date.groupby(fin_out_date['published_at'].dt.strftime('%Y-%m-%d'))['value_1'].sum()
    group_by_week = fin_out_date.groupby(fin_out_date['published_at'].dt.strftime('%Y-%W'))['value_1'].sum()
    group_by_month = fin_out_date.groupby(fin_out_date['published_at'].dt.strftime('%Y-%m'))['value_1'].sum()

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))

    group_by_day.plot(ax=ax[0])
    ax[0].set_xlabel('Time (YY-mm-dd)')
    ax[0].set_ylabel('Total Count')
    ax[0].set_title(ckpt_folder.split('-')[0] + ', daily')
    ax[0].tick_params(axis='x', rotation=30)

    group_by_week.plot(ax=ax[1])
    ax[1].set_xlabel('Time (YY-WW)')
    ax[1].set_ylabel('Total Count')
    ax[1].set_title(ckpt_folder.split('-')[0] + ', weekly')

    group_by_month.plot(ax=ax[2])
    ax[2].set_xlabel('Time (YY-mm)')
    ax[2].set_ylabel('Total Count')
    ax[2].set_title(ckpt_folder.split('-')[0] + ', monthly')

    plt.tight_layout()
    plt.savefig(f'./ckpt-{ckpt_folder}/past_three_combined_difficulty.png')
    plt.clf()

    return


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    final_metrics = {'easy': {}, 'hard': {}}

    # Main experiment loop for each difficulty split
    for diff_split in ['easy', 'hard']:
        split_metrics = {model: defaultdict(list) for model in ['lgb', 'rf', 'lr', 'svm', 'mlp', 'knn']}
        
        # Seed loop (10 seeds total)
        for seed in range(42, 52):
            # ========== BERT Training Phase ==========
            # 1. Train Rating-BERT with current seed
            if not (os.path.exists('train_ratings.csv') and os.path.exists('test_ratings.csv')):
                create_rating_csv(seed)
            
            ds_rating = create_rating_dataset()
            ds_rating = preprocess_bert_dataset(ds_rating, label_col='rating')
            rating_ckpt_dir = f"./ckpt-rating-seed{seed}-fast"
            
            if not os.path.exists(os.path.join(rating_ckpt_dir, "pytorch_model.bin")):
                train_eval_text(4, ds_rating['train'], ds_rating['test'], "rating", 
                               seed=seed, ckpt=None)

            # 2. Train Difficulty-BERT with current seed
            if not os.path.exists(f"train_{diff_split}.csv"):
                create_difficulty_csv(diff_split)
            
            ds_diff_full = load_dataset("csv", data_files={'train': f"train_{diff_split}.csv", 
                                                        'test': f"test_{diff_split}.csv"})
            ds_diff_full = preprocess_bert_dataset(ds_diff_full, label_col='difficulty')
            
            diff_ckpt_base = f"./ckpt-{diff_split}-seed{seed}-fast"
            if not os.path.exists(os.path.join(diff_ckpt_base, "pytorch_model.bin")):
                train_eval_text(2, ds_diff_full['train'], ds_diff_full['test'], diff_split,
                              seed=seed, ckpt=rating_ckpt_dir)

            # ========== Tabular Training Phase ==========
            # 3. Generate pseudo-labels with current seed's BERT
            df_all, df_labeled_all = prepare_pseudo(diff_split, load_df_all=True)
            df_comment, (cont_feat, disc_feat) = assign_pseudo_label(
                df_all, df_labeled_all, diff_split, 
                ckpt=diff_ckpt_base, label_type='soft')

            # 4. Train and evaluate tabular models
            train_df, val_df, test_df = split_tabular(diff_split, df_comment, 
                                                    cont_feat + disc_feat, split_type="raw",seed=seed)
            ckpt_folder = f"./ckpt-final-{diff_split}-seed{seed}"
            metrics = train_eval_tabular(train_df, val_df, test_df, 
                                       cont_feat, disc_feat, ckpt_folder, 
                                       "raw", seed=seed)

            # Store metrics for all models
            for model in ['lgb', 'rf', 'lr', 'svm', 'mlp', 'knn']:
                split_metrics[model]['auc'].append(metrics['test'][model]['roc'])
                split_metrics[model]['f1'].append(metrics['test'][model]['f1'])
                split_metrics[model]['acc'].append(metrics['test'][model]['acc'])
                split_metrics[model]['pr'].append(metrics['test'][model]['pr'])

        # ========== Result Aggregation ==========
        final_metrics[diff_split] = {
            model: {
                'auc': {'mean': np.mean(vals['auc']), 'std': np.std(vals['auc'])},
                'f1': {'mean': np.mean(vals['f1']), 'std': np.std(vals['f1'])},
                'acc': {'mean': np.mean(vals['acc']), 'std': np.std(vals['acc'])},
                'pr': {'mean': np.mean(vals['pr']), 'std': np.std(vals['pr'])}
            }
            for model, vals in split_metrics.items()
        }

    # ========== Final Reporting ==========
    print("\n=== Final Results (Mean ± StdDev over 10 seeds) ===")
    for split in ['easy', 'hard']:
        print(f"\n===== {split.upper()} TASKS =====")
        for model in ['lgb', 'rf', 'lr', 'svm', 'mlp', 'knn']:
            m = final_metrics[split][model]
            print(f"{model.upper():<4} | " +
                  f"AUC: {m['auc']['mean']:.3f}±{m['auc']['std']:.3f} | " +
                  f"F1: {m['f1']['mean']:.3f}±{m['f1']['std']:.3f} | " +
                  f"ACC: {m['acc']['mean']:.3f}±{m['acc']['std']:.3f} | " +
                  f"PR-AUC: {m['pr']['mean']:.3f}±{m['pr']['std']:.3f}")

    print("\nExperiment completed successfully!")

'''
if __name__ == "__main__":
    pd.options.mode.chained_assignment = None

    # 0) Possibly create rating CSV & train rating BERT
    if not os.path.exists('train_ratings.csv') or not os.path.exists('test_ratings.csv'):
        create_rating_csv()
    ds_rating = create_rating_dataset()
    ds_rating = preprocess_bert_dataset(ds_rating, label_col='rating')
    train_ratings = ds_rating['train']
    test_ratings  = ds_rating['test']

    rating_ckpt_dir = "./ckpt-rating-seed42-fast"
    print("\n===== BERT for Rating (One‐time) =====")
    rating_auc = train_eval_text(
        num_labels=4,
        train_dataset=train_ratings,
        eval_dataset=test_ratings,
        split="rating",
        ckpt=None,    # training from scratch
        seed=42       # or do multiple seeds if you prefer
    )
    print(f"BERT Rating AUC={rating_auc}")

    # 1) Create BERT for difficulty => e.g. easy/hard
    for diff_split in ['easy','hard']:
        if not os.path.exists(f"train_{diff_split}.csv"):
            create_difficulty_csv(diff_split)

        ds_diff = load_dataset("csv", data_files={
            'train':f"train_{diff_split}.csv",
            'test': f"test_{diff_split}.csv"
        })
        ds_diff = preprocess_bert_dataset(ds_diff, label_col='difficulty')

        print(f"\n===== BERT for Difficulty={diff_split} (One‐time) =====")
        train_diff = ds_diff['train']
        test_diff  = ds_diff['test']
        diff_auc = train_eval_text(
            num_labels=2,
            train_dataset=train_diff,
            eval_dataset=test_diff,
            split=diff_split,
            ckpt=rating_ckpt_dir,  # or actual rating checkpoint
            seed=42
        )
        print(f"BERT Difficulty={diff_split}, AUC={diff_auc}")

    ###########################################################
    # 2) 10-Seed Loop for the final Step
    #    We do the pseudo-label + tabular approach
    ###########################################################
    seeds = [42,43,44,45,46,47,48,49,50,51]
    model_names = ['lgb','rf','lr','svm','mlp','knn']

    for diff_split in ['easy','hard']:
        # aggregator: model => {acc:[], f1:[], roc:[], pr:[]}
        all_runs = {
            m: {'acc':[], 'f1':[], 'roc':[], 'pr':[]} for m in model_names
        }

        print(f"\n============== FINAL ALGORITHM - {diff_split.upper()} ==============")
        # read the big DF & labeled CSV
        df_all, df_labeled_all = prepare_pseudo(diff_split, load_df_all=True)

        # (A) For each random seed in seeds
        for s in seeds:
            print(f"--- {diff_split.upper()} | seed={s} ---")
            
            # (A) Fine-tune BERT on difficulty with current seed
            train_diff = ds_diff['train']
            test_diff  = ds_diff['test']
            diff_ckpt_dir = f"./ckpt-{diff_split}-seed{s}-fast"
            train_eval_text(
                num_labels=2,
                train_dataset=train_diff,
                eval_dataset=test_diff,
                split=diff_split,
                ckpt=rating_ckpt_dir,  # Pre-trained rating model
                seed=s  # Current seed
            )
            # pseudo-label the big dataset using the difficulty BERT
            # if you want each seed to also affect the BERT inference, you'd have to set
            # a seed in pipeline. But typically we fix the BERT weights & get the same pseudo‐labels.
        #ckpt_diff = f"./ckpt-{diff_split}-seed{s}-fast"  # Seed-specific checkpoint
            df_comment, _ = assign_pseudo_label(
                df_all,
                df_labeled_all,
                split=diff_split,
                ckpt=diff_ckpt_dir,  # ✅ Correct checkpoint
                label_type='soft'
            )

            # get base features
            cont_feat, disc_feat = get_base_features(df_comment)

            print("Continuous features used:", cont_feat)
            print("Discrete features used:", disc_feat)
            
            # pick your split_type: "time" or "raw"
            train_df, val_df, test_df = split_tabular(
                diff_split,
                df_comment,
                cont_feat+disc_feat,
                split_type="raw"
            )

            # (B) Train tabular models
            ckpt_folder = f"./ckpt-final-{diff_split}-seed{s}"
            model_res = train_eval_tabular(
                train_df, val_df, test_df,
                cont_feat, disc_feat,
                ckpt_folder=ckpt_folder,
                split_type="raw",
                seed=s
            )

            # (C) collect final test metrics
            for m in model_names:
                all_runs[m]['acc'].append(model_res['test'][m]['acc'])
                all_runs[m]['f1'].append(model_res['test'][m]['f1'])
                all_runs[m]['roc'].append(model_res['test'][m]['roc'])
                all_runs[m]['pr'].append(model_res['test'][m]['pr'])

        # after all seeds, compute mean±std
        print(f"\n=== Aggregated Results for FINAL {diff_split.upper()} Over {len(seeds)} Seeds ===")
        for m in model_names:
            acc_arr = np.array(all_runs[m]['acc'])
            f1_arr  = np.array(all_runs[m]['f1'])
            roc_arr = np.array(all_runs[m]['roc'])
            pr_arr  = np.array(all_runs[m]['pr'])
            print(
                f"{m:>3}: "
                f"ACC={acc_arr.mean():.3f}±{acc_arr.std():.3f}, "
                f"F1={f1_arr.mean():.3f}±{f1_arr.std():.3f}, "
                f"AUC={roc_arr.mean():.3f}±{roc_arr.std():.3f}, "
                f"PR={pr_arr.mean():.3f}±{pr_arr.std():.3f}"
            )

    print("\nDone. Table above shows final algorithm metrics for easy/hard.")
'''    


















    #########################################
    # FINAL-DIIFICULTY-PREDICTION
    #########################################
    # best_ckpt = {'easy':70, 'hard':130}
    # for ckpt_folder in ['easy-scratch','hard-scratch']:
    #     split = ckpt_folder.split('-')[0]
    #     df_all, df_labeled_all = prepare_pseudo(split, load_df_all=False)
    #     annot_comments, features = assign_pseudo_label(df_all, df_labeled_all, split, ckpt=f"./ckpt-{split}-scratch/checkpoint-{best_ckpt[split]}",plan='scratch',label_type='soft')
    #     continuous_feat, discrete_feat = features
    #     train_dataset, val_dataset, test_dataset = split_tabular(split, annot_comments, continuous_feat + discrete_feat, 'time')
    #     res, (train_val_dataset, test_dataset) = train_eval_tabular(train_dataset, val_dataset, test_dataset, continuous_feat, discrete_feat, f"./ckpt-{split}-scratch", 'time')
    #     fin_prediction(train_val_dataset, test_dataset, 
    #                    continuous_feat, discrete_feat, ckpt_folder)
    
    # for ckpt_folder, threshold in [('easy-scratch',0.16956734065005472)
    #                                ,('hard-scratch',0.020885445740001483)]:
    #     temporal_analysis(ckpt_folder, threshold)
        




    #########################################
    # FINAL-DIIFICULTY-PREDICTION
    #########################################
    # best_ckpt = {'easy':70, 'hard':130}
    # for ckpt_folder in ['easy-scratch','hard-scratch']:
    #     split = ckpt_folder.split('-')[0]
    #     df_all, df_labeled_all = prepare_pseudo(split, load_df_all=False)
    #     annot_comments, features = assign_pseudo_label(df_all, df_labeled_all, split, ckpt=f"./ckpt-{split}-scratch/checkpoint-{best_ckpt[split]}",plan='scratch',label_type='soft')
    #     continuous_feat, discrete_feat = features
    #     train_dataset, val_dataset, test_dataset = split_tabular(split, annot_comments, continuous_feat + discrete_feat, 'time')
    #     res, (train_val_dataset, test_dataset) = train_eval_tabular(train_dataset, val_dataset, test_dataset, continuous_feat, discrete_feat, f"./ckpt-{split}-scratch", 'time')
    #     fin_prediction(train_val_dataset, test_dataset, 
    #                    continuous_feat, discrete_feat, ckpt_folder)
    
    # for ckpt_folder, threshold in [('easy-scratch',0.16956734065005472)
    #                                ,('hard-scratch',0.020885445740001483)]:
    #     temporal_analysis(ckpt_folder, threshold)
        

#1. log in access good 2. code no bug and run /train initia version model // track performance matric 3. 
# DBeaver to visualize data



# REPORT AUC RUNNING 3 MODEL TO GET PERFORMANCE METRIC 
# TRAIN DIFF HP PARA kep119

"""
    # Set training arguments
    if ckpt is None:
        if num_labels == 4:
            training_args = TrainingArguments(
                output_dir=output_dir, 
                evaluation_strategy="no",
                num_train_epochs=3,
                logging_steps=500,
                save_steps=500  # Set save_steps to save intermediate checkpoints
            )
        elif num_labels == 2:
            training_args = TrainingArguments(
                output_dir=output_dir, 
                evaluation_strategy="no",
                num_train_epochs=4,
                logging_steps=10,
                save_steps=500  # Set save_steps to save intermediate checkpoints
            )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir, 
            evaluation_strategy="no",
            num_train_epochs=3,
            logging_steps=10,
            save_steps=500  # Set save_steps to save intermediate checkpoints
        )
"""

"""
    for split in ['easy','hard']: # our final models here
        #print(split)
        print(f"\n============== PSEUDO-LABEL & TABULAR MODEL ({split}) ==============")
        if not os.path.exists(f'/home/mic175/food-rescue-difficulty/train_{split}.csv'):
            #This method gets the data files from PostGres and then processes then and renames them based off of the split value. 
            #It also creates the difficulty attribute and relables them to make sense
            create_difficulty_csv(split)

        #This method takes the split value and looks for the test and training csvs from the root path that it is given. 
        #After this, it returns the file for further processing
        train_test_difficulty = create_difficulty_dataset(split)
        #Tokenization: The text field is tokenized using the BERT tokenizer, ensuring text data is formatted correctly.
        #Label Adjustments: Depending on whether label_col is 'difficulty', the corresponding label field is adjusted.
        #Dataset Processing: The method processes the dataset by mapping the tokenization and relabeling functions over the dataset.
        #This function basically just relabels the dataset so that the ML model is able to understand
        processed_train_test_difficulty = preprocess_bert_dataset(train_test_difficulty, label_col='difficulty')
        train_difficulty = processed_train_test_difficulty['train']
        test_difficulty = processed_train_test_difficulty['test']
        #The train_eval_text function is designed to train a BERT model for a text classification task, using either a fresh or fine-tuned model. 
        #It sets up the training process, including logging, evaluation metrics, and model saving. 
        #The ROC AUC metric is used to evaluate the model’s performance, particularly useful for binary or multi-class classification tasks.
        train_eval_text(2, train_difficulty, test_difficulty, split, ckpt=None)
"""


'''
    if ckpt is None:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", 
                                                                    num_labels=num_labels)
        if num_labels == 4:
            output_dir = "./ckpt-rating"
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",  # <-- Changed from "no" to "epoch"
                num_train_epochs=3,
                logging_steps=500,
                save_steps=500
            )
        elif num_labels == 2:
            output_dir = f"./ckpt-{split}-scratch"
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",  # <-- Changed
                num_train_epochs=4,
                logging_steps=10,
                save_steps=500
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, 
                                                                    num_labels=num_labels,
                                                                    ignore_mismatched_sizes=True)
        output_dir = f"./ckpt-{split}-finetune"
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",  # <-- Use epoch as well
            num_train_epochs=3,
            logging_steps=10,
            save_steps=500
        )
    
    # Ensure the checkpoint directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created checkpoint directory: {output_dir}")
    else:
        print(f"Checkpoint directory already exists: {output_dir}")
    '''
    # Define the metric and trainer
    #metric = evaluate.load("roc_auc")
'''
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        #predictions = np.argmax(logits, axis=-1)
        pred_scores = sigmoid(logits)[:, 1]

        # Add multi_class='ovr' or 'ovo' for multi-class AUC handling
        return metric.compute(
            prediction_scores=pred_scores, 
            references=labels,
            multi_class='ovr'  # or 'ovo', depending on your use case
        )
'''