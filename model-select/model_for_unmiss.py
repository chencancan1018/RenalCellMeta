import os
import sys
import time
import pickle
import random
import numpy as np
import pandas as pd
import os.path as osp

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.model_selection import KFold

sys.path.append("../")
import utils

def split_dev(dev_idxes, fold=5):
    fold_idxes_lst = []
    kf = KFold(n_splits=fold, shuffle=True)
    for train_idx, val_idx in kf.split(dev_idxes):
        print(f"{fold} fold cross-validation: {len(train_idx), len(val_idx)}")
        train_idx = list(np.array(dev_idxes)[train_idx])
        val_idx = list(np.array(dev_idxes)[val_idx])
        fold_idxes_lst.append([train_idx, val_idx])
    return fold_idxes_lst

def get_labels(arr):
    labels = []
    for i in range(len(arr)):
        if arr[i] == 'M0':
            labels.append(0)
        elif arr[i] == 'M1':
            labels.append(1)
        else:
            raise KeyError(f"wrong label with{i, arr[i]}")
    return np.array(labels)

if __name__ == "__main__":
    
    develop_df = pd.read_csv("", encoding='utf8')
    test_df = pd.read_csv("", encoding='utf8')
    ext_df = pd.read_csv("", encoding='utf8')
    outdir = ""
    os.makedirs(outdir, exist_ok=True)

    valid_cols = ['Sex', 'Age', 'Tumor Grade', 'Side', 'Pathology Type', 'N stage', 'T stage', 'Surgery', 'Tumor size', 'Married status']
    label_col = 'M stage'

    dev_idxes_path = ''
    if os.path.exists(dev_idxes_path):
        with open(dev_idxes_path, 'rb') as f:
            fold_idxes_lst = pickle.load(f)
    else:
        dev_idxes = [i for i in range(develop_df.shape[0])]
        fold_idxes_lst = split_dev(dev_idxes)
        with open(dev_idxes_path, 'wb') as f:
            pickle.dump(fold_idxes_lst, f)

    test_x = test_df.loc[:, valid_cols].values
    test_y = get_labels(test_df.loc[:, label_col].values)
    ext_x = ext_df.loc[:, valid_cols].values
    ext_y = get_labels(ext_df.loc[:, label_col].values)
    test_pids = test_df.iloc[:, 0].values
    ext_pids = ext_df.iloc[:, 0].values
    
    scaler = MinMaxScaler()
    scaler1 = Normalizer(norm='l2')
    test_x = scaler.fit_transform(test_x)
    ext_x = scaler.fit_transform(ext_x)

    for i, ll in enumerate(fold_idxes_lst):
        train_idxes = ll[0]
        val_idxes = ll[1]
        print(f"train num: {len(train_idxes)}, val num: {len(val_idxes)}, test num: {test_df.shape[0]}, ext num: {ext_df.shape[0]}")

        train_idxes_again = sorted(random.sample(train_idxes, k=len(val_idxes)))
        train_x = develop_df.loc[train_idxes, valid_cols].values
        train_y = get_labels(develop_df.loc[train_idxes, label_col].values)
        val_x = develop_df.loc[val_idxes, valid_cols].values
        val_y = get_labels(develop_df.loc[val_idxes, label_col].values)
        train_pids = develop_df.iloc[train_idxes, 0].values
        val_pids = develop_df.iloc[val_idxes, 0].values
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.fit_transform(val_x)

        model_strs = ["bayes", "svm", "decision_tree", "random_forest", "neutral_network", "xgboost", "logistic"]
        for ms in model_strs:
            print(f"start to train model {ms}")
            start_time = time.time()
            if ms == "svm":
                model = SVC(gamma='auto', class_weight='balanced', probability=True)
            elif ms == "bayes":
                model = GaussianNB()
            elif ms == "random_forest":
                model = RandomForestClassifier(n_estimators=30, max_depth=10, min_samples_leaf=5, class_weight='balanced', bootstrap=True)
            elif ms == "decision_tree":
                model = DecisionTreeClassifier(max_depth=20, min_samples_split=10, min_samples_leaf=10, class_weight='balanced')
            elif ms == "neutral_network":
                model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 10), random_state=1)
            elif ms == "xgboost":
                model = XGBClassifier(max_depth=10, n_estimators=30)
            elif ms == "logistic":
                model = LogisticRegression(class_weight='balanced', random_state=0)
            else:
                raise KeyError(f"wrong model indicator with {ms}")
            
            cur_ms_outdir = osp.join(outdir, 'fold' + str(i+1) + 'fortest', ms)
            os.makedirs(cur_ms_outdir, exist_ok=True)
            model.fit(train_x, train_y)
            f = open(os.path.join(cur_ms_outdir, ms + '.pickle'), 'wb')
            pickle.dump(model, f)
            f.close()
            train_pred_y = model.predict_proba(train_x)
            val_pred_y = model.predict_proba(val_x)
            test_pred_y = model.predict_proba(test_x)
            ext_pred_y = model.predict_proba(ext_x) 
            pd.DataFrame(
                np.hstack([train_pids[:, None], train_pred_y, train_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
            ).to_csv(os.path.join(cur_ms_outdir, "train_pred.csv"), index=None)
            pd.DataFrame(
                np.hstack([val_pids[:, None], val_pred_y, val_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
            ).to_csv(os.path.join(cur_ms_outdir, "val_pred.csv"), index=None)
            pd.DataFrame(
                np.hstack([test_pids[:, None], test_pred_y, test_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
            ).to_csv(os.path.join(cur_ms_outdir, "test_pred.csv"), index=None)
            pd.DataFrame(
                np.hstack([ext_pids[:, None], ext_pred_y, ext_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
            ).to_csv(os.path.join(cur_ms_outdir, "ext_pred.csv"), index=None)
            
            train_auc, train_acc, train_sens, train_spec, train_ppv, train_npv, train_f1, train_auprc = utils.get_matrix(train_pred_y, train_y)
            val_auc, val_acc, val_sens, val_spec, val_ppv, val_npv, val_f1, val_auprc = utils.get_matrix(val_pred_y, val_y)
            test_auc, test_acc, test_sens, test_spec, test_ppv, test_npv, test_f1, test_auprc = utils.get_matrix(test_pred_y, test_y)
            ext_auc, ext_acc, ext_sens, ext_spec, ext_ppv, ext_npv, ext_f1, ext_auprc = utils.get_matrix(ext_pred_y, ext_y)

            print(f"For train dataset, train acc: {train_acc:.4f}, train auc: {train_auc:.4f}, train_f1: {train_f1: .4f}, {train_sens, train_spec, train_ppv, train_npv, train_auprc}")
            print(f"For val dataset, val acc: {val_acc:.4f}, val auc: {val_auc:.4f}, val_f1: {val_f1: .4f}, {val_sens, val_spec, val_ppv, val_npv, val_auprc}")
            print(f"For test dataset, test acc: {test_acc:.4f}, test auc: {test_auc:.4f}, test_f1: {test_f1: .4f}, {test_sens, test_spec, test_ppv, test_npv, test_auprc}")
            print(f"For ext dataset, ext acc: {ext_acc:.4f}, ext auc: {ext_auc:.4f}, ext_f1: {ext_f1: .4f}, {ext_sens, ext_spec, ext_ppv, ext_npv, ext_auprc}")
            end_time = time.time()
            use_time = int((end_time - start_time)/ 60)
            print(f"training model {ms} cost {use_time} min!")



        