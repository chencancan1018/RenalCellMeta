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

def split_dev_downsample(dev_idxes, labels, fold=5):
    m0_idxes = [i for i in dev_idxes if labels[i] == 0]
    m1_idxes = [i for i in dev_idxes if labels[i] == 1]
    m0_idxes_ = sorted(random.sample(m0_idxes, k= len(m1_idxes)))
    remained_m0_idxes = sorted(list(set(m0_idxes) - set(m0_idxes_)))
    m0_idxes = m0_idxes_
    kf = KFold(n_splits=fold, shuffle=True)
    m0_lst = []
    for train_idx, val_idx in kf.split(m0_idxes):
        print(f"{fold} fold cross-validation for m0: {len(train_idx), len(val_idx)}")
        train_idx = list(np.array(m0_idxes)[train_idx])
        val_idx = list(np.array(m0_idxes)[val_idx])
        m0_lst.append([train_idx, val_idx, remained_m0_idxes])
    m1_lst = []
    for train_idx, val_idx in kf.split(m1_idxes):
        print(f"{fold} fold cross-validation for m1: {len(train_idx), len(val_idx)}")
        train_idx = list(np.array(m1_idxes)[train_idx])
        val_idx = list(np.array(m1_idxes)[val_idx])
        m1_lst.append([train_idx, val_idx])
    fold_idxes_lst = [m0_lst, m1_lst]
    return fold_idxes_lst

def split_dev_on_upsample(dev_idxes, labels, fold=5):
    m0_idxes = [i for i in dev_idxes if labels[i] == 0]
    m1_idxes = [i for i in dev_idxes if labels[i] == 1]
    multiple = int(len(m0_idxes) // (len(m1_idxes) + 1))
    m1_idxes_ = m1_idxes * multiple
    kf = KFold(n_splits=fold, shuffle=True)
    m0_lst = []
    for train_idx, val_idx in kf.split(m0_idxes):
        print(f"{fold} fold cross-validation for m0: {len(train_idx), len(val_idx)}")
        train_idx = list(np.array(m0_idxes)[train_idx])
        val_idx = list(np.array(m0_idxes)[val_idx])
        m0_lst.append([train_idx, val_idx])
    m1_lst = []
    for train_idx, val_idx in kf.split(m1_idxes_):
        print(f"{fold} fold cross-validation for m1: {len(train_idx), len(val_idx)}")
        train_idx = list(np.array(m1_idxes_)[train_idx])
        val_idx = list(np.array(m1_idxes_)[val_idx])
        m1_lst.append([train_idx, val_idx])
    fold_idxes_lst = [m0_lst, m1_lst]
    return fold_idxes_lst

def RFTest(train_x, train_y, val_x, val_y, test_x, test_y, ext_x, ext_y):
    test_aucs = []; test_accs = []; test_f1s = []
    ext_aucs = []; ext_accs = []; ext_f1s = []
    for iteration in range(1):
        para_ests = [i for i in range(10,100,10)]
        para_deps = [i for i in range(5,50,5)]
        para_leafs = [i for i in range(5,20,5)]
        for para_est in para_ests:
            for para_dep in para_deps:
                for para_leaf in para_leafs:
                    cur_outdir = os.path.join(outdir, 'rf_'+str(iteration)+'_'+str(para_est)+'_'+str(para_dep)+'_'+str(para_leaf))
                    model = RandomForestClassifier(n_estimators=para_est, max_depth=para_dep, min_samples_leaf=para_leaf, class_weight='balanced', bootstrap=True)
                    model.fit(train_x, train_y)
                    train_pred_y = model.predict_proba(train_x)
                    val_pred_y = model.predict_proba(val_x)
                    test_pred_y = model.predict_proba(test_x)
                    ext_pred_y = model.predict_proba(ext_x) 
                    train_auc, train_acc, train_sens, train_spec, train_ppv, train_npv, train_f1, train_auprc = utils.get_matrix(train_pred_y, train_y)
                    val_auc, val_acc, val_sens, val_spec, val_ppv, val_npv, val_f1, val_auprc = utils.get_matrix(val_pred_y, val_y)
                    test_auc, test_acc, test_sens, test_spec, test_ppv, test_npv, test_f1, test_auprc = utils.get_matrix(test_pred_y, test_y)
                    ext_auc, ext_acc, ext_sens, ext_spec, ext_ppv, ext_npv, ext_f1, ext_auprc = utils.get_matrix(ext_pred_y, ext_y)
                    test_aucs.append(test_auc); test_accs.append(test_acc); test_f1s.append(test_f1)
                    ext_aucs.append(ext_auc); ext_accs.append(ext_acc); ext_f1s.append(ext_f1)
                    print(f'train {iteration} perf: {train_auc, train_acc, train_sens, train_spec, train_ppv, train_npv, train_f1, train_auprc}; paras: {para_est, para_dep, para_leaf}')
                    print(f'val {iteration} perf: {val_auc, val_acc, val_sens, val_spec, val_ppv, val_npv, val_f1, val_auprc}; paras: {para_est, para_dep, para_leaf}')
                    print(f'test {iteration} perf: {test_auc, test_acc, test_sens, test_spec, test_ppv, test_npv, test_f1, test_auprc}; paras: {para_est, para_dep, para_leaf}')
                    print(f'external {iteration} perf: {ext_auc, ext_acc, ext_sens, ext_spec, ext_ppv, ext_npv, ext_f1, ext_auprc}; paras: {para_est, para_dep, para_leaf}')
    print(max(test_aucs), max(test_f1s), max(ext_aucs), max(ext_f1s))
    
def RFModel(train_x, train_y, val_x, val_y, test_x, test_y, ext_x, ext_y, pids, feats_name, outdir):
    test_aucs = []; test_accs = []; test_f1s = []
    ext_aucs = []; ext_accs = []; ext_f1s = []
    for iteration in range(50):
        para_ests = [i for i in range(1,30)]
        para_deps = [i for i in range(3,10,1)]
        para_leafs = [i for i in range(1,10,1)]
        for para_est in para_ests:
            for para_dep in para_deps:
                for para_leaf in para_leafs:
                    cur_outdir = os.path.join(outdir, 'rf_'+str(iteration)+'_'+str(para_est)+'_'+str(para_dep)+'_'+str(para_leaf))
                    model = RandomForestClassifier(n_estimators=para_est, max_depth=para_dep, min_samples_leaf=para_leaf, class_weight='balanced', bootstrap=True)
                    model.fit(train_x, train_y)
                    train_pred_y = model.predict_proba(train_x)
                    val_pred_y = model.predict_proba(val_x)
                    test_pred_y = model.predict_proba(test_x)
                    ext_pred_y = model.predict_proba(ext_x) 
                    train_auc, train_acc, train_sens, train_spec, train_ppv, train_npv, train_f1, train_auprc = utils.get_matrix(train_pred_y, train_y)
                    val_auc, val_acc, val_sens, val_spec, val_ppv, val_npv, val_f1, val_auprc = utils.get_matrix(val_pred_y, val_y)
                    test_auc, test_acc, test_sens, test_spec, test_ppv, test_npv, test_f1, test_auprc = utils.get_matrix(test_pred_y, test_y)
                    ext_auc, ext_acc, ext_sens, ext_spec, ext_ppv, ext_npv, ext_f1, ext_auprc = utils.get_matrix(ext_pred_y, ext_y)
                    test_aucs.append(test_auc); test_accs.append(test_acc); test_f1s.append(test_f1)
                    ext_aucs.append(ext_auc); ext_accs.append(ext_acc); ext_f1s.append(ext_f1)
                    
                    if (
                        (test_auc > 0.8) and (test_acc > 0.8) and
                        (ext_auc > 0.78) and (ext_acc > 0.8)
                        ):
                        os.makedirs(cur_outdir, exist_ok=True)
                        f = open(os.path.join(cur_outdir, 'rf.pickle'), 'wb')
                        pickle.dump(model, f)
                        f.close()
                        feats_importance = model.feature_importances_
                        pd.DataFrame(
                            {'features': feats_name, 'importance': feats_importance}
                        ).to_csv(os.path.join(cur_outdir, "feature_importance.csv"), index=None)
                        print(f'train {iteration} perf: {train_auc, train_acc, train_sens, train_spec, train_ppv, train_npv, train_f1, train_auprc}; paras: {para_est, para_dep, para_leaf}')
                        print(f'val {iteration} perf: {val_auc, val_acc, val_sens, val_spec, val_ppv, val_npv, val_f1, val_auprc}; paras: {para_est, para_dep, para_leaf}')
                        print(f'test {iteration} perf: {test_auc, test_acc, test_sens, test_spec, test_ppv, test_npv, test_f1, test_auprc}; paras: {para_est, para_dep, para_leaf}')
                        print(f'external {iteration} perf: {ext_auc, ext_acc, ext_sens, ext_spec, ext_ppv, ext_npv, ext_f1, ext_auprc}; paras: {para_est, para_dep, para_leaf}')
                        train_pids = pids[0]
                        val_pids = pids[1]
                        test_pids = pids[2]
                        ext_pids = pids[3]
                        pd.DataFrame(
                            np.hstack([train_pids[:, None], train_pred_y, train_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
                        ).to_csv(os.path.join(cur_outdir, "train_pred.csv"), index=None)
                        pd.DataFrame(
                            np.hstack([val_pids[:, None], val_pred_y, val_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
                        ).to_csv(os.path.join(cur_outdir, "val_pred.csv"), index=None)
                        pd.DataFrame(
                            np.hstack([test_pids[:, None], test_pred_y, test_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
                        ).to_csv(os.path.join(cur_outdir, "test_pred.csv"), index=None)
                        pd.DataFrame(
                            np.hstack([ext_pids[:, None], ext_pred_y, ext_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
                        ).to_csv(os.path.join(cur_outdir, "ext_pred.csv"), index=None)
    print(max(test_aucs), max(ext_aucs))

if __name__ == "__main__":
    modes = ["bayes", "svm", "decision_tree", "random_forest","neutral_network", "xgboost", "logistic"]
    for mode in modes:
        develop_df = pd.read_csv("", encoding='utf8')
        test_df = pd.read_csv("", encoding='utf8')
        ext_df = pd.read_csv("", encoding='utf8')
        outdir = ""
        os.makedirs(outdir, exist_ok=True)

        valid_cols = ['Sex', 'Age', 'Tumor Grade', 'Side', 'Pathology Type', 'N stage', 'T stage', 'Surgery', 'Tumor size', 'Married status']
        label_col = 'M stage'

        dev_idxes_path = './checkpoint_0423/miss/train_val_lst.pickle'
        if os.path.exists(dev_idxes_path):
            with open(dev_idxes_path, 'rb') as f:
                fold_idxes_lst = pickle.load(f)
        else:
            dev_idxes = [i for i in range(develop_df.shape[0])]
            dev_labels = get_labels(develop_df.loc[:, "M stage"].values)
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
            val_idxes = sorted(list(set(val_idxes)))
            print(f"train num: {len(train_idxes)}, val num: {len(val_idxes)}, test num: {test_df.shape[0]}, ext num: {ext_df.shape[0]}")

            train_m0_idxes = [i for i in train_idxes if develop_df.loc[i, label_col] == 'M0']
            train_m1_idxes = [i for i in train_idxes if develop_df.loc[i, label_col] == 'M1']
            num_for_downsample = len(train_m1_idxes)
            multiple_for_upsample = int(len(train_m0_idxes) // (len(train_m1_idxes) + 1))
            # # downsample
            # new_train_idxes = sorted(random.sample(train_m0_idxes, k=num_for_downsample) + train_m1_idxes)
            # for upsample
            new_train_idxes = sorted(train_m0_idxes + train_m1_idxes * multiple_for_upsample)
            new_train_x = develop_df.loc[new_train_idxes, valid_cols].values
            new_train_y = get_labels(develop_df.loc[new_train_idxes, label_col].values)

            train_x = develop_df.loc[train_idxes, valid_cols].values
            train_y = get_labels(develop_df.loc[train_idxes, label_col].values)
            val_x = develop_df.loc[val_idxes, valid_cols].values
            val_y = get_labels(develop_df.loc[val_idxes, label_col].values)
            train_pids = develop_df.iloc[train_idxes, 0].values
            val_pids = develop_df.iloc[val_idxes, 0].values
            train_x = scaler.fit_transform(train_x)
            new_train_x = scaler.fit_transform(new_train_x)
            val_x = scaler.fit_transform(val_x)

            tmp_outdir = osp.join(outdir, 'fold' + str(i+1) + 'fordev')
            os.makedirs(tmp_outdir, exist_ok=True)
            all_pids = [train_pids, val_pids, test_pids, ext_pids]
            print(f"================== fold-{i}  =====================")
            print("train: ", (train_y == 0).sum(), (train_y == 1).sum(), len(train_idxes))
            print("val: ", (val_y == 0).sum(), (val_y == 1).sum(), len(val_idxes))
            print("test: ", (test_y == 0).sum(), (test_y == 1).sum(), test_df.shape[0])
            print("ext: ", (ext_y == 0).sum(), (ext_y == 1).sum(), ext_df.shape[0])

            if mode in ["bayes", "svm", "decision_tree", "random_forest", "neutral_network", "xgboost", "logistic"]:
                cur_outdir = os.path.join(tmp_outdir, mode)
                os.makedirs(cur_outdir, exist_ok=True)

                model = utils.get_mode(mode)
                # model.fit(train_x, train_y)
                model.fit(new_train_x, new_train_y)
                train_pred_y = model.predict_proba(train_x)
                val_pred_y = model.predict_proba(val_x)
                test_pred_y = model.predict_proba(test_x)
                ext_pred_y = model.predict_proba(ext_x) 
                train_auc, train_acc, train_sens, train_spec, train_ppv, train_npv, train_f1, train_auprc = utils.get_matrix(train_pred_y, train_y)
                val_auc, val_acc, val_sens, val_spec, val_ppv, val_npv, val_f1, val_auprc = utils.get_matrix(val_pred_y, val_y)
                test_auc, test_acc, test_sens, test_spec, test_ppv, test_npv, test_f1, test_auprc = utils.get_matrix(test_pred_y, test_y)
                ext_auc, ext_acc, ext_sens, ext_spec, ext_ppv, ext_npv, ext_f1, ext_auprc = utils.get_matrix(ext_pred_y, ext_y)
                print(f"For {mode} and train, train acc: {train_acc:.4f}, train auc: {train_auc:.4f}, train_f1: {train_f1: .4f}, {train_sens, train_spec, train_ppv, train_npv, train_auprc}")
                print(f"For {mode} and val, val acc: {val_acc:.4f}, val auc: {val_auc:.4f}, val_f1: {val_f1: .4f}, {val_sens, val_spec, val_ppv, val_npv, val_auprc}")
                print(f"For {mode} and test, test acc: {test_acc:.4f}, test auc: {test_auc:.4f}, test_f1: {test_f1: .4f}, {test_sens, test_spec, test_ppv, test_npv, test_auprc}")
                print(f"For {mode} and ext, ext acc: {ext_acc:.4f}, ext auc: {ext_auc:.4f}, ext_f1: {test_f1: .4f}, {ext_sens, ext_spec, ext_ppv, ext_npv, ext_auprc}")
                f = open(os.path.join(cur_outdir, mode+'.pickle'), 'wb')
                pickle.dump(model, f)
                f.close()
                # feats_importance = model.feature_importances_
                # pd.DataFrame(
                #     {'features': valid_cols, 'importance': feats_importance}
                # ).to_csv(os.path.join(cur_outdir, "feature_importance.csv"), index=None)

                train_pids = all_pids[0]
                val_pids = all_pids[1]
                test_pids = all_pids[2]
                ext_pids = all_pids[3]
                pd.DataFrame(
                    np.hstack([train_pids[:, None], train_pred_y, train_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
                ).to_csv(os.path.join(cur_outdir, "train_pred.csv"), index=None)
                pd.DataFrame(
                    np.hstack([val_pids[:, None], val_pred_y, val_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
                ).to_csv(os.path.join(cur_outdir, "val_pred.csv"), index=None)
                pd.DataFrame(
                    np.hstack([test_pids[:, None], test_pred_y, test_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
                ).to_csv(os.path.join(cur_outdir, "test_pred.csv"), index=None)
                pd.DataFrame(
                    np.hstack([ext_pids[:, None], ext_pred_y, ext_y[:, None]]), columns=['pid', '0-prob', '1-prob',  'label']
                ).to_csv(os.path.join(cur_outdir, "ext_pred.csv"), index=None)
        
            # elif mode == "random_forest":
            #     RFTest(train_x, train_y, val_x, val_y, test_x, test_y, ext_x, ext_y)
            #     # RFModel(train_x, train_y, val_x, val_y, test_x, test_y, ext_x, ext_y, all_pids, valid_cols, tmp_outdir)
            else:
                raise KeyError


