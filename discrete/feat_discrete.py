import os 
import numpy as np
import pandas as pd

def kernel_2(x, y):
    '''
    Mann-Whitney statistic
    '''
    return .5 if x==y else int(y < x)

def get_two_groups_auc(preds, actual):
    X, Y = group_preds_by_label(preds, actual)
    return 1/(len(X)*len(Y)) * sum([kernel_2(x, y) for x in X for y in Y])

def group_preds_by_label(preds, actual):
    X = [p for (p, a) in zip(preds, actual) if a==1]
    Y = [p for (p, a) in zip(preds, actual) if a==0]
    return X, Y

def get_three_groups_auc(preds, actual):
    labels0 = (actual.copy() == 0) * 1
    auc0 = get_two_groups_auc(preds, labels0)
    labels1 = (actual.copy() == 1) * 1
    auc1 = get_two_groups_auc(preds, labels1)
    labels2 = (actual.copy() == 2) * 1
    auc2 = get_two_groups_auc(preds, labels2)
    mean_auc = np.mean([auc0, auc1, auc2])
    return mean_auc

def get_auc(preds, actual):
    if len(np.unique(actual)) == 2:
        return get_two_groups_auc(preds, actual)
    elif len(np.unique(actual)) == 3:
        return get_three_groups_auc(preds, actual)
    else:
        raise ValueError(f"Wrong labels: {np.unique(actual)}!")
    
def dis_based_on_auc(feat_df, all_labels, columns, valid_idxes):
    opt_cuts = []
    for col in columns:
        cur_f = feat_df.loc[:, col].values
        assert len(cur_f) == len(all_labels)
        
        tmp_cur_f = cur_f.copy()[valid_idxes]
        tmp_all_lables = all_labels.copy()[valid_idxes]
        
        max_f = tmp_cur_f.max()
        min_f = tmp_cur_f.min()
        bins = np.linspace(min_f, max_f, 30)[2:28]
        opt_bin = bins[0]
        cur_f_bin = (tmp_cur_f.copy() > opt_bin) * 1
        opt_auc = get_auc(cur_f_bin, tmp_all_lables)
        for b in bins:
            cur_f_bin = (tmp_cur_f.copy() > b) * 1
            auc = get_auc(cur_f_bin, tmp_all_lables)
            if auc > opt_auc:
                opt_bin = b
                opt_auc = auc
        print(col, opt_bin, opt_auc)
        cur_f_bin = (cur_f.copy() > opt_bin) * 1
        feat_df.loc[:, col] = cur_f_bin
        opt_cuts.append([col, opt_bin])
    cuts_df = pd.DataFrame(opt_cuts, columns=['feature', 'opt cut-off'])
    return feat_df, cuts_df


def dis_by_bins(feat_df, all_labels, columns, valid_idxes):
    opt_cuts = []
    for col in columns:
        cur_f = feat_df.loc[:, col].values
        assert len(cur_f) == len(all_labels)

        tmp_cur_f = cur_f.copy()[valid_idxes]
        tmp_all_lables = all_labels.copy()[valid_idxes]
        max_f = tmp_cur_f.max()
        min_f = tmp_cur_f.min()
        opt_bin = (max_f + min_f) / 2
        cur_f_bin = (tmp_cur_f.copy() >= opt_bin) * 1
        opt_auc = get_auc(cur_f_bin, tmp_all_lables)
        print(col, opt_bin, opt_auc)
        cur_f_bin = (cur_f.copy() > opt_bin) * 1
        feat_df.loc[:, col] = cur_f_bin
        opt_cuts.append([col, opt_bin])
    cuts_df = pd.DataFrame(opt_cuts, columns=['feature', 'opt cut-off'])
    return feat_df, cuts_df

def dis_by_equifrequency(feat_df, all_labels, columns, valid_idxes):
    opt_cuts = []
    for col in columns:
        cur_f = feat_df.loc[:, col].values
        assert len(cur_f) == len(all_labels)
        index = int(len(cur_f) // 2)
        tmp_cur_f = cur_f.copy()[valid_idxes]
        tmp_all_lables = all_labels.copy()[valid_idxes]
        opt_cut = sorted(list(tmp_cur_f))[index]
        cur_f_cut = (tmp_cur_f.copy() >= opt_cut) * 1
        opt_auc = get_auc(cur_f_cut, tmp_all_lables)
        print(col, opt_cut, opt_auc)
        cur_f_cut = (cur_f.copy() > opt_cut) * 1
        feat_df.loc[:, col] = cur_f_cut
        opt_cuts.append([col, opt_cut])
    cuts_df = pd.DataFrame(opt_cuts, columns=['feature', 'opt cut-off'])
    return feat_df, cuts_df

def get_info_entropy(labels_set):
    elements = set(labels_set)
    ent = 0
    for e in elements:
        p = labels_set.count(e) / len(labels_set)
        if p > 0:
            ent += -p * np.log2(p)
    return ent

def dis_by_info_entropy(feat_df, all_labels, columns, valid_idxes):
    opt_cuts = []
    for col in columns:
        cur_f = feat_df.loc[:, col].values
        assert len(cur_f) == len(all_labels)

        if len(np.unique(cur_f)) <= 2:
            opt_cut = list(np.unique(cur_f))[0]
        elif len(np.unique(cur_f)) == 3:
            opt_cut = list(np.unique(cur_f))[1]
        elif len(np.unique(cur_f)) > 3:
            tmp_cur_f = cur_f.copy()[valid_idxes]
            tmp_all_lables = all_labels.copy()[valid_idxes]
            tmp_values = sorted(list(tmp_cur_f.copy()))
            tmp_idxes = [int(len(tmp_values) * i) for i in list(np.linspace(0.1,0.9,10))]
            tmp_cuts = [tmp_values[i] for i in tmp_idxes]
            opt_cut = tmp_cuts[0]
            cur_f_cut = (tmp_cur_f.copy() >= opt_cut) * 1
            ori_ent = get_info_entropy(list(tmp_all_lables.copy()))
            set_A = [tmp_all_lables[i] for i in range(len(tmp_all_lables)) if cur_f_cut[i] >= 0.5]
            set_B = [tmp_all_lables[i] for i in range(len(tmp_all_lables)) if cur_f_cut[i] < 0.5]
            split_ent = (get_info_entropy(set_A) * len(set_A) + get_info_entropy(set_B) * len(set_B)) / len(tmp_all_lables)
            opt_ent = ori_ent - split_ent
            print(col, opt_cut, opt_ent)
            for c in tmp_cuts[1:]:
                cur_f_cut = (tmp_cur_f.copy() >= c) * 1
                set_A = [tmp_all_lables[i] for i in range(len(tmp_all_lables)) if cur_f_cut[i] >= 0.5]
                set_B = [tmp_all_lables[i] for i in range(len(tmp_all_lables)) if cur_f_cut[i] < 0.5]
                split_ent = (get_info_entropy(set_A) * len(set_A) + get_info_entropy(set_B) * len(set_B)) / len(tmp_all_lables)
                cur_ent = ori_ent - split_ent
                if cur_ent > opt_ent:
                    opt_cut = c
                    opt_ent = cur_ent
        else:
            raise IOError
        cur_f_bin = (cur_f.copy() > opt_cut) * 1
        feat_df.loc[:, col] = cur_f_bin
        opt_cuts.append([col, opt_cut])
    cuts_df = pd.DataFrame(opt_cuts, columns=['feature', 'opt cut-off'])
    return feat_df, cuts_df


if __name__ == "__main__":
    feat_df = pd.read_csv('', encoding='utf8',low_memory=False)
    # columns = feat_df.columns.tolist()[3:]
    # valid_cols = ['Age', 'Tumor size']
    valid_cols = ['Sex', 'Age', 'Tumor Grade', 'Side', 'Pathology Type', 'N stage', 'T stage', 'Surgery', 'Tumor size', 'Married status']
        
    all_labels = []
    for i in range(feat_df.shape[0]):
        if feat_df.loc[i, "M stage"] == "M0":
            all_labels.append(0)
        elif feat_df.loc[i, "M stage"] == "M1":
            all_labels.append(1)
        else:
            raise ValueError('wrong label')
    
    all_labels = np.array(all_labels)
    assert len(all_labels) == feat_df.shape[0]

    idxes = [i for i in range(feat_df.shape[0])]
    feat_df, cuts_df = dis_by_info_entropy(feat_df, all_labels, valid_cols, idxes)
    feat_df.to_csv('', index=None, encoding='utf8')
    cuts_df.to_csv('', index=None, encoding='utf8')

            
        
        
        