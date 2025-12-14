import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import os.path as osp
import utils

def get_dev(dir):
    train_df = pd.read_csv(os.path.join(dir, 'train_pred.csv'), encoding='utf8')
    val_df = pd.read_csv(os.path.join(dir, 'val_pred.csv'), encoding='utf8')
    preds = []
    for i in range(train_df.shape[0]):
        preds.append(train_df.iloc[i, :].tolist())
    for i in range(val_df.shape[0]):
        preds.append(val_df.iloc[i, :].tolist())
    preds = sorted(preds, key=lambda l:l[0])
    return pd.DataFrame(preds, columns=['pid','0-prob','1-prob','label'])

def get_test(dir):
    return pd.read_csv(os.path.join(dir, 'test_pred.csv'), encoding='utf8')

def get_ext(dir):
    return pd.read_csv(os.path.join(dir, 'ext_pred.csv'), encoding='utf8')

def ensem(df1, df2, df3, df4, df5):
    assert np.all(np.array(df1.shape)) == np.all(np.array(df2.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df3.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df4.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df5.shape))
    res = []
    for i in range(df1.shape[0]):
        pid = df1.iloc[i, 0]
        pred_0 = np.mean([df1.iloc[i, 1], df2.iloc[i, 1], df3.iloc[i, 1], df4.iloc[i, 1], df5.iloc[i, 1]])
        pred_1 = 1 - pred_0
        label = df1.iloc[i, -1]
        res.append([pid, pred_0, pred_1, label])
    return pd.DataFrame(res, columns=['pid','0-prob','1-prob','label'])

def ensem10(df1, df2, df3, df4, df5, df6, df7, df8, df9, df10):
    assert np.all(np.array(df1.shape)) == np.all(np.array(df2.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df3.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df4.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df5.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df6.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df7.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df8.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df9.shape))
    assert np.all(np.array(df1.shape)) == np.all(np.array(df10.shape))
    res = []
    for i in range(df1.shape[0]):
        pid = df1.iloc[i, 0]
        pred_0 = np.mean([
            df1.iloc[i, 1], df2.iloc[i, 1], df3.iloc[i, 1], df4.iloc[i, 1], df5.iloc[i, 1],
            df6.iloc[i, 1], df7.iloc[i, 1], df8.iloc[i, 1], df9.iloc[i, 1], df10.iloc[i, 1],
        ])
        pred_1 = 1 - pred_0
        label = df1.iloc[i, -1]
        res.append([pid, pred_0, pred_1, label])
    return pd.DataFrame(res, columns=['pid','0-prob','1-prob','label'])

if __name__ == "__main__":



    input_dir = ""

    modes = ["bayes", "svm", "decision_tree", "random_forest", "neutral_network", "xgboost", "logistic"]

    outdir = osp.join(input_dir, "ensemble")
    # outdir = osp.join(os.path.dirname(input_dir), "ensemble")
    os.makedirs(outdir, exist_ok=True)

    for mode in modes:
        cur_outdir = osp.join(outdir, mode)
        os.makedirs(cur_outdir,exist_ok=True)
        
        fds = ["fold1fordev", "fold2fordev", "fold3fordev", "fold4fordev", "fold5fordev"]
        tmpdir1 = osp.join(input_dir, fds[0], mode)
        tmpdir2 = osp.join(input_dir, fds[1], mode)
        tmpdir3 = osp.join(input_dir, fds[2], mode)
        tmpdir4 = osp.join(input_dir, fds[3], mode)
        tmpdir5 = osp.join(input_dir, fds[4], mode)
        # tmpdir6 = osp.join(input_dir1, fds[0], mode)
        # tmpdir7 = osp.join(input_dir1, fds[1], mode)
        # tmpdir8 = osp.join(input_dir1, fds[2], mode)
        # tmpdir9 = osp.join(input_dir1, fds[3], mode)
        # tmpdir10 = osp.join(input_dir1, fds[4], mode)

        dev1_df = get_dev(tmpdir1)
        dev2_df = get_dev(tmpdir2)
        dev3_df = get_dev(tmpdir3)
        dev4_df = get_dev(tmpdir4)
        dev5_df = get_dev(tmpdir5)
        # dev6_df = get_dev(tmpdir6)
        # dev7_df = get_dev(tmpdir7)
        # dev8_df = get_dev(tmpdir8)
        # dev9_df = get_dev(tmpdir9)
        # dev10_df = get_dev(tmpdir10)
        dev_df = ensem(dev1_df, dev2_df, dev3_df, dev4_df, dev5_df)
        # dev_df = ensem10(dev1_df, dev2_df, dev3_df, dev4_df, dev5_df, dev6_df, dev7_df, dev8_df, dev9_df, dev10_df)
        dev_df.to_csv(osp.join(cur_outdir,"dev_pred.csv"), index=None, encoding='utf8')

        test1_df = get_test(tmpdir1)
        test2_df = get_test(tmpdir2)
        test3_df = get_test(tmpdir3)
        test4_df = get_test(tmpdir4)
        test5_df = get_test(tmpdir5)
        # test6_df = get_test(tmpdir6)
        # test7_df = get_test(tmpdir7)
        # test8_df = get_test(tmpdir8)
        # test9_df = get_test(tmpdir9)
        # test10_df = get_test(tmpdir10)
        test_df = ensem(test1_df, test2_df, test3_df, test4_df, test5_df)
        # test_df = ensem10(test1_df, test2_df, test3_df, test4_df, test5_df, test6_df, test7_df, test8_df, test9_df, test10_df)
        test_df.to_csv(osp.join(cur_outdir,"test_pred.csv"), index=None, encoding='utf8')

        ext1_df = get_ext(tmpdir1)
        ext2_df = get_ext(tmpdir2)
        ext3_df = get_ext(tmpdir3)
        ext4_df = get_ext(tmpdir4)
        ext5_df = get_ext(tmpdir5)
        # ext6_df = get_ext(tmpdir6)
        # ext7_df = get_ext(tmpdir7)
        # ext8_df = get_ext(tmpdir8)
        # ext9_df = get_ext(tmpdir9)
        # ext10_df = get_ext(tmpdir10)
        ext_df = ensem(ext1_df, ext2_df, ext3_df, ext4_df, ext5_df)
        # ext_df = ensem10(ext1_df, ext2_df, ext3_df, ext4_df, ext5_df, ext6_df, ext7_df, ext8_df, ext9_df, ext10_df)
        ext_df.to_csv(osp.join(cur_outdir,"ext_pred.csv"), index=None, encoding='utf8')

        dev_auc, dev_acc, dev_sens, dev_spec, dev_ppv, dev_npv, dev_f1, dev_auprc = utils.get_matrix(dev_df.iloc[:, 1:-1].values, dev_df.iloc[:, -1].values)
        test_auc, test_acc, test_sens, test_spec, test_ppv, test_npv, test_f1, test_auprc = utils.get_matrix(test_df.iloc[:, 1:-1].values, test_df.iloc[:, -1].values)
        ext_auc, ext_acc, ext_sens, ext_spec, ext_ppv, ext_npv, ext_f1, ext_auprc = utils.get_matrix(ext_df.iloc[:, 1:-1].values, ext_df.iloc[:, -1].values)
        print(f"For {mode} and val, val acc: {dev_acc:.4f}, val auc: {dev_auc:.4f}, val_f1: {dev_f1: .4f}, {dev_sens, dev_spec, dev_ppv, dev_npv, dev_auprc}")
        print(f"For {mode} and test, test acc: {test_acc:.4f}, test auc: {test_auc:.4f}, test_f1: {test_f1: .4f}, {test_sens, test_spec, test_ppv, test_npv, test_auprc}")
        print(f"For {mode} and ext, ext acc: {ext_acc:.4f}, ext auc: {ext_auc:.4f}, ext_f1: {test_f1: .4f}, {ext_sens, ext_spec, ext_ppv, ext_npv, ext_auprc}")

