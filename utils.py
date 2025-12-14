from sklearn import preprocessing
from scipy import stats
import os
import random
import numpy as np 

from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score, recall_score, precision_score, precision_recall_curve


def search_best(data):
    if isinstance(data, list):
        temp = list(data)
    maximum = max(temp)
    index = temp.index(max(temp))
    return index, maximum
    
def standard(data):
    """Standard data with 0 mean and 1 variance at axis = 0"""
    scaled_data = preprocessing.scale(data)
    return scaled_data

def normalize(data):
    """Normalize data with norm-1 at axis=1."""
    normalized_data = preprocessing.normalize(data, norm='l1')
    return normalized_data

def min_max(data):
    """Convert data into the interval-[0,1]"""
    from sklearn.preprocessing import MinMaxScaler
    mm_scaler = MinMaxScaler()
    scaled_data = mm_scaler.fit_transform(data)
    return scaled_data

def cross_val(data, fold_num = 10):
    """the data splitting for 10-fold cross-validation by t test, and return the list containing all ten validation data!"""
    print('the length of data need to be splitted into 10-folder is {}'.format(data.shape[0]))
    train_npy = ''
    if os.path.exists(train_npy):
        if isinstance(train_npy, str):
            series = np.load(train_npy, allow_pickle=True)
            series = [list(s) for s in series]
        else:
            raise TypeError('The input about train npy has wrong type!')
    else:
        size = data.shape[0]
        seq = list(np.arange(size))
        series = list()
        for i in np.arange(fold_num-1):
            value = 0.05
            while value < 0.2:
                val_list = random.sample(list(seq), int(size*0.1))
                t_list = list(set(list(np.arange(size)))-set(val_list))
                train = data.iloc[t_list,1:-1]
                test = data.iloc[val_list, 1:-1]
                p = stats.ttest_ind(train, test, equal_var=False)
                # print(p)
                value = np.min(p[1])
                print('The sequence-{} with p-{} larger than 0.2 is reasonable!'.format(i, value))
            series.append(val_list)
            print('the length of {}-folder validation is {}!!'.format(i+1, len(val_list)))
            seq = sorted(list(set(seq)-set(val_list)))
            print('the remaining length of data is {}'.format(len(seq)))
        series.append(seq)
        np.save(train_npy, series)
    return series

def plot_roc(truth_y, prob_predicted_y, outdir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(truth_y, prob_predicted_y)
    roc_auc = auc(fpr, tpr)
    plt.figure(0).clf()
    plt.plot(fpr, tpr, 'b', label=r'$AUC_{ml model} = %0.4f$' % roc_auc)
    plt.title('ROC curves comparision')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    lg = plt.legend(loc='lower right', borderaxespad=1.)
    lg.get_frame().set_edgecolor('k')
    plt.grid(True, linestyle='-')
    plt.savefig(outdir) 
    return roc_auc

def all_argmax(arr):
   max_val = np.max(arr)
   return np.where(arr == max_val)[0]

def all_argmin(arr):
   min_val = np.min(arr)
   return np.where(arr == min_val)[0]
def threshold_roc(predict_proba, y_true, method: str = 'youden'):
    y_true = np.array(y_true)
    predict_proba = np.array(predict_proba)
    fpr, tpr, thresholds = roc_curve(y_true, predict_proba)
    se, sp = tpr, 1 - fpr
    roc_auc = roc_auc_score(y_true, predict_proba)
    if method == 'youden':
        criteria = tpr - fpr
        optimal_idxes = all_argmax(criteria)
    elif method == 'er':
        # here the er and sqrt(er) is Equivalent in this optimization problems. we use later one.
        criteria = (1 - se) ** 2 + (1 - sp) ** 2
        optimal_idxes = all_argmin(criteria)
    elif method == 'cz':
        criteria = se * sp
        optimal_idxes = all_argmax(criteria)
    elif method == 'iu':
        criteria = abs(se - roc_auc) + abs(sp - roc_auc)
        optimal_idxes = all_argmin(criteria)

    # if multi optimal exists, choose middle one.
    if len(optimal_idxes) == 1:
        optimal_idx = optimal_idxes[0]
    else:
        se1, sp1, thresholds1 = se[optimal_idxes], sp[optimal_idxes], thresholds[optimal_idxes]
        # this method is the second criteria of the iu method, we apply it to all methods for the supplements.
        criteria1 = abs(se1 - sp1)
        optimal_idxes1 = all_argmin(criteria1)
        optimal_idx = optimal_idxes1[len(optimal_idxes1) // 2]
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def get_matrix(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    assert len(preds) == len(labels) # 判断样本数量一致

    opt_theta = threshold_roc(preds[:, 1], labels)
    preds_max = (preds[:, 1] > opt_theta) * 1
    c_matrix = confusion_matrix(labels, preds_max) #二分类混淆矩阵
    auc_value = roc_auc_score(labels, preds[:, 1])
    precision, recall, threshold = precision_recall_curve(labels, preds[:, 1])
    auprc = auc(recall, precision)

    tp = c_matrix[1,1]
    fn = c_matrix[1,0]
    fp = c_matrix[0,1]
    tn = c_matrix[0,0]
    acc = round(((tp + tn) / (tp + tn + fn + fp)), 4)
    sens = round((tp / (tp + fn)), 4)
    spec = round((tn / (tn + fp)), 4)
    PPV = round((tp / (tp + fp)), 4)
    NPV = round((tn / (tn + fn)), 4)
    f1 = round((2 * tp / (2 * tp + fn + fp)), 4)
    return auc_value, acc, sens, spec, PPV, NPV, f1, auprc

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
def get_mode(mode):
    if mode == "svm":
        model = SVC(gamma='auto', class_weight='balanced', probability=True)
    elif mode == "bayes":
        model = GaussianNB()
    elif mode == "random_forest":
        model = RandomForestClassifier(n_estimators=30, max_depth=10, min_samples_leaf=5, class_weight='balanced', bootstrap=True)
    elif mode == "decision_tree":
        model = DecisionTreeClassifier(max_depth=20, min_samples_split=10, min_samples_leaf=10, class_weight='balanced')
    elif mode == "neutral_network":
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 10), random_state=1)
    elif mode == "xgboost":
        model = XGBClassifier(max_depth=10, n_estimators=30)
    elif mode == "logistic":
        model = LogisticRegression(class_weight='balanced', random_state=0)
    else:
        raise KeyError(f"wrong model indicator with {mode}")
    return model