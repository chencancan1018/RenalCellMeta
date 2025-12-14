import os
import csv
import random
import pickle
import numpy as np 
import pandas as pd 
from sklearn import metrics

import utils
from MachineLearning import ModelSelect

def model_test():  
    data = pd.read_csv('', encoding='utf8')
    height, weight = data.shape
    print('the  shape of our data is ', data.shape)
    # columns = data.columns.tolist()
    train_x = data.iloc[:int(height*0.8), 1:-1]; train_y = data.iloc[:int(height*0.8), -1].tolist()
    test_x = data.iloc[int(height*0.8):, 1:-1]; test_y = data.iloc[int(height*0.8):, -1].tolist()
    print(len(train_x), len(train_y))
    print(len(test_x), len(test_y))

    assert len(train_x) == len(train_y) and len(test_x) == len(test_y)

    # C = [0.4,0.8,1.0,1.5,2,3]
    # Kern = ['linear', 'rbf', 'sigmoid']
    # ml = SVC(C=c, kernel=k, gamma='auto', probability=True)   
    # ne =[10, 50, 100, 500, 1000, 2000, 4000, 6000]
    # max_dep = [10, 20, 50, 80, 100]
    # min_sam_spl = [10, 100, 500, 1000, 3000]
    # min_sam_le = [10, 50,100,300, 500, 1000, 3000]
    # from sklearn.ensemble import RandomForestClassifier
    # ml = RandomForestClassifier(n_estimators=m0, max_depth=m1, min_samples_split=m2, min_samples_leaf=m3)
    # modelname = ['svm', 'bayes', 'decision_tree', 'random_forest', 'neutral_network']
    # depth = [10, 20,30, 40, 50,60,70, 80,90, 100, 200,300, 400,500]
    # width = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # d=40; w = 10
    # from sklearn.neural_network import MLPClassifier
    # ml = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(d, w), random_state=1)

    md = [5,10, 20,30,40,50,60,70,80,90,100]
    num = [5,10,15,20,50,100,150,200,300,400,500,600,700,800,900,1000,1200]
    for n in num:
        d = 5;n=10
        print('Start to train neutral networh with depth-{} and num estimators-{}......'.format(d, n))
        from xgboost import XGBClassifier
        ml = XGBClassifier(max_depth=d, learning_rate=1, n_estimators=n)#, silent=True)#, objective='binary:logistic')
        clf = ml.fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        prob_pred_y = clf.predict_proba(test_x)[:,1]
        auc = utils.plot_roc(test_y, prob_pred_y, os.path.join('/data/chencc/data/bladder/code/save/test/','XGBClassifier.jpg'))
        print('The auc of the XGBClassifier model with depth-{} and num estimators-{} is {}......'.format(d, n, auc))
        accuracy = metrics.accuracy_score(test_y, pred_y, normalize=True)
        print('XGBClassifier model with depth-{} and num estimators-{}--the acc is {}......'.format(d, n,accuracy))


def main_cross_val():  
    data = pd.read_csv('', encoding='utf8')
    save = ''
    # height, weight = data.shape
    print('the  shape of our data is ', data.shape)
    print('>>>>>>>>>>>The save path: ', save)
    # columns = data.columns.tolist()

    # series = utils.cross_val(data.shape[0])
    series = utils.cross_val(data)
    # series = utils.cross_val(data.shape[0], 5) #if 5-fold cross validation
    assert len(series) == 10, 'The number of cross validation is unmatched!'
    seq = list(np.arange(data.shape[0]))

    # Standard = False; Normalize = False; Min_max = True
    modelname = ['svm', 'bayes', 'decision_tree', 'random_forest', 'neutral_network', 'xgboost']
    auc_dict = dict()
    for index in range(len(modelname)):
        print('Start to train model: ', modelname[index])
        ml = ModelSelect(modelname[index])
        model = ml.dismatch()
        outdir1 = os.path.join(save, modelname[index])
        if not os.path.exists(outdir1):
            os.mkdir(outdir1)

        auc_list = list()
        for i in np.arange(len(series)):
            test_seq = series[i]
            train_seq = sorted(list(set(seq)-set(test_seq)))
            train_x = data.iloc[train_seq, 1:-1]; train_y = data.iloc[train_seq, -1].tolist()
            test_x = data.iloc[test_seq, 1:-1]; test_y = data.iloc[test_seq, -1].tolist()
            print(len(train_x), len(train_y))
            print(len(test_x), len(test_y))

            assert len(train_x) == len(train_y) and len(test_x) == len(test_y), 'the length of features and label is not equal!'
            outdir2 = os.path.join(outdir1, 'val_'+str(i+1))
            if not os.path.exists(outdir2):
                os.mkdir(outdir2)

            # if Standard:
            if modelname[index] == 'neutral_network':
                train_x = utils.standard(train_x)
                test_x = utils.standard(test_x)
                print('The preprocessing is standardization!')
                # outdir3 = os.path.join(outdir2, 'standard')
                # if not os.path.exists(outdir3):
                #     os.mkdir(outdir3)
            # elif Normalize:
            #     train_x = utils.normalize(train_x)
            #     test_x = utils.normalize(test_x)
            #     print('The preprocessing is normalization')
            #     outdir3 = os.path.join(outdir2, 'normalize')
            #     if not os.path.exists(outdir3):
            #         os.mkdir(outdir3)
            # elif Min_max:
            elif modelname[index] == 'svm':
                train_x = utils.min_max(train_x)
                test_x = utils.min_max(test_x)
                print('The preprocessing is min max scaler with data in [0,1]')
                # outdir3 = os.path.join(outdir2, 'min_max')
                # if not os.path.exists(outdir3):
                #     os.mkdir(outdir3)
            else:
                pass
                # outdir3 = os.path.join(outdir2, 'unprepro')
                # if not os.path.exists(outdir3):
                #     os.mkdir(outdir3)

            clf = model.fit(train_x, train_y)

            pred_y = clf.predict(test_x)
            prob_pred_y = clf.predict_proba(test_x)[:,1]
            auc = utils.plot_roc(test_y, prob_pred_y, os.path.join(outdir2, '{}_auc.jpg'.format(modelname[index])))
            auc_list.append(float("{0:.5f}".format(auc)))
            print('The auc of the current model {} with val-{} is {}!'.format(modelname[index], i+1, auc))
            
            f = open(os.path.join(outdir2, '{}.pickle'.format(modelname[index])), 'wb') ##Save model which has been trained
            pickle.dump(clf, f)
            f.close()

            pred_info = dict() ##Save the predicted results by ml model
            # pred_info['index'] = list(patients) #The index of patients is needed
            pred_info['patient'] = list(data.iloc[test_seq, 0])
            pred_info['label'] = list(test_y)
            pred_info['predict'] = list(pred_y)
            pred_info['probability'] = list(prob_pred_y)
            df = pd.DataFrame(pred_info)
            df.to_csv(outdir2+'/'+'predict_result.csv', index=None, header=None, encoding='utf8')     
        best_val, best_auc = utils.search_best(auc_list) 
        auc_dict[modelname[index]] = auc_list
        print('For the ml model {}: the best auc-{} is corresponding to the validation-{}'.format(modelname[index], best_auc, best_val+1))
    df = pd.DataFrame(auc_dict)
    # df = pd.DataFrame(df.values.T, index=df.columns,columns=df.index)
    df.to_csv(save+'')
        

if __name__ == '__main__':
    main_cross_val()
