import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVR
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import normalize

def load_file(path_file):
    lines = list(open(path_file, 'r', encoding='utf8', errors='ignore').readlines())
    lines = [l.strip() for l in lines]
    return lines

def eval(labels, predicts, thresh=0.5):
    TP, FN, FP, TN = 0, 0, 0, 0
    for lable, predict in zip(labels, predicts):
        # print(predict)
        if predict >= thresh and lable == 1:
            TP += 1
        if predict >= thresh and lable == 0:
            FP += 1
        if predict < thresh and lable == 1:
            FN += 1
        if predict < thresh and lable == 0:
            TN += 1
    
    # print(TP, FN, FP, TN)
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)

        A = (TP+TN)/len(labels)
        E = FP/(TP+FP)

        print('Test data at Threshold %.2f -- Accuracy: %.2f, False Positives: %.2f, Precision: %.2f, Recall: %.2f'%(thresh, A, E, P, R))
    except Exception:
        # division by zero
        pass

def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    return acc, prc, rc, f1, auc_

def loading_variable(pname):
    f = open('../variables/' + pname + '.pkl', 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df.values


def loading_data(project):
    train, test = loading_variable(project + '_train'), loading_variable(project + '_test')
    dictionary = (loading_variable(project + '_dict_msg'), loading_variable(project + '_dict_code'))
    return train, test, dictionary


def get_features(data):
    # return the features of yasu data
    return data[:, 5:32]


def get_ids(data):
    # return the labels of yasu data
    return data[:, 1:2].flatten().tolist()


def get_label(data):
    data = data[:, 3:4].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data


def load_df_yasu_data(path_data):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes, new_ids, new_labels, new_features = list(), list(), list(), list()
    cnt_noexits = 0
    for i in range(0, len(ids)):
        try:
            indexes.append(i)
        except FileNotFoundError:
            print('File commit id no exits', ids[i], cnt_noexits)
            cnt_noexits += 1
    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    features = normalize(features, axis=0)
    # print(features)
    return (ids, np.array(labels), features)


def load_yasu_data(project):
    train_path_data = './{}/train.csv'.format(project)
    test_path_data = './{}/test.csv'.format(project)
    train, test = load_df_yasu_data(train_path_data), load_df_yasu_data(test_path_data)
    return train, test



def baseline_algorithm(train, test, algorihm):
    _, y_train, X_train = train
    _, y_test, X_test = test
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    if algorihm == 'svr_rbf':
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        y_pred = model.fit(X_train, y_train).predict(X_test)
    elif algorihm == 'svr_poly':
        model = SVR(kernel='poly', C=1e3, degree=2)
        y_pred = model.fit(X_train, y_train).predict(X_test)
    elif algorihm == 'lr':
        model = LogisticRegression()
        y_pred = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    elif algorihm == 'svm':
        model = svm.SVC(probability=True).fit(X_train, y_train)
        y_pred = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    elif algorihm == 'ridge':
        model = linear_model.Ridge()
        y_pred = model.fit(X_train, y_train).predict(X_test)
    else:
        print('You need to give the correct algorithm name')

    acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    eval(y_test, y_pred, thresh=0.1)
    eval(y_test, y_pred, thresh=0.2)
    eval(y_test, y_pred, thresh=0.3)
    eval(y_test, y_pred, thresh=0.4)
    eval(y_test, y_pred, thresh=0.5)
    eval(y_test, y_pred, thresh=0.6)
    eval(y_test, y_pred, thresh=0.7)
    eval(y_test, y_pred, thresh=0.8)
    eval(y_test, y_pred, thresh=0.9)




if __name__ == '__main__':
    # project = 'openstack'
    project = 'eclipse'
    # load training/testing data
    train, test = load_yasu_data(project=project)
    baseline_algorithm(train=train, test=test, algorihm='lr')
    # baseline_algorithm(train=train, test=test, algorihm='svr_rbf')
    # baseline_algorithm(train=train, test=test, algorihm='svr_poly')
    # baseline_algorithm(train=train, test=test, algorihm='svm')
    # baseline_algorithm(train=train, test=test, algorihm='ridge')
