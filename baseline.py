import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.svm import SVR
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import argparse
import sys
import time
from LR import LR
import torch.nn as nn
import torch
import math
import random
import copy
from tqdm import tqdm


from torch.functional import Tensor
from DBN import DBN


parser = argparse.ArgumentParser()

parser.add_argument('-project', type=str,
                    default='qt')
parser.add_argument('-data', type=str,
                    default='k')
parser.add_argument('-algorithm', type=str,
                    default='lr')
parser.add_argument('-drop', type=str,
                    default='')
parser.add_argument('-only', type=str,
                    default='')

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


def mini_batches_update(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X, mini_batch_Y = shuffled_X[indexes], shuffled_Y[indexes]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    # f1 = 2 * prc * rc / (prc + rc)
    f1 = 0
    return acc, prc, rc, f1, auc_


def loading_variable(pname):
    f = open('../variables/' + pname + '.pkl', 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    if args.drop:
        df = df.drop(columns=[args.drop])
    elif args.only:
        df = df[['Unnamed: 0','_id','date','bug','__',args.only]]
        # df = df[["Unnamed: 0","commit_id","author_date","bugcount","fixcount",args.only]]
    else:
        df = df
        # df = df[["Unnamed: 0","commit_id","author_date","bugcount","fixcount",'nf']]
        # df = df[['Unnamed: 0','_id','date','bug','__', 'la']]
        # bns,lbs,ats
    return df.values


def get_features(data):
    # return the features of yasu data
    # return data.take([7, 12], 1)
    return data[:, 5:19]


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
    indexes = list()
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
    # features = normalize(features, axis=0)
    # print(features)
    return (ids, np.array(labels), features)


def load_yasu_data(args):
    train_path_data = 'data/{}/{}_train.csv'.format(args.project, args.data)
    test_path_data = 'data/{}/{}_test.csv'.format(args.project, args.data)
    train, test = load_df_yasu_data(train_path_data), load_df_yasu_data(test_path_data)
    return train, test


def DBN_JIT(train_features, train_labels, test_features, test_labels, hidden_units=[20, 12, 12], num_epochs_LR=200):
    # training DBN model
    #################################################################################################
    starttime = time.time()
    # dbn_model = DBN(visible_units=train_features.shape[1],
    #                 hidden_units=hidden_units,
    #                 use_gpu=False)
    # dbn_model.train_static(train_features, train_labels, num_epochs=10)
    # # Finishing the training DBN model
    # # print('---------------------Finishing the training DBN model---------------------')
    # # using DBN model to construct features
    # DBN_train_features, _ = dbn_model.forward(train_features)
    # DBN_test_features, _ = dbn_model.forward(test_features)
    # DBN_train_features = DBN_train_features.numpy()
    # DBN_test_features = DBN_test_features.numpy()

    # train_features = np.hstack((train_features, DBN_train_features))
    # test_features = np.hstack((test_features, DBN_test_features))


    if len(train_labels.shape) == 1:
        num_classes = 1
    else:
        num_classes = train_labels.shape[1]
    # lr_model = LR(input_size=hidden_units, num_classes=num_classes)
    lr_model = LR(input_size=train_features.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(lr_model.parameters(), lr=0.00001)
    steps = 0
    batches_test = mini_batches(X=test_features, Y=test_labels)
    for epoch in range(1, num_epochs_LR + 1):
        # building batches for training model
        batches_train = mini_batches_update(X=train_features, Y=train_labels)
        for batch in batches_train:
            x_batch, y_batch = batch
            if torch.cuda.is_available():
                x_batch, y_batch = torch.tensor(x_batch).cuda(), torch.cuda.FloatTensor(y_batch)
            else:
                x_batch, y_batch = torch.tensor(x_batch).float(), torch.tensor(y_batch).float()

            optimizer.zero_grad()
            predict = lr_model.forward(x_batch)
            loss = nn.BCELoss()
            loss = loss(predict, y_batch)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % 10 == 0:
                pass
                # print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))

    endtime = time.time()
    dtime = endtime - starttime
    print("Train Time: %.8s s" % dtime)  #显示到微秒 

    starttime = time.time()
    y_pred, lables = lr_model.predict(data=batches_test)
    endtime = time.time()
    dtime = endtime - starttime
    print("Eval Time: %.8s s" % dtime)  #显示到微秒 
    return y_pred


def normalize(train, test):
    for i in range(train.shape[1]):
        c_max = np.max(train[:,i])
        c_min = np.min(train[:,i])
        if c_min == c_max:
            continue
        train[:,i] = (train[:,i] - c_min) / (c_max - c_min)
        test[:,i] = (test[:,i] - c_min) / (c_max - c_min)
    
    return train, test


def esb(X_train, X_valid, y_train, algorithm='GradientBoostingClassifier'):
	if algorithm == 'BaggingClassifier':
		clf = BaggingClassifier(KNeighborsClassifier())
	elif algorithm == 'AdaBoostClassifier':
		clf = AdaBoostClassifier()
	elif algorithm == 'GradientBoostingClassifier':
		clf = GradientBoostingClassifier()
	else:
		clf = RandomForestClassifier()

	clf.fit(X_train, y_train)

	pred = clf.predict(X_valid)

	return pred


def train_and_evl(data, label):
    size = int(label.shape[0]*0.1)
    auc_ = []

    for i in range(10):
        idx = size * i
        X_e = data[idx:idx+size]
        y_e = label[idx:idx+size]

        X_t = np.vstack((data[:idx], data[idx+size:]))
        y_t = np.hstack((label[:idx], label[idx+size:]))

        model = LogisticRegression(max_iter=1000).fit(X_t, y_t)
        y_pred = model.predict_proba(X_e)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_e, y_score=y_pred, pos_label=1)
        auc_.append(auc(fpr, tpr))

    return np.mean(auc_)


def select2(X_train, X_test, y_train, y_test):
    best_auc = 0
    best_idx = 0
    pg = []

    for i in range(X_train.shape[1]):
        X_sel = X_train[:,i:i+1]
        X_sel_ = X_test[:,i:i+1]

        auc_ = train_and_evl(X_sel, y_train)

        pg.append((i, auc_, X_sel, X_sel_, [i]))

    for i in range(X_train.shape[1]):
        pg.sort(key=lambda x:x[1], reverse=True)
        next_pg = []
        for p in pg[:X_train.shape[1]]:
            X_sel = np.hstack((p[2], X_train[:,i:i+1]))
            X_sel_ = np.hstack((p[3], X_test[:,i:i+1]))

            auc_ = train_and_evl(X_sel, y_train)
            # print(i, auc_)
            if auc_ > best_auc:
                best_auc = auc_
                best_idx = i
            p_list = copy.deepcopy(p[4])
            p_list.append(i)
            next_pg.append((i, auc_, X_sel, X_sel_, p_list))
            next_pg.append(p)
        pg = next_pg

    # print(best_idx, best_auc)
    pg.sort(key=lambda x:x[1], reverse=True)
    # sel = set([i for i in range(14)])
    sel = set()
    for p in pg[:4]:
        # print(p[4])
        sel = sel.union(set(p[4]))
    print(sel)

    best_p = pg[0]
    X, X_ = None, None

    for i in sel:
        if X is None:
            X = X_train[:,i:i+1]
            X_ = X_test[:,i:i+1]
        else:
            X = np.hstack((X, X_train[:,i:i+1]))
            X_= np.hstack((X_, X_test[:,i:i+1]))

    y = y_train
    # X_test = best_p[3]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred = model.predict_proba(X_)[:, 1]
    return y_pred


def select(X_train, X_test, y_train, y_test):
    best_auc = 0
    best_idx = 0
    pg = []

    for i in range(X_train.shape[1]):
        X_sel = X_train[:,i:i+1]
        X_sel_ = X_test[:,i:i+1]

        auc_ = train_and_evl(X_sel, y_train)

        pg.append((i, auc_, X_sel, X_sel_, [i]))

    for _ in tqdm(range(X_train.shape[1])):
        pg.sort(key=lambda x:x[1], reverse=True)
        next_pg = []
        for p in pg[:4]:
            for i in range(X_train.shape[1]):
                X_sel = np.hstack((p[2], X_train[:,i:i+1]))
                X_sel_ = np.hstack((p[3], X_test[:,i:i+1]))

                auc_ = train_and_evl(X_sel, y_train)
                # print(i, auc_)
                if auc_ > best_auc:
                    best_auc = auc_
                    best_idx = i
                p_list = copy.deepcopy(p[4])
                p_list.append(i)
                next_pg.append((i, auc_, X_sel, X_sel_, p_list))

        if abs(sum([p[1] for p in pg]) - sum([p[1] for p in next_pg])) < 0.01: break

        pg = next_pg

    # print(best_idx, best_auc)
    pg.sort(key=lambda x:x[1], reverse=True)
    # sel = set([i for i in range(14)])
    sel = set()
    for p in pg[:4]:
        # print(p[4])
        sel = sel.union(set(p[4]))
    print(sel)

    best_p = pg[0]
    X, X_ = None, None

    for i in sel:
        if X is None:
            X = X_train[:,i:i+1]
            X_ = X_test[:,i:i+1]
        else:
            X = np.hstack((X, X_train[:,i:i+1]))
            X_= np.hstack((X_, X_test[:,i:i+1]))

    y = y_train
    # X_test = best_p[3]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred = model.predict_proba(X_)[:, 1]
    return y_pred


def baseline_algorithm(train, test, algorithm, hidden_layer_sizes=(20, 20)):
    _, y_train, X_train = train
    _, y_test, X_test = test
    # X_train = np.where(X_train<10, X_train, 10)
    # X_test = np.where(X_test<10, X_test, 10)

    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    # X_train, X_test = normalize(X_train, X_test)
    if algorithm == 'svr_rbf':
        model = SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == 'svr_poly':
        model = SVR(kernel='poly', C=1e3, degree=2).fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm == 'lr':
        starttime = time.time()
        model = LogisticRegression(max_iter=7000).fit(X_train, y_train)
        endtime = time.time()
        dtime = endtime - starttime
        print("Train Time: %.8s s" % dtime)  #显示到微秒 

        starttime = time.time()
        y_pred = model.predict_proba(X_test)[:, 1]
        endtime = time.time()
        dtime = endtime - starttime
        print("Eval Time: %.8s s" % dtime)  #显示到微秒 
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        # print(y_pred)
        # auc_ = train_and_evl(X_train, y_train)
        acc, prc, rc, f1 = 0, 0, 0, 0
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    elif algorithm == 'svm':
        model = svm.SVC(probability=True).fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
    elif algorithm == 'ridge':
        model = linear_model.Ridge().fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm =='mlp':
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000).fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif algorithm =='dbn':
        y_pred = DBN_JIT(X_train, y_train, X_test, y_test)
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    elif algorithm == 'esb':
        y_pred = esb(X_train, X_test, y_train)
    elif algorithm == 'sel':
        starttime = time.time()
        y_pred = select(X_train, X_test, y_train, y_test)
        endtime = time.time()
        dtime = endtime - starttime
        print("Train Time: %.8s s" % dtime)  #显示到微秒 
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))
    elif algorithm == 'sel2':
        starttime = time.time()
        y_pred = select2(X_train, X_test, y_train, y_test)
        endtime = time.time()
        dtime = endtime - starttime
        print("Train Time: %.8s s" % dtime)  #显示到微秒 
    else:
        print('You need to give the correct algorithm name')
        return

    return y_test, y_pred 


def save_result(labels, predicts, path):
    results = []
    for lable, predict in zip(labels, predicts):
        results.append('{}\t{}\n'.format(lable, predict))
    
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(results)


if __name__ == '__main__':
    # baseline_algorithm(train=train, test=test, algorithm='svr_rbf')
    # baseline_algorithm(train=train, test=test, algorithm='svr_poly')
    # baseline_algorithm(train=train, test=test, algorithm='svm')
    # baseline_algorithm(train=train, test=test, algorithm='ridge')
    args = parser.parse_args()

    if len(sys.argv) < 2:
        print('Usage: python gettit_extraction.py [model]')

    # for size in ['10k', '50k']:
    #     print('Size:', size)
    #     args.year = size
    #     train, test = load_yasu_data(args)
        
    #     labels, predicts = baseline_algorithm(train=train, test=test, algorithm=args.algorithm, hidden_layer_sizes=(40,40))

    #     save_path = 'data/{}/{}/{}_{}_{}.result'.format(args.project, size, args.project, size, args.algorithm)
    #     save_result(labels, predicts, save_path)

    if args.algorithm == 'la':
        args.algorithm = 'lr'
        args.only = 'la'
    train, test = load_yasu_data(args)
        
    labels, predicts = baseline_algorithm(train=train, test=test, algorithm=args.algorithm, hidden_layer_sizes=(40,40))

    save_path = 'data/{}/{}_{}.result'.format(args.project, args.project, args.algorithm)
    save_result(labels, predicts, save_path)