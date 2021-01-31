import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import argparse
import sys
import os


parser = argparse.ArgumentParser()

parser.add_argument('-split_data', action='store_true', default=False)
parser.add_argument('-split_size', action='store_true', default=False)
parser.add_argument('-prepare', action='store_true', default=False)

parser.add_argument('-project', type=str,
                    default='qt')
parser.add_argument('-data', type=str,
                    default='k')
parser.add_argument('-train', type=int, default=10000)
parser.add_argument('-test', type=int, default=2000)

def prepare(args, key=False):
    train_pkl = './data/{}/{}_train_dextend.pkl'.format(args.project, args.project)
    test_pkl = './data/{}/{}_test_dextend.pkl'.format(args.project, args.project)
    
    csv_data = './data/{}/{}_all.csv'.format(args.project, args.project)
    train_csv = './data/{}/ok_train.csv'.format(args.project)
    test_csv = './data/{}/ok_test.csv'.format(args.project)

    train_data = pickle.load(open(train_pkl, 'rb'))
    test_data = pickle.load(open(test_pkl, 'rb'))
    train_ids, _, _, _ = train_data 
    test_ids, _, _, _ = test_data 
    data = pd.read_csv(csv_data)

    new_train_data = None
    # new_train_data = data[data['commit_id'].isin(train_ids)]

    for _id in train_ids:
        if new_train_data is None:
            new_train_data = data[data['commit_id'] == _id]
        else:
            new_train_data = new_train_data.append(data[data['commit_id'] == _id], ignore_index=True)

    print(new_train_data.head())
    print(train_ids[:5])
    new_train_data.to_csv(train_csv, index=key)
    

    new_test_data = None

    for _id in test_ids:
        if new_test_data is None:
            new_test_data = data[data['commit_id'] == _id]
        else:
            new_test_data = new_test_data.append(data[data['commit_id'] == _id], ignore_index=True)

    new_test_data.to_csv(test_csv, index=key)

def get_features(data):
    # return the features of yasu data
    return data[:, 4:32]

def replace_value_dataframe(df):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean())
    return df.values

def prepare_ftr(pkl_data, csv_data, save_data, key=False):
    data = pickle.load(open(pkl_data, 'rb'))
    ids, _, _, _ = data 
    data = pd.read_csv(csv_data)
    # new_data = data[data['commit_id'].isin(ids)]
    new_data = None

    for _id in ids:
        if new_data is None:
            new_data = data[data['commit_id'] == _id]
        else:
            new_data = new_data.append(data[data['commit_id'] == _id], ignore_index=True)

    print(len(ids))
    # print(new_data.head())
    # print(ids[:5])
    new_data = replace_value_dataframe(df=new_data)
    ftr = np.array(get_features(new_data).tolist(), dtype=float)
    ftr = normalize(ftr, axis=0)
    print(ftr.shape)
    # ftr = torch.tensor(ftr)
    # print(ftr)
    with open(save_data, 'wb') as f:
        pickle.dump(ftr, f)

# def split_data(args):
#     for year in range(2010,2020):
#         k_feature = pd.read_csv("data/{}/{}/{}_{}_feature.csv".format(args.project, year, args.project, args.data))

#         num = int(len(k_feature) * 0.8)
#         k_feature[:num].to_csv('data/{}/{}/{}_train.csv'.format(args.project, year, args.data), index=False)
#         k_feature[num:].to_csv('data/{}/{}/{}_test.csv'.format(args.project, year, args.data), index=False)


def split_list(args, data, size=False, v=False):
    idx1 = args.train + args.test
    idx2 = args.test
    idx = int(len(data)*0.8)
    idx3 = int(len(data)*0.1)

    if size:
        return data[-idx1:-idx2], data[-idx2:]
    if v:
        return data[:idx-idx3], data[idx-idx3:idx], data[idx:]
    return data[:idx], data[idx:]

def split_size(args):
    for size in ['10k', '20k', '30k', '40k', '50k']:
        k_feature = pd.read_csv("data/{}/{}/{}_{}_feature.csv".format(args.project, size, args.project, args.data))

        if size == '50k':
            args.train = 50000
        elif size == '40k':
            args.train = 40000
        elif size == '30k':
            args.train = 30000
        elif size == '20k':
            args.train = 20000
        elif size == '10k':
            args.train = 10000

        train, test = split_list(args, k_feature, size=True)
        train.to_csv('data/{}/{}/{}_train.csv'.format(args.project, size, args.data), index=False)
        test.to_csv('data/{}/{}/{}_test.csv'.format(args.project, size, args.data), index=False)

def split_data(args):
    k_feature = pd.read_csv("data/{}/{}_{}_feature.csv".format(args.project, args.project, args.data))
    # k_feature = k_feature.sort_values(by=['author_date'])
    # train, valid, test = split_list(args, k_feature, v=True)
    train, test = split_list(args, k_feature)

    train.to_csv('data/{}/{}_train.csv'.format(args.project, args.data), index=False)
    test.to_csv('data/{}/{}_test.csv'.format(args.project, args.data), index=False)
    # valid.to_csv('data/{}/{}_valid.csv'.format(args.project, args.data), index=False)

def sp_size():
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    cmd = 'python prepare_data.py -data hk -split_size -project {}'

    for project in projects:
        print(cmd.format(project))
        os.system(cmd.format(project))

def sp():
    projects = ['qt', 'openstack', 'jdt', 'platform', 'gerrit', 'go']
    cmd = 'python prepare_data.py -data hk -split_data -project {}'

    for project in projects:
        print(cmd.format(project))
        os.system(cmd.format(project))

if __name__ == "__main__":

    # # prepare(pkl_data, csv_data, save_data)
    # # prepare(pkl_data, csv_data, save_data, key=True)
    # prepare_ftr(pkl_data, csv_data, save_data, key=True)


    # pkl_data = './{}/{}_test_dextend.pkl'.format(project, project)
    # # csv_data = './{}/{}_all.csv'.format(project, project)
    # # save_data = './{}/test.csv'.format(project)
    # save_data = './{}/test.pkl'.format(project)

    # # prepare(pkl_data, csv_data, save_data)
    # # prepare(pkl_data, csv_data, save_data, key=True)
    # prepare_ftr(pkl_data, csv_data, save_data, key=True)
    args = parser.parse_args()

    if len(sys.argv) < 2:
        print('Usage: python gettit_extraction.py [model]')
    elif args.split_data:
        split_data(args)
    elif args.prepare:
        prepare(args)
    elif args.split_size:
        split_size(args)
    else:
        sp()
    


