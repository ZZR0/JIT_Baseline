import pickle
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import normalize

def prepare(pkl_data, csv_data, save_data, key=False):
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

    print(new_data.head())
    print(ids[:5])
    new_data.to_csv(save_data, index=key)

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

def split_data(project):
    # k_feature = pd.read_csv("eclipse/k_feature.csv")
    # num = int(len(k_feature) * 0.8)
    # k_feature[:num].to_csv('eclipse/train.csv', index=False)
    # k_feature[num:].to_csv('eclipse/test.csv', index=False)

    # k_feature = pd.read_csv("go/k_feature.csv")
    k_feature = pd.read_csv("{}/m_feature.csv".format(project))

    num = int(len(k_feature) * 0.8)
    k_feature[:num].to_csv('{}/train.csv'.format(project), index=False)
    k_feature[num:].to_csv('{}/test.csv'.format(project), index=False)


if __name__ == "__main__":
    # project = 'openstack'
    # pkl_data = './{}/{}_train_dextend.pkl'.format(project, project)
    # # csv_data = './{}/{}_all.csv'.format(project, project)
    # csv_data = './{}/{}.csv'.format(project, project)
    # # save_data = './{}/train.csv'.format(project)
    # save_data = './{}/train.pkl'.format(project)

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

    split_data('eclipse')


