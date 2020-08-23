import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle, os, re, operator, gc
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

# load datasets
df_shopinfo = pd.read_csv('../data/shopInfo.txt', delimiter='\t')
df_shopcat = pd.read_csv('../data/shopCategoryInfo.txt', delimiter='\t')
df_trainset = pd.read_csv('../data/trainSampleInfo.txt', delimiter='\t')
df_testset = pd.read_csv('../data/testSampleInfo.txt', delimiter='\t')

df_shopinfo = df_shopinfo.merge(df_shopcat, on='category_id', how='left')
train = df_trainset.merge(df_shopinfo, how='left', on='shop_id')
train['row_id'] = train.index
test = df_testset.merge(df_shopinfo, how='left', on='shop_id')
test['row_id'] = test.index

res = None
train_samples = []
test_samples = []

for build_id in tqdm(train.build_id.unique()):
    sub_train = train[train.build_id == build_id]
    sub_test = test[test.build_id == build_id]

    train_set = []
    for index, row in sub_train.iterrows():
        wifi_dict = {}
        for wifi in row.wifi_infos.split(';'):
            bssid, signal = wifi.split(',')
            wifi_dict[bssid] = int(signal)
        train_set.append(wifi_dict)

    test_set = []
    for index, row in sub_test.iterrows():
        wifi_dict = {}
        for wifi in row.wifi_infos.split(';'):
            bssid, siginal = wifi.split(',')
            wifi_dict[bssid] = int(siginal)
        test_set.append(wifi_dict)

    v = DictVectorizer(sparse=False, sort=False)
    train_set = v.fit_transform(train_set)
    test_set = v.fit_transform(test_set)
    train_set[train_set == 0] = np.NaN
    test_set[test_set == 0] = np.NaN

    sub_train = pd.concat([sub_train.reset_index(), pd.DataFrame(train_set)], axis=1)
    sub_test = pd.concat([sub_test.reset_index(), pd.DataFrame(test_set)], axis=1)

    lbl = LabelEncoder()
    lbl.fit(list(sub_train['shop_id'].values()))
    sub_train['label'] = lbl.transform(list(sub_train['shop_id'].values))
    num_class = sub_train['label'].max() + 1
    feature = [x for x in sub_train.columns if x not in ['label', 'shop_id', 'wifi_infos', 'category_name', 'build_id', 'row_id']]

    params = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'max_depth': 8,
        'eval_metric': 'auc',
        'seed': 0,
        'silent': 1
    }

    xgbtrain = xgb.DMatrix(sub_train[feature], sub_train['label'])
    # xgbtest = xgb.DMatrix(sub_test[feature])
    X = sub_train[feature]
    Y = sub_train['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    xgbtrain = xgb.DMatrix(X_train, Y_train)
    xgbtest = xgb.DMatrix(X_test, Y_test)
    # watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
    model = xgb.train(params, xgbtrain, num_boost_round=100, verbose_eval=False)
    '''
    preds = model.predict(xgbtest)
    for row, pred in enumerate(preds):
        row_id = sub_test['row_id'].iloc[row]
        predSorted = (-pred).argsort()
        for i in range(10):
            test_samples.append({'row_id': row_id, 'shop_id': lbl.inverse_transform(predSorted[i]), 'prob': pred[predSorted[i]]})
            if pred[predSorted[i]] > 0.99:
                break
    '''
    preds = model.predict(xgbtrain)
    for row, pred in enumerate(preds):
        row_id = sub_train['row_id'].iloc[row]
        predSorted = (-pred).argsort()
        for i in range(10):
            train_samples.append({'row_id': row_id, 'shop_id': lbl.inverse_transform(predSorted[i]), 'prob': pred[predSorted[i]]})

train_samples = pd.DataFrame(train_samples)
test_samples = pd.DataFrame(test_samples)

train_samples.to_pickle(open('../data/train_samples_top10.pkl', 'wb'))
test_samples.to_pickle(open('../data/test_samples_top10.pkl', 'wb'))