import utils
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

#path_to_data = '/courses/cs342/Assignment2/'
path_to_data = ''

train, train_meta = utils.load_train(path_to_data)

g_train, eg_train, g_meta, eg_meta, g_target, eg_target = utils.gal_split_data(train, train_meta, True)

g_features = utils.feature_engineering(g_train, g_meta)

eg_features = utils.feature_engineering(eg_train, eg_meta)

g_clf = RandomForestClassifier(n_estimators=5000, max_depth=25, class_weight='balanced')
g_clf.fit(g_features, g_target)

eg_clf = RandomForestClassifier(n_estimators=5000, max_depth=25, class_weight='balanced')
eg_clf.fit(eg_features, eg_target)

g_features = pd.read_csv(path_to_data + 'test_g_features.csv', header=0)

meta = g_features['object_id']
g_features = g_features.drop(['object_id'], axis=1)

g_preds = g_clf.predict_proba(g_features)

g_preds_99 = utils.predict_99(g_preds)
g_preds_df = utils.store_preds(g_preds, utils.g_class_names(), g_preds_99, meta)

for i in utils.eg_class_names():
    g_preds_df[i] = 0
g_preds_df = g_preds_df.reindex_axis(['object_id']+utils.g_class_names()+utils.eg_class_names()+['class_99'], axis=1)

g_preds_df.to_csv('predictions.csv',  header=True, mode='a', index=False)

for i_c, data_chunk in enumerate(pd.read_csv(path_to_data + 'test_eg_features.csv', chunksize = 500000, iterator = True)):
    #eg_features = pd.read_csv(path_to_data + 'test_eg_features.csv', header=0)
    meta = data_chunk['object_id'].reset_index(drop=True)
    data_chunk = data_chunk.drop(['object_id'], axis=1).reset_index(drop=True)
    eg_preds = eg_clf.predict_proba(data_chunk)
    eg_preds_99 = utils.predict_99(eg_preds)

    eg_preds_df = utils.store_preds(eg_preds, utils.eg_class_names(), eg_preds_99, meta)
    for i in utils.g_class_names():
        eg_preds_df[i] = 0
    eg_preds_df = eg_preds_df.reindex_axis(['object_id']+utils.g_class_names()+utils.eg_class_names()+['class_99'], axis=1)

    eg_preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False)
