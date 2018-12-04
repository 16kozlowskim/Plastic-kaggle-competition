import time
import sys
import utils
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.utils import to_categorical

chunks = 5000000
path_to_data=''
test_meta = pd.read_csv(path_to_data+'test_set_metadata.csv')
straddler = None

for i_c, data_chunk in enumerate(pd.read_csv(path_to_data + 'test_set.csv', chunksize = chunks, iterator = True)):
    if i_c != 0:
        data_chunk = pd.concat([straddler, data_chunk], ignore_index=True)

    arr = data_chunk['object_id'].unique()
    straddler = data_chunk.loc[data_chunk['object_id'] == arr[len(arr)-1]]
    data_chunk = data_chunk[data_chunk.object_id != arr[len(arr)-1]]
    data_chunk = data_chunk.reset_index(drop=True)

    meta_chunk = test_meta[test_meta['object_id'].isin(data_chunk['object_id'].unique())]
    meta_chunk = meta_chunk.reset_index(drop=True)

    g_data, eg_data, g_meta, eg_meta = utils.gal_split_data(data_chunk, meta_chunk, False)

    g_features = None
    eg_features = None

    if g_meta.shape[0] > 0:
        #make meta not drop object_id in the feature engineering function
        g_features = utils.feature_engineering(g_data, g_meta, False)
        if i_c == 0:
            g_features.to_csv('test_g_features.csv',  header=True, mode='a', index=False)
        else:
            g_features.to_csv('test_g_features.csv',  header=False, mode='a', index=False)

    if eg_meta.shape[0] > 0:

        eg_features = utils.feature_engineering(eg_data, eg_meta, False)
        if i_c == 0:
            eg_features.to_csv('test_eg_features.csv',  header=True, mode='a', index=False)
        else:
            eg_features.to_csv('test_eg_features.csv',  header=False, mode='a', index=False)

data_chunk = straddler

meta_chunk = test_meta[test_meta['object_id'].isin(data_chunk['object_id'].unique())]
meta_chunk = meta_chunk.reset_index(drop=True)


g_data, eg_data, g_meta, eg_meta = utils.gal_split_data(data_chunk, meta_chunk, False)

g_features = None
eg_features = None

if g_meta.shape[0] > 0:
    #make meta not drop object_id in the feature engineering function
    g_features = utils.feature_engineering(g_data, g_meta, False)
    g_features.to_csv('test_g_features.csv',  header=False, mode='a', index=False)

if eg_meta.shape[0] > 0:

    eg_features = utils.feature_engineering(eg_data, eg_meta, False)
    eg_features.to_csv('test_eg_features.csv',  header=False, mode='a', index=False)
