import utils
import gc
import time
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Activation, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical

warnings.simplefilter(action = 'ignore')


#path_to_data = '/courses/cs342/Assignment2/'
path_to_data = ''
split_count = 3
columns = 64

start = time.time()
chunks = 5000000

test_meta = pd.read_csv(path_to_data+'test_set_metadata.csv')

g_wtable, eg_wtable = utils.get_wtables(path_to_data)

g_clfs = utils.load_models(split_count, True, g_wtable)
eg_clfs = utils.load_models(split_count, False, eg_wtable)

folds = StratifiedKFold(n_splits = split_count, shuffle = True)

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

    preds_df = None

    g_preds_df = None

    eg_preds_df = None

    if g_meta.shape[0] > 0:
        g_conv_data = utils.conv_data(g_data, columns)
        g_preds = utils.predict(g_clfs, g_conv_data, folds)
        g_preds_99 = utils.predict_99(g_preds)
        g_preds_df = utils.store_preds(g_preds, utils.g_class_names(), g_preds_99, g_meta)
        for i in utils.eg_class_names():
            g_preds_df[i] = 0
        g_preds_df = g_preds_df.reindex_axis(['object_id']+utils.g_class_names()+utils.eg_class_names()+['class_99'], axis=1)
        preds_df = g_preds_df

    if eg_meta.shape[0] > 0:
        eg_conv_data = utils.conv_data(eg_data, columns)
        eg_preds = utils.predict(eg_clfs, eg_conv_data, folds)
        eg_preds_99 = utils.predict_99(eg_preds)
        eg_preds_df = utils.store_preds(eg_preds, utils.eg_class_names(), eg_preds_99, eg_meta)
        for i in utils.g_class_names():
            eg_preds_df[i] = 0
        eg_preds_df = eg_preds_df.reindex_axis(['object_id']+utils.g_class_names()+utils.eg_class_names()+['class_99'], axis=1)
        preds_df = eg_preds_df

    if g_meta.shape[0] > 0 and eg_meta.shape[0] > 0:
        preds_df = pd.concat([g_preds_df, eg_preds_df], ignore_index=True)

    if i_c == 0:
        preds_df.to_csv('predictions.csv',  header=True, mode='a', index=False)
    else:
        preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False)

    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

data_chunk = straddler

meta_chunk = test_meta[test_meta['object_id'].isin(data_chunk['object_id'].unique())]
meta_chunk = meta_chunk.reset_index(drop=True)

g_data, eg_data, g_meta, eg_meta = utils.gal_split_data(data_chunk, meta_chunk, False)

preds_df = None

g_preds_df = None

eg_preds_df = None

if g_meta.shape[0] > 0:
    g_conv_data = utils.conv_data(g_data, columns)
    g_preds = utils.predict(g_clfs, g_conv_data, folds)
    g_preds_99 = utils.predict_99(g_preds)
    g_preds_df = utils.store_preds(g_preds, utils.g_class_names(), g_preds_99, g_meta)
    for i in utils.eg_class_names():
        g_preds_df[i] = 0
    g_preds_df = g_preds_df.reindex_axis(['object_id']+utils.g_class_names()+utils.eg_class_names()+['class_99'], axis=1)
    preds_df = g_preds_df

if eg_meta.shape[0] > 0:
    eg_conv_data = utils.conv_data(eg_data, columns)
    eg_preds = utils.predict(eg_clfs, eg_conv_data, folds)
    eg_preds_99 = utils.predict_99(eg_preds)
    eg_preds_df = utils.store_preds(eg_preds, utils.eg_class_names(), eg_preds_99, eg_meta)
    for i in utils.g_class_names():
        eg_preds_df[i] = 0
    eg_preds_df = eg_preds_df.reindex_axis(['object_id']+utils.g_class_names()+utils.eg_class_names()+['class_99'], axis=1)
    preds_df = eg_preds_df

if g_meta.shape[0] > 0 and eg_meta.shape[0] > 0:
    preds_df = pd.concat([g_preds_df, eg_preds_df], ignore_index=True)

preds_df.to_csv('predictions.csv', header=False, mode='a', index=False)
