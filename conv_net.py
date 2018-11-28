import utils
import numpy as np
import pandas as pd
from scipy.stats import skew
from matplotlib import pylab as plt

from sklearn.model_selection import StratifiedKFold

import gc
import time
import warnings

warnings.simplefilter(action = 'ignore')

import tensorflow as tf
from tensorflow import keras

path_to_data = ''
columns = 256

train, train_meta = utils.load_train(path_to_data)

g_data, eg_data, g_meta, eg_meta, g_target, eg_target = utils.gal_split_data(train, train_meta, True)

g_conv_flux = utils.conv_preprocess_data(g_data, 'flux', 256)
g_conv_flux_err = utils.conv_preprocess_data(g_data, 'flux_err', 256)

g_conv_flux = utils.split_conv(g_conv_flux)
g_conv_flux_err = utils.split_conv(g_conv_flux_err)

g_conv = []

for i, j in zip(g_conv_flux, g_conv_flux_err):
    g_conv.append([i, j])


'''
def get_stats(df):
    groups = df.groupby('passband')
    res = groups['mjd'].apply(np.count_nonzero).values
    res = np.vstack((res, groups['mjd'].apply(np.asarray).apply(lambda x: np.median(x[1:] - x[:-1]))))

    return np.transpose(res)

def to_binned_timeseries(ndar, step):
    warnings.simplefilter(action = 'ignore')

    # the first time for object
    start = np.min(ndar[:, 0])
    # sequence duration for object
    mjd_lendth = np.max(ndar[:, 0]) - start
    # count of bins for object timeseries
    timeseries_lendth = int(mjd_lendth / step) + 1
    # matrix for counts in each bin for each row
    cnt = np.zeros((6, timeseries_lendth))
    # matrix for result with 3 channels: flux, flux_err, detected
    # corresponds to data_format = 'channels_last' for CPU
    res = np.zeros((6, timeseries_lendth, 3))

    # loop for rows in sourse array for calculating summs
    for i in range(ndar.shape[0]):
        row = ndar[i, :]
        col_num = int((row[0] - start) / step)
        cnt[int(row[1]), col_num] += 1
        res[int(row[1]), col_num, 0] += row[2]
        res[int(row[1]), col_num, 1] += row[3]
        res[int(row[1]), col_num, 2] += row[4]

    # get mean values exclude nans
    res[:, :, 0] /= cnt
    res[:, :, 1] /= cnt
    res[:, :, 2] /= cnt

    # normalizing flux channels by rows
    for channel in range(2):
        means = np.reshape([np.mean(res[i, ~np.isnan(res[i, :, channel]), channel]) for i in range(6)]*timeseries_lendth,
                           (6, timeseries_lendth), order = 'F')
        stds = np.reshape([np.std(res[i, ~np.isnan(res[i, :, channel]), channel]) for i in range(6)]*timeseries_lendth,
                          (6, timeseries_lendth), order = 'F')
        res[:, :, channel] = (res[:, :, channel] - means) / stds

    # replacing nans to zeros
    res = np.nan_to_num(res)

    return res

path_to_data = ''

train, train_meta = utils.load_train(path_to_data)

g_data, eg_data, g_meta, eg_meta, g_target, eg_target = utils.gal_split_data(train, train_meta, True)

MAX_LENGTH = -1
for obj in g_meta['object_id'].values:
    ndar = g_data[g_data['object_id'] == obj][['mjd', 'passband', 'flux', 'flux_err', 'detected']]
    stats = get_stats(ndar)
    data = to_binned_timeseries(ndar.values, np.median(stats[:, 1]))
    print data.T
    if data.shape[1] > MAX_LENGTH:
        MAX_LENGTH = data.shape[1]
    import sys
    sys.exit()

print('Count of columns in `image`:', MAX_LENGTH)

'''
