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


path_to_data = ''
columns = 128

train, train_meta = utils.load_train(path_to_data)

g_data, eg_data, g_meta, eg_meta, g_target, eg_target = utils.gal_split_data(train, train_meta, True)

g_conv_data = utils.conv_data(g_data, columns)
eg_conv_data = utils.conv_data(eg_data, columns)

g_wtable, g_labels, g_classes, g_target_map = utils.preprocess_target(g_target)
eg_wtable, eg_labels, eg_classes, eg_target_map = utils.preprocess_target(eg_target)

utils.train_conv_net(g_conv_data, g_wtable, g_labels, g_classes, g_target_map, True)
utils.train_conv_net(eg_conv_data, eg_wtable, eg_labels, eg_classes, eg_target_map, False)
