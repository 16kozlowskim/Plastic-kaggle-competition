import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint

from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Activation, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical

from tsfresh.feature_extraction import extract_features

def multi_weighted_logloss(classes, y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    cw = {}
    for i in classes:
        cw[i] = class_weight[i]
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([cw[k] for k in sorted(cw.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

def mywloss_wrapper(wtable):
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
    def mywloss(y_true, y_pred):
        yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
        loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
        return loss

    return mywloss

def plot_loss_acc(history):
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['acc'][1:])
    plt.plot(history.history['val_acc'][1:])
    plt.title('model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()

def create_model(input_dimensions, output_dimensions):
    model = Sequential()

    model.add(Dense(2056, input_dim=input_dimensions))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(output_dimensions, activation='softmax'))

    return model

def load_train(path_to_data):
    train = pd.read_csv(path_to_data + 'aug_training_set.csv', header=0)
    train_meta = pd.read_csv(path_to_data + 'aug_training_set_metadata.csv', header=0)

    return train, train_meta

def gal_split_data(data, meta, data_is_train):

    meta = meta.drop(['ra', 'decl', 'gal_l', 'gal_b', 'hostgal_specz'], axis=1)

    g_meta = meta.loc[meta['hostgal_photoz'] == 0.0000]
    g_meta = g_meta.reset_index(drop=True)
    eg_meta = meta.loc[meta['hostgal_photoz'] != 0.0000]
    eg_meta = eg_meta.reset_index(drop=True)

    g_data = data.loc[data['object_id'].isin(g_meta['object_id'].values)]
    g_data = g_data.reset_index(drop=True)
    eg_data = data.loc[data['object_id'].isin(eg_meta['object_id'].values)]
    eg_data = eg_data.reset_index(drop=True)

    g_target = None
    eg_target = None

    if data_is_train:
        g_target = g_meta['target']
        eg_target = eg_meta['target']
        g_meta=g_meta.drop(['target'], axis=1)
        eg_meta=eg_meta.drop(['target'], axis=1)

    g_meta = g_meta.drop(['hostgal_photoz', 'hostgal_photoz_err', 'distmod'], axis=1)

    if data_is_train:
        return g_data, eg_data, g_meta, eg_meta, g_target, eg_target
    else:
        return g_data, eg_data, g_meta, eg_meta

def preprocess_data(data, meta):

    features = data.groupby(
        ['object_id', 'passband'])['flux', 'flux_err', 'detected'].agg(
        ['mean', 'max', 'min', 'std']).unstack('passband').reset_index(drop=True)

    features = features.join(meta.drop(['object_id'], axis=1))

    features = features.fillna(0)

    return features

#https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135
def feature_engineering(df, meta, isTrain=True):

    fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},{'coeff': 1, 'attr': 'abs'}],'kurtosis' : None, 'skewness' : None}

    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']

    aggs = None

    df_copy = df.copy()

    if 'distmod' in meta.columns:
        min = abs(df['flux'].min())
        df_copy = pd.merge(df_copy, meta, on='object_id')
        df_copy['abs_mag'] = (-2.5*np.log(df_copy['flux'] + min + 1)) - df_copy['distmod']

        aggs = {
            'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
            'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
            'abs_mag': ['min', 'max', 'mean', 'median', 'std','skew'],
            'detected': ['mean'],
            'flux_ratio_sq':['sum','skew'],
            'flux_by_flux_ratio_sq':['sum','skew'],
        }
    else:
        aggs = {
            'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
            'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
            'detected': ['mean'],
            'flux_ratio_sq':['sum','skew'],
            'flux_by_flux_ratio_sq':['sum','skew'],
        }

    agg_df = df_copy.groupby('object_id').agg(aggs)

    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_df.columns = new_columns
    agg_df['flux_diff'] = agg_df['flux_max'] - agg_df['flux_min']
    agg_df['flux_dif2'] = (agg_df['flux_max'] - agg_df['flux_min']) / agg_df['flux_mean']
    agg_df['flux_w_mean'] = agg_df['flux_by_flux_ratio_sq_sum'] / agg_df['flux_ratio_sq_sum']
    agg_df['flux_dif3'] = (agg_df['flux_max'] - agg_df['flux_min']) / agg_df['flux_w_mean']
    # Add more features with
    agg_df_ts = extract_features(df, column_id='object_id', column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=4)
    # Add smart feature that is suggested here https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    # dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    df_det = df[df['detected']==1].copy()

    agg_df_mjd = extract_features(df_det, column_id='object_id', column_value = 'mjd', default_fc_parameters = {'maximum':None, 'minimum':None}, n_jobs=4)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'] - agg_df_mjd['mjd__minimum']
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']
    agg_df_ts = pd.merge(agg_df_ts, agg_df_mjd, on = 'id')
    # tsfresh returns a dataframe with an index name='id'
    agg_df_ts.index.rename('object_id',inplace=True)
    agg_df = pd.merge(agg_df, agg_df_ts, on='object_id')

    agg_df = agg_df.reset_index(drop=True)

    if isTrain:
        agg_df = agg_df.join(meta.drop(['object_id'], axis=1))
    else:
        agg_df = agg_df.join(meta)

    agg_df.fillna(agg_df.mean(), inplace=True)

    return agg_df

def standardize_data(features):
    scaler = StandardScaler()
    data = scaler.fit_transform(features)
    return data

def preprocess_target(target):
    le = LabelEncoder()

    classes = np.sort(target.unique())
    classes_map = le.fit_transform(classes)

    target_map = le.transform(target)
    labels = to_categorical(target_map)

    wtable = np.zeros(len(classes))
    freq = dict()

    for elem in classes_map:
        freq[elem] = 0
    for elem in target_map:
        freq[elem] += 1

    for i in range(len(wtable)):
        wtable[i] = freq[i]/float(target.shape[0])

    return wtable, labels, classes, target_map

def lgb_multi_weighted_logloss_wrapper(classes):

    def lgb_multi_weighted_logloss(y_true, y_preds):
        """
        @author olivier https://www.kaggle.com/ogrellier
        multi logloss for PLAsTiCC challenge
        """
        # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
        # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
        # with Kyle Boone's post https://www.kaggle.com/kyleboone
        class_weight_all = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
        class_weight = {}
        for i in classes:
            class_weight[i] = class_weight_all[i]
        #if len(np.unique(y_true)) > 14:
        #    classes.append(99)
        #    class_weight[99] = 2
        y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

        # Trasform y_true in dummies
        y_ohe = pd.get_dummies(y_true)
        # Normalize rows and limit y_preds to 1e-15, 1-1e-15
        y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
        # Transform to log
        y_p_log = np.log(y_p)
        # Get the log for ones, .values is used to drop the index of DataFrames
        # Exclude class 99 for now, since there is no class99 in the training set
        # we gave a special process for that class
        y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
        # Get the number of positives for each class
        nb_pos = y_ohe.sum(axis=0).values.astype(float)
        # Weight average and divide by the number of positives
        class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
        y_w = y_log_ones * class_arr / nb_pos

        loss = - np.sum(y_w) / np.sum(class_arr)
        return 'wloss', loss, False

    return lgb_multi_weighted_logloss

def train_gbm(features, wtable, labels, classes, target):

    clfs = []
    folds = StratifiedKFold(n_splits=4, shuffle=True)
    oof_preds = np.zeros((len(features), len(classes)))
    lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss',
    'learning_rate': 0.03,
    'subsample': .9,
    'colsample_bytree': 0.5,
    'reg_alpha': .01,
    'reg_lambda': .01,
    'min_split_gain': 0.01,
    'min_child_weight': 10,
    'n_estimators': 1000,
    'silent': -1,
    'verbose': -1,
    'max_depth': 3
    }

    w = target.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}

    for fold_, (trn_, val_) in enumerate(folds.split(target, target)):
        x_train, y_train = features.values[trn_], target[trn_]
        x_valid, y_valid = features.values[val_], target[val_]
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_metric=lgb_multi_weighted_logloss_wrapper(classes),
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=y_train.map(weights)
        )

        oof_preds[val_, :] = model.predict_proba(x_valid, num_iteration=model.best_iteration_)
        clfs.append(model)

    print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(classes,labels,oof_preds))

    return clfs


def train_mlp(features, wtable, labels, classes, target_map, is_galactic):

    clfs = []
    folds = StratifiedKFold(n_splits=4, shuffle=True)
    oof_preds = np.zeros((len(features), len(classes)))

    for fold_, (trn_, val_) in enumerate(folds.split(features, target_map)):

        checkPoint = ModelCheckpoint("./best.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=1)

        x_train, y_train = features[trn_], labels[trn_]
        x_valid, y_valid = features[val_], labels[val_]

        model = create_model(features.shape[1], len(classes))
        model.compile(loss=mywloss_wrapper(wtable), optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=400, batch_size=512,shuffle=True,verbose=1,callbacks=[checkPoint])

        #plot_loss_acc(history)

        print('Loading Best Model')
        model.load_weights('./best.model')
        # Get predicted probabilities for each class
        oof_preds[val_, :] = model.predict_proba(x_valid,batch_size=512)
        print(multi_weighted_logloss(classes, y_valid, model.predict_proba(x_valid,batch_size=512)))
        clfs.append(model)

    print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(classes,labels,oof_preds))

    galactic = None
    if is_galactic:
        galactic = 'g'
    else:
        galactic = 'eg'

    for i in range(len(clfs)):
        filepath = './' + galactic + str(i) + '.h5'
        clfs[i].save(filepath)


def train_conv_net(features, wtable, labels, classes, target_map, is_galactic):

    clfs = []
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    oof_preds = np.zeros((features.shape[0], len(classes)))

    for fold_, (trn_, val_) in enumerate(folds.split(features, target_map)):

        checkPoint = ModelCheckpoint("./best.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=1)

        x_train, y_train = features[trn_], labels[trn_]
        x_valid, y_valid = features[val_], labels[val_]

        model = create_conv_model(features.shape[1:], len(classes))
        model.compile(loss=mywloss_wrapper(wtable), optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=100, batch_size=128,shuffle=True,verbose=1,callbacks=[checkPoint])

        #plot_loss_acc(history)

        print('Loading Best Model')
        model.load_weights('./best.model')
        # Get predicted probabilities for each class
        oof_preds[val_, :] = model.predict_proba(x_valid,batch_size=128)
        print(multi_weighted_logloss(classes, y_valid, model.predict_proba(x_valid,batch_size=128)))
        clfs.append(model)

    print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(classes,labels,oof_preds))

    galactic = None
    if is_galactic:
        galactic = 'g'
    else:
        galactic = 'eg'

    for i in range(len(clfs)):
        filepath = './' + galactic + str(i) + '.h5'
        clfs[i].save(filepath)


def get_class_names(path_to_data):
    sample_sub = pd.read_csv(path_to_data+'sample_submission.csv')
    class_names = list(sample_sub.columns[1:-1])
    return class_names

def load_models(count, is_galactic, wtable):
    clfs = []

    galactic = None
    if is_galactic:
        galactic = 'g'
    else:
        galactic = 'eg'

    for i in range(count):
        filepath = './' + galactic + str(i) + '.h5'
        clfs.append(load_model(filepath, custom_objects={'mywloss': mywloss_wrapper(wtable)}))

    return clfs

def predict(clfs, features, folds):
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(features) / folds.n_splits
        else:
            preds += clf.predict_proba(features) / folds.n_splits
    return preds

def predict_99(preds):
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])
    return preds_99

def g_class_names():
    return ['class_' + str(i) for i in [6,16,53,65,92]]
def eg_class_names():
    return ['class_' + str(i) for i in [15,42,52,62,64,67,88,90,95]]

def store_preds(preds, class_names, preds_99, meta):
    preds_df = pd.DataFrame(preds, columns=class_names)
    #meta = meta.set_index(preds_df.index)
    preds_df['object_id'] = meta#['object_id']
    preds_df['class_99'] = 0.14 * preds_99 / np.mean(preds_99)
    return preds_df

def get_wtables(path_to_data):
    train, train_meta = load_train(path_to_data)
    g_train, eg_train, g_meta, eg_meta, g_target, eg_target = gal_split_data(train, train_meta, True)
    g_wtable, g_labels, g_classes, g_target_map = preprocess_target(g_target)
    eg_wtable, eg_labels, eg_classes, eg_target_map = preprocess_target(eg_target)
    return g_wtable, eg_wtable


def conv_preprocess_data(data, type, columns):
    columns = columns - 1

    ts_mod = data[['object_id', 'mjd', 'passband', type]].copy()
    dif = ts_mod['mjd'].max() - ts_mod['mjd'].min()
    div = dif/columns

    ts_mod['mjd_d5'] = (ts_mod['mjd'] / div).astype(int)

    ts_mod = ts_mod.groupby(['object_id', 'mjd_d5', 'passband'])[type].mean().reset_index()

    # pivotting
    ts_piv = pd.pivot_table(ts_mod,
                            index='object_id',
                            columns=['mjd_d5', 'passband'],
                            values=type,
                            dropna=False)

    # resetting column index to fill mjd_d5 gaps
    t_min, t_max = ts_piv.columns.levels[0].min(), ts_piv.columns.levels[0].max()
    t_range = range(t_min, t_max + 1)
    mux = pd.MultiIndex.from_product([list(t_range), list(range(6))],
                                     names=['mjd_d5', 'passband'])
    ts_piv = ts_piv.reindex(columns=mux).stack('passband')

    ts_piv = ts_piv.fillna(0)

    return ts_piv

def split_conv(df):
    object_count = df.shape[0]/6
    objects = []
    for i in range(object_count):
        objects.append(df.iloc[range(i*6,6+(i*6))].values)
    return objects

def create_conv_model(input_shape, output_dimensions):
    model = Sequential()

    model.add(Conv2D(filters=8, kernel_size=(2,4), data_format='channels_first', padding='same', input_shape=input_shape, kernel_regularizer = l2()))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=8, kernel_size=(2,4), data_format='channels_first', padding='same', kernel_regularizer = l2()))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,4), data_format='channels_first'))
    '''
    model.add(Conv2D(filters=16, kernel_size=3, data_format='channels_first', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=16, kernel_size=3, data_format='channels_first', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(1,2), data_format='channels_first'))
    '''
    model.add(Flatten(data_format='channels_first'))

    model.add(Dense(output_dimensions*4))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(output_dimensions*2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(output_dimensions))
    model.add(Activation('softmax'))


    return model

def conv_data(data, columns):
    conv_flux = conv_preprocess_data(data, 'flux', columns)
    conv_flux_err = conv_preprocess_data(data, 'flux_err', columns)

    conv_flux = split_conv(conv_flux)
    conv_flux_err = split_conv(conv_flux_err)

    conv = []

    for i, j in zip(conv_flux, conv_flux_err):
        conv.append(np.array([i, j]))

    return np.array(conv)
