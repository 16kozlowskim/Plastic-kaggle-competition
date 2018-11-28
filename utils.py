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

    model.add(Dense(1024, input_dim=input_dimensions))
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
    train = pd.read_csv(path_to_data + 'training_set.csv', header=0)
    train_meta = pd.read_csv(path_to_data + 'training_set_metadata.csv', header=0)

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
    scaler = StandardScaler()

    try:
        features = scaler.fit_transform(features)
    except:
        print features.columns[features.isna().any()].tolist()
        features = features.fillna(0)
        features = scaler.fit_transform(features)

    return features

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

def train_mlp(features, wtable, labels, classes, target_map, is_galactic):

    clfs = []
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    oof_preds = np.zeros((len(features), len(classes)))

    for fold_, (trn_, val_) in enumerate(folds.split(features, target_map)):

        checkPoint = ModelCheckpoint("./best.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=0)

        x_train, y_train = features[trn_], labels[trn_]
        x_valid, y_valid = features[val_], labels[val_]

        model = create_model(features.shape[1], len(classes))
        model.compile(loss=mywloss_wrapper(wtable), optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=500, batch_size=256,shuffle=True,verbose=0,callbacks=[checkPoint])

        #plot_loss_acc(history)

        print('Loading Best Model')
        model.load_weights('./best.model')
        # Get predicted probabilities for each class
        oof_preds[val_, :] = model.predict_proba(x_valid,batch_size=256)
        print(multi_weighted_logloss(classes, y_valid, model.predict_proba(x_valid,batch_size=256)))
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
    meta = meta.set_index(preds_df.index)
    preds_df['object_id'] = meta['object_id']
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
