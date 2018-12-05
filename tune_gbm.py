import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lightgbm as lgb
import utils

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint

from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Activation, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from hyperopt import hp, tpe, STATUS_OK, Trials, fmin
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

from tsfresh.feature_extraction import extract_features

path_to_data=''

params = {
'boosting_type': 'gbdt',
'objective': 'multiclass',
'metric': 'multi_logloss',
'verbose': -1,
'silent': -1,
'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
'subsample': hp.uniform ('subsample', 0.8, 1),
'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
'max_depth': scope.int(hp.quniform('max_depth', 3, 25, 1)),
'n_estimators': scope.int(hp.quniform('n_estimators', 100, 10000, 1)),
'num_leaves': scope.int(hp.quniform('num_leaves', 4, 10000, 1)),
'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 15, 25, 1)),
'feature_fraction': hp.uniform('feature_fraction', 0.7, 0.9),
'lambda_l1': hp.uniform('lambda_l1', 0, 0.015),
'lambda_l2': hp.uniform('lambda_l2', 0, 0.015),
'min_gain_to_split': hp.uniform('min_gain_to_split', 0, 0.02),
}

def objective_wrapper(features, wtable, labels, classes, target):
    def objective(space):
        w = target.value_counts()
        weights = {i : np.sum(w) / w[i] for i in w.index}

        folds = StratifiedKFold(n_splits=5, shuffle=True)
        oof_preds = np.zeros((len(features), len(classes)))

        for fold_, (trn_, val_) in enumerate(folds.split(target, target)):
            x_train, y_train = features.values[trn_], target[trn_]
            x_valid, y_valid = features.values[val_], target[val_]
            model = lgb.LGBMClassifier(num_classes=len(classes), **space)
            model.fit(
                x_train, y_train,
                eval_set=[(x_train, y_train), (x_valid, y_valid)],
                eval_metric=utils.lgb_multi_weighted_logloss_wrapper(classes),
                verbose=-1,
                early_stopping_rounds=50,
                sample_weight=y_train.map(weights)
            )

            oof_preds[val_, :] = model.predict_proba(x_valid, num_iteration=model.best_iteration_)

        return{'loss': utils.multi_weighted_logloss(classes, labels, oof_preds), 'status': STATUS_OK}
    return objective

train, train_meta = utils.load_train(path_to_data)

g_train, eg_train, g_meta, eg_meta, g_target, eg_target = utils.gal_split_data(train, train_meta, True)

g_features = utils.feature_engineering(g_train, g_meta)
g_wtable, g_labels, g_classes, g_target_map = utils.preprocess_target(g_target)

trials = Trials()
best1 = fmin(fn=objective_wrapper(g_features, g_wtable, g_labels, g_classes, g_target),
            space=params,
            algo=tpe.suggest,
            max_evals=200, # change
            trials=trials)

eg_features = utils.feature_engineering(eg_train, eg_meta)
eg_wtable, eg_labels, eg_classes, eg_target_map = utils.preprocess_target(eg_target)

trials = Trials()
best2 = fmin(fn=objective_wrapper(eg_features, eg_wtable, eg_labels, eg_classes, eg_target),
            space=params,
            algo=tpe.suggest,
            max_evals=300, # change
            trials=trials)

print best1, best2
