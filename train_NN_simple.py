import utils

#path_to_data = '/courses/cs342/Assignment2/'
path_to_data = ''

train, train_meta = utils.load_train(path_to_data)

g_train, eg_train, g_meta, eg_meta, g_target, eg_target = utils.gal_split_data(train, train_meta, True)

g_features = utils.feature_engineering(g_train, g_meta)
g_wtable, g_labels, g_classes, g_target_map = utils.preprocess_target(g_target)
g_features = utils.standardize_data(g_features)
utils.train_mlp(g_features, g_wtable, g_labels, g_classes, g_target_map, True)


eg_features = utils.feature_engineering(eg_train, eg_meta)
eg_wtable, eg_labels, eg_classes, eg_target_map = utils.preprocess_target(eg_target)
eg_features = utils.standardize_data(eg_features)
utils.train_mlp(eg_features, eg_wtable, eg_labels, eg_classes, eg_target_map, False)
