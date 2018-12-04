import pandas as pd
import numpy as np
import utils

path_to_data = ''

data, meta = utils.load_train(path_to_data)

factor = 3

objects = meta['object_id'].values


for i, obj in enumerate(objects):
    aug_data = pd.DataFrame(columns=data.columns)
    df = data.loc[data['object_id'] == obj]
    aug_data = aug_data.append(df)
    for i in range(factor-1):
        for index, row in df.iterrows():
            row[3] = np.random.normal(loc=row[3], scale=row[4])
            row[0] = row[0] + (0.001*(i+1))
            aug_data = aug_data.append(row)
    if i == 0:
        aug_data.to_csv('aug_training_set.csv',  header=True, mode='a', index=False)
    else:
        aug_data.to_csv('aug_training_set.csv',  header=False, mode='a', index=False)


aug_meta = pd.DataFrame(columns=meta.columns)

for index, row in meta.iterrows():
    aug_meta = aug_meta.append(row)
    row_copy = row.copy()
    for i in range(factor-1):
        row_copy[0] = row[0] + (0.001*(i+1))
        aug_meta = aug_meta.append(row_copy)


aug_meta.to_csv('aug_training_set_metadata.csv',  header=True, mode='a', index=False)
