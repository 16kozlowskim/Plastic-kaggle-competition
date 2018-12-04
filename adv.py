import numpy as np
import pandas as pd
max = 0

straddler = None

for i_c, data_chunk in enumerate(pd.read_csv('test_set.csv', chunksize = 5000000, iterator = True)):
    if i_c != 0:
        data_chunk = pd.concat([straddler, data_chunk], ignore_index=True)

    arr = data_chunk['object_id'].unique()
    straddler = data_chunk.loc[data_chunk['object_id'] == arr[len(arr)-1]]
    data_chunk = data_chunk[data_chunk.object_id != arr[len(arr)-1]]
    data_chunk = data_chunk.reset_index(drop=True)

    objects = np.unique(data_chunk['object_id'].values)
    for obj in objects:
        df = data_chunk.loc[data_chunk['object_id'] == obj]
        if df.shape[0] > max:
            max = df.shape[0]
    print 'i'
data_chunk = straddler

objects = np.unique(data_chunk['object_id'].values)
for obj in objects:
    df = data_chunk.loc[data_chunk['object_id'] == obj]
    if df.shape[0] > max:
        max = df.shape[0]

print max
