import pandas as pd
import numpy as np
import utils

path_to_data = ''
data, meta = utils.load_train(path_to_data)

objects = meta['object_id'].values

for obj in objects:
    df = data.loc[data['object_id'] == obj]
    arr =  utils.conv_preprocess_data(df, 355feature)
    print pd.DataFrame(arr[0][0])
    break
