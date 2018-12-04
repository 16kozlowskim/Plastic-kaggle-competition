import numpy as np
import pandas as pd


for i_c, data_chunk in enumerate(pd.read_csv('test_eg_features.csv', chunksize = 5000000, iterator = True)):

    data_chunk['max_abs_mag'] = -2.5*np.log(data_chunk['flux_max'] + data_chunk['flux_min'].abs()) - data_chunk['distmod']

    if i_c ==0:
        data_chunk.to_csv('test2_eg_features.csv',  header=True, mode='a', index=False)
    else:
        data_chunk.to_csv('test2_eg_features.csv',  header=False, mode='a', index=False)
