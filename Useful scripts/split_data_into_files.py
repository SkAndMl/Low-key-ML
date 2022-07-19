import os
import pandas as pd


def split_data_into_files(data,file_prefix,dir_name="dataset",n_files=10):
    
    path = os.path.join(os.curdir,dir_name)
    if os.path.exists(path=path):
        pass
    else:
        os.mkdir(path=path)
        print("\nNew directory created")

    filepaths = []
    for i in range(n_files):
        lower_bound = int(i*len(data)/n_files)
        upper_bound = int((i+1)*len(data)/n_files)
        data_temp = data.iloc[lower_bound:upper_bound].copy()
        data_temp.to_csv(f"{path}/{file_prefix}_{i}.csv",index=False)
        filepaths.append(f"{path}/{file_prefix}_{i}.csv")
    
    return filepaths